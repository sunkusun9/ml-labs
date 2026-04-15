import re
import os
import sys
import uuid
import pickle as pkl
import shutil
import traceback
import warnings
from pathlib import Path

import pandas as pd
from cachetools import LRUCache

from sklearn.model_selection import ShuffleSplit

from ._data_wrapper import wrap, unwrap, DataWrapperProvider
from ._flow import TrainDataFlow
from ._store import NodeStore
from ._describer import desc_spec, desc_status, desc_obj_vars
from ._logger import DefaultLogger

from ._pipeline import Pipeline
from ._node_processor import resolve_columns
from ._connector import Connector
from .collector import Collector, MetricCollector, StackingCollector, ModelAttrCollector, SHAPCollector, OutputCollector, ProcessCollector
from ._trainer import Trainer


class OuterFold:
    """One outer fold: test indices, base path, and per-inner-fold TrainDataFlows.

    Serializes test_idx, path, and TrainDataFlow list.
    DataWrapperProvider inside each TrainDataFlow persists only indices — DataWrapper is transient.

    Call set_data(data) to re-inject DataWrapper and cache after load.
    """

    def __init__(self, outer_idx, path, data, test_idx, train_idx_list, cache=None, aug_data=None):
        self.outer_idx = outer_idx
        self.path = Path(path)
        self.test_idx = test_idx
        self.data = data
        self.train_data_flows = [
            TrainDataFlow(
                path=self.path / str(j),
                data_source=DataWrapperProvider(data, train_idx, valid_idx=valid_idx, aug_data=aug_data),
                cache=cache,
                outer_idx=outer_idx,
                inner_idx=j,
            )
            for j, (train_idx, valid_idx) in enumerate(train_idx_list)
        ]
        self.artifact_stores = [
            NodeStore(path=self.path / str(j))
            for j in range(len(train_idx_list))
        ]

    def set_data(self, data, cache=None, aug_data=None):
        self.data = data
        for flow in self.train_data_flows:
            flow.data_source.set_data(data, aug_data)
            if cache is not None:
                flow.cache = cache

    def get_data(self, data, edges, inner_idx=0):
        return self.train_data_flows[inner_idx].get_data(data, edges)

    def get_test_data(self, edges, inner_idx=0):
        test_source = self.data.iloc(self.test_idx)
        return self.get_data(test_source, edges, inner_idx)



def _get_data_size(data):
    if data is None:
        return 0
    if isinstance(data, list):
        return sum(_get_data_size(item) for item in data)
    if isinstance(data, tuple):
        return sum(_get_data_size(item) for item in data)
    if hasattr(data, 'nbytes'):
        return data.nbytes
    if hasattr(data, 'memory_usage'):
        return data.memory_usage(deep=True).sum()
    return sys.getsizeof(data)

class DataCache():
    """LRU cache for Stage node outputs, keyed by ``(node, type, fold_idx)``.

    Capacity is measured in bytes using ``nbytes`` / ``memory_usage``.

    Args:
        maxsize (int): Maximum cache capacity in bytes. Default 4 GB.
    """

    def __init__(self, maxsize=4 * 1024 ** 3):  # 4GB 기본값
        self.cache_dic = LRUCache(maxsize=maxsize, getsizeof=_get_data_size)

    def get_data(self, node, outer_idx, inner_idx, typ):
        key = (node, outer_idx, inner_idx, typ)
        return self.cache_dic.get(key, None)

    def put_data(self, node, outer_idx, inner_idx, typ, data):
        key = (node, outer_idx, inner_idx, typ)
        self.cache_dic[key] = data

    def clear(self):
        self.cache_dic.clear()

    def clear_nodes(self, nodes):
        node_set = set(nodes)
        keys_to_delete = [k for k in self.cache_dic.keys() if k[0] in node_set]
        for k in keys_to_delete:
            del self.cache_dic[k]

class Experimenter():
    """Executes and manages a Pipeline experiment on a single dataset.

    Splits data using *sp* (outer) and optionally *sp_v* (inner), then runs
    Stage builds and Head experiments fold-by-fold.

    Args:
        data: Input dataset (pandas DataFrame, polars DataFrame, or numpy array).
        path (str | Path): Directory for persisting experiment artifacts.
        data_names (list[str], optional): Column names override.
        sp: Outer splitter (sklearn splitter API). Default
            ``ShuffleSplit(n_splits=1, random_state=1)``.
        sp_v: Inner splitter for nested cross-validation. ``None`` disables.
        splitter_params (dict, optional): Maps splitter keyword args to column
            names in *data*, e.g. ``{'y': 'target'}``.
        title (str, optional): Human-readable experiment title.
        data_key (str, optional): Identifier verified on :meth:`load` to prevent
            data mismatch.
        cache_maxsize (int): Stage output cache size in bytes. Default 4 GB.
        logger: Logger instance. Default ``DefaultLogger(level=['info', 'progress'])``.

    Attributes:
        pipeline (Pipeline): The pipeline being experimented on.
        node_objs (dict): ``{node_name: StageObj}`` — stage nodes only; head node state is checked on-demand via :meth:`get_status`.
        cache (DataCache): Shared LRU cache.
        collectors (dict): Registered :class:`~mllabs.collector.Collector` instances.
        trainers (dict): Registered :class:`~mllabs._trainer.Trainer` instances.
        status (str): ``'open'`` or ``'closed'``.
    """

    def __init__(
            self, data, path, data_names = None, sp = ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None, cache_maxsize=4 * 1024 ** 3,
            logger = DefaultLogger(level=['info', 'progress']), aug_data=None, _save=True
        ):
        self.cache_maxsize = cache_maxsize
        self.logger = logger
        self.path = Path(path)
        if not os.path.exists(path):
            self.path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Created directory: {self.path}")
        data_native = data
        self.data = wrap(data)
        self.aug_data = wrap(aug_data) if aug_data is not None else None
        self.title = title
        self.data_key = data_key
        self.sp = sp
        self.sp_v = sp_v
        self.splitter_params = splitter_params if splitter_params is not None else {}
        self.exp_id = str(uuid.uuid4())

        split_params = {}
        if data_names is None:
            data_names = self.data.get_columns()
        for k, v in self.splitter_params.items():
            split_params[k] = unwrap(self.data.select_columns(v))

        raw_splits = []
        for outer_train_idx, test_idx in sp.split(data_native, **split_params):
            if sp_v is not None:
                train_data = self.data.iloc(outer_train_idx)
                train_data_native = unwrap(train_data)
                inner_split_params = {'X': train_data_native}
                for k, v in self.splitter_params.items():
                    inner_split_params[k] = unwrap(train_data.select_columns(v))
                inner_folds = [
                    (outer_train_idx[train_idx], outer_train_idx[valid_idx])
                    for train_idx, valid_idx in sp_v.split(**inner_split_params)
                ]
            else:
                inner_folds = [(outer_train_idx, None)]
            raw_splits.append((test_idx, inner_folds))

        self.pipeline = Pipeline()
        self.cache = DataCache(maxsize=cache_maxsize)

        self.outer_folds = [
            OuterFold(
                outer_idx=i,
                path=self.path / '__folds' / str(i),
                data=self.data,
                test_idx=test_idx,
                train_idx_list=inner_folds,
                cache=self.cache,
                aug_data=self.aug_data,
            )
            for i, (test_idx, inner_folds) in enumerate(raw_splits)
        ]
        self.collectors = {}
        self.trainers = {}
        self.status = "open"
        if _save:
            self._save()

    def _check_open(self):
        """상태가 open인지 확인하고, 아니면 에러 발생"""
        if self.status != "open":
            raise RuntimeError(f"Experimenter is '{self.status}'. Only 'open' status allows modifications.")

    def open(self):
        """Experimenter를 open 상태로 변경"""
        self.status = "open"
        self._save()
        self.logger.info("Experimenter status changed to 'open'")

    def close(self):
        """Experimenter를 close 상태로 변경"""
        self.status = "close"
        self._save()
        self.logger.info("Experimenter status changed to 'close'")

    @staticmethod
    def create(data, path, data_names=None, sp=ShuffleSplit(n_splits=1, random_state=1), sp_v=None,
            splitter_params=None, title=None, data_key=None, cache_maxsize=4 * 1024 ** 3,
            logger = DefaultLogger(level=['info', 'progress']), aug_data=None):

        if os.path.exists(path):
            raise RuntimeError(f"Exists: {path}")
        return Experimenter(
            data, path, data_names, sp=sp, sp_v=sp_v, splitter_params=splitter_params,
            title=title, data_key=data_key, cache_maxsize=cache_maxsize, logger=logger,
            aug_data=aug_data)

    def get_n_splits(self):
        return len(self.outer_folds)

    def get_n_splits_inner(self):
        return len(self.outer_folds[0].train_data_flows)

    def get_collector(self, name):
        return self.collectors.get(name)

    def remove_collector(self, name):
        if name in self.collectors:
            collector_path = self.path / '__collector' / name
            if collector_path.exists():
                shutil.rmtree(collector_path)
            del self.collectors[name]
            self._save()

    def add_collector(self, collector, exist = 'skip'):
        """Register a Collector and immediately collect from built Head nodes.

        Args:
            collector (Collector): Collector instance to register.
            exist (str): ``'skip'`` (default) returns existing if already registered;
                ``'error'`` raises; ``'replace'`` removes the existing collector and
                registers the new one from scratch.

        Returns:
            Collector: The registered collector.
        """
        if collector.name in self.collectors:
            if exist == 'skip':
                return self.collectors[collector.name]
            elif exist == 'error':
                raise RuntimeError("")
            elif exist == 'replace':
                self.remove_collector(collector.name)

        self._check_open()
        collector.path = self.path / '__collector' / collector.name
        collector._setup(
            len(self.outer_folds), len(self.outer_folds[0].train_data_flows)
        )
        collector.save()
        self.collectors[collector.name] = collector
        self.collect(collector)
        self._save()
        return collector

    def get_collect_status(self, collector, nodes=None):
        if isinstance(collector, str):
            collector = self.collectors[collector]
        all_node_names = self.pipeline.get_node_names(nodes)
        head_nodes = [
            n for n in all_node_names
            if n is not None and self.pipeline.get_node_attrs(n).get('role') == 'head'
            and collector.connector.match(self.pipeline.get_node_attrs(n))
        ]
        result = {}
        for node in head_nodes:
            if collector.has_node(node):
                result[node] = 'collected'
            else:
                node_status = self.get_status(node)
                if node_status == 'finalized':
                    result[node] = 'finalized'
                elif node_status == 'error':
                    result[node] = 'error'
                else:
                    result[node] = 'not_collected'
        return result

    def get_trainer(self, name):
        return self.trainers.get(name)

    def remove_trainer(self, name):
        if name in self.trainers:
            del self.trainers[name]
            self._save()

    def add_trainer(self, name, data=None, splitter="same", splitter_params=None, exist='skip', aug_data=None):
        """Create and register a Trainer.

        Args:
            name (str): Trainer name.
            data: Dataset for the Trainer. ``None`` → use Experimenter's data.
            splitter: Splitter to use. ``'same'`` reuses ``sp_v``; pass a
                sklearn splitter object for a custom split strategy; ``None``
                trains on the full dataset.
            splitter_params (dict): Column mappings for the splitter. Must be
                ``None`` when ``splitter='same'``.
            exist (str): ``'skip'`` (default) returns existing if name already
                registered; ``'error'`` raises.

        Returns:
            Trainer: The newly created (or existing) Trainer.
        """
        if name in self.trainers:
            if exist == 'skip':
                return self.trainers[name]
            elif exist == 'error':
                raise RuntimeError(f"Trainer '{name}' already exists")

        if data is None:
            trainer_data = self.data
        else:
            trainer_data = wrap(data)

        if splitter == 'same':
            if splitter_params is not None:
                raise ValueError("splitter_params must be None when splitter='same'")
            trainer_splitter = self.sp_v
            trainer_splitter_params = self.splitter_params
        else:
            trainer_splitter = splitter
            trainer_splitter_params = splitter_params if splitter_params is not None else {}

        trainer = Trainer(
            name=name,
            pipeline=self.pipeline,
            data=trainer_data,
            path=self.path / '__trainer' / name,
            splitter=trainer_splitter,
            splitter_params=trainer_splitter_params,
            logger=self.logger,
            cache=self.cache,
            aug_data=aug_data,
        )
        self.trainers[name] = trainer
        self._save()
        return trainer

    def _validate_name(self, name):
        """Node 또는 NodeGroup 이름 검증

        Args:
            name: 검증할 이름

        Raises:
            ValueError: 이름이 유효하지 않을 경우
        """
        if name is None:
            return

        # '__' 포함 금지
        if '__' in name:
            raise ValueError(f"Name '{name}' cannot contain '__'")

        # 파일/폴더명으로 사용 불가한 문자 금지
        invalid_chars = ['/', '\\', '\0', '<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            if char in name:
                raise ValueError(f"Name '{name}' cannot contain '{char}'")

    def get_grp_path(self, grp):
        if self.path is None:
            return None
        if isinstance(grp, str):
            grp = self.pipeline.get_grp(grp)
        path_parts = [grp.name]
        current = self.pipeline.get_grp(grp.parent)
        while current is not None:
            path_parts.insert(0, current.name)
            current = self.pipeline.get_grp(current.parent)
        return self.path / '/'.join(path_parts)

    def get_node_path(self, node):
        if isinstance(node, str):
            node = self.pipeline.get_node(node)
        grp_path = self.get_grp_path(node.grp)
        return grp_path / node.name

    def set_grp(self, name, role=None, processor=None, edges=None, method=None, parent=None, adapter=None, params=None, desc=None, exist='diff'):
        self._check_open()
        result_obj = self.pipeline.set_grp(
            name, role, processor, edges, method, parent, adapter, params, desc=desc, exist=exist
        )
        
        affected_nodes = result_obj['affected_nodes']
        self.reset_nodes(affected_nodes)
        new_grp_path = self.get_grp_path(result_obj['grp'])
        if "old_grp" in result_obj:
            old_grp_path = self.get_grp_path(result_obj['old_grp'])
            if old_grp_path != new_grp_path:
                os.makedirs(new_grp_path, exist_ok=True)
                for fname in os.listdir(old_grp_path):
                    src_path = os.path.join(old_grp_path, fname)
                    dst_path = os.path.join(new_grp_path, fname)
                    shutil.move(src_path, dst_path)

            self.logger.info(f"Group '{name}' updated, {len(affected_nodes)} node(s) affected")
        self._save()
        return result_obj

    def rename_grp(self, name_from, name_to):
        self._check_open()
        old_grp_path = self.get_grp_path(name_from)
        self.pipeline.rename_grp(name_from, name_to)
        new_grp_path = self.get_grp_path(name_to)
        if old_grp_path.exists():
            os.makedirs(new_grp_path, exist_ok=True)
            for fname in os.listdir(old_grp_path):
                src_path = os.path.join(old_grp_path, fname)
                dst_path = os.path.join(new_grp_path, fname)
                shutil.move(src_path, dst_path)
            shutil.rmtree(old_grp_path)
        self._save()
    
    def remove_grp(self, name):
        self._check_open()
        self.pipeline.remove_grp(name)

        self.logger.info(f"Group '{name}' removed")
        self._save()


    def remove_node(self, name):
        """노드를 제거

        Args:
            name: 제거할 노드 이름

        Raises:
            ValueError: 노드가 존재하지 않거나, 자식 노드가 있는 경우
        """
        self._check_open()
        self.pipeline.remove_node(name)
        for v in self.collectors.values():
            v.reset_nodes([name])

        self.logger.info(f"Node '{name}' removed")
        self._save()

    def get_status(self, node_name):
        """Return the disk status of a head node across all folds.

        Reads info.pkl from every artifact_store (outer × inner folds).
        Returns the common status if all folds agree, or ``'inconsistent'``
        if they differ.

        Returns:
            ``'built'``, ``'finalized'``, ``'error'``, ``None`` (init),
            or ``'inconsistent'``.
        """
        statuses = {
            artifact_store.status(node_name)
            for outer_fold in self.outer_folds
            for artifact_store in outer_fold.artifact_stores
        }
        return statuses.pop() if len(statuses) == 1 else 'inconsistent'

    def finalize(self, nodes):
        """Release memory for built Head nodes (``built`` → ``finalized``).

        Disk artifacts are preserved so nodes can be reloaded.

        Args:
            nodes: Node query for Head nodes to finalize.
        """
        self._check_open()
        node_names = self.pipeline.get_node_names(nodes)
        for i in node_names:
            if i is None:
                continue
            node = self.pipeline.get_node(i)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'head':
                if self.get_status(i) == 'built':
                    self.logger.info(f"Finalize '{i}'")
                    for outer_fold in self.outer_folds:
                        for artifact_store in outer_fold.artifact_stores:
                            artifact_store.finalize(i)

    def reinitialize(self, nodes):
        self._check_open()
        node_names = self.pipeline.get_node_names(nodes)
        for i in node_names:
            if i is None:
                continue
            node = self.pipeline.get_node(i)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'stage':
                if i in self.node_objs and self.node_objs[i].status == 'finalized':
                    self.logger.info(f"reinitialize '{i}'")
                    del self.node_objs[i]
            else:
                reinitialized = False
                for outer_fold in self.outer_folds:
                    for artifact_store in outer_fold.artifact_stores:
                        if artifact_store.status(i) == 'finalized':
                            artifact_store.reset_node(i)
                            reinitialized = True
                if reinitialized:
                    self.logger.info(f"reinitialize '{i}'")

    def close_exp(self):
        """Finalize all built nodes and mark the experiment as closed.

        Collector data is preserved. After this call, :attr:`status` is
        ``'closed'`` and no further builds or experiments are permitted until
        :meth:`reopen_exp` is called.
        """
        if self.status != "open":
            raise RuntimeError("")
        for name in list(self.pipeline.nodes):
            if name is None:
                continue
            node = self.pipeline.get_node(name)
            grp = self.pipeline.get_grp(node.grp)
            logged = False
            for outer_fold in self.outer_folds:
                stores = outer_fold.train_data_flows if grp.role == 'stage' else outer_fold.artifact_stores
                for store in stores:
                    if store.status(name) == 'built':
                        if not logged:
                            self.logger.info(f"Finalize '{name}'")
                            logged = True
                        store.finalize(name)
        self.status = "closed"
        self._save()

    def reopen_exp(self):
        """Reopen a closed experiment and rebuild Stage nodes.

        Clears all node objects, sets status back to ``'open'``, then calls
        :meth:`build`.
        """
        if self.status != "closed":
            raise RuntimeError("")
        for name in list(self.pipeline.nodes):
            if name is None:
                continue
            node = self.pipeline.get_node(name)
            grp = self.pipeline.get_grp(node.grp)
            for outer_fold in self.outer_folds:
                if grp.role == 'stage':
                    for store in outer_fold.train_data_flows:
                        store.reset_node(name)
        self.status = "open"
        self.build()
        self._save()

    def set_node(
        self, name, grp, processor=None, edges=None,
        method=None, adapter=None, params=None, desc=None, exist='diff'
    ):
        self._check_open()
        result_obj = self.pipeline.set_node(
            name, grp, processor, edges, method, adapter, params, desc=desc, exist=exist
        )

        # 기존 노드를 업데이트한 경우, 하위 노드들도 재빌드
        if len(result_obj['affected_nodes']) > 0:
            affected_nodes = result_obj['affected_nodes']
            self.logger.info(f"Affected {len(affected_nodes)} dependent node(s): {sorted(affected_nodes)}")
            self.reset_nodes(affected_nodes)
        if result_obj['result'] == 'update':
            self.reset_nodes([name])
        self._save()
        return result_obj

    def reset_nodes(self, nodes):
        """Reset nodes to ``init`` state.

        Removes node objects, clears cache entries, and resets Collector and
        Trainer data for the affected nodes.

        Args:
            nodes (list[str]): Node names to reset.
        """
        for name in nodes:
            node = self.pipeline.get_node(name)
            if node is None:
                continue
            grp = self.pipeline.get_grp(node.grp)
            if grp is None:
                 continue
            for outer_fold in self.outer_folds:
                if grp.role == 'stage':
                    for flow in outer_fold.train_data_flows:
                        flow.reset_node(name)
                else:
                    for store in outer_fold.artifact_stores:
                        store.reset_node(name)

        self.cache.clear_nodes(nodes)

        for v in self.collectors.values():
            v.reset_nodes(nodes)

        for v in self.trainers.values():
            v.reset_nodes(nodes)

    def show_error_nodes(self, nodes=None, traceback=False):
        """Print nodes in ``error`` state.

        Args:
            nodes: Node query to filter. ``None`` checks all nodes.
            traceback (bool): Include full traceback in output.
        """
        node_names = self.pipeline.get_node_names(nodes)
        error_nodes = []
        for n in node_names:
            if n is None:
                continue
            node = self.pipeline.get_node(n)
            grp = self.pipeline.get_grp(node.grp)
            stores = (
                [flow for of in self.outer_folds for flow in of.train_data_flows]
                if grp.role == 'stage' else
                [store for of in self.outer_folds for store in of.artifact_stores]
            )
            if any(s.status(n) == 'error' for s in stores):
                error_nodes.append(n)
        if not error_nodes:
            self.logger.info("No error nodes found")
            return
        for n in error_nodes:
            node = self.pipeline.get_node(n)
            grp = self.pipeline.get_grp(node.grp)
            stores = (
                [flow for of in self.outer_folds for flow in of.train_data_flows]
                if grp.role == 'stage' else
                [store for of in self.outer_folds for store in of.artifact_stores]
            )
            err = next((s.get_info(n)['error'] for s in stores if s.status(n) == 'error'), None)
            if err is None:
                continue
            if traceback:
                self.logger.info(f"[{n}] {err['type']}: {err['message']}\n{err['traceback']}")
            else:
                self.logger.info(f"[{n}] {err['type']}: {err['message']}")

    def build(self, nodes=None, rebuild=False, n_jobs=1, gpu_id_list=None):
        """Build Stage nodes.

        Args:
            nodes: Node query — ``None`` (all stages), ``list``, or regex ``str``.
            rebuild (bool): If ``True``, rebuild already-built nodes.
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs to use for GPU-enabled nodes.
        """
        from ._executor import _build_flow_single, _build_flow_multi
        from ._tracker import LoggerExecuteTracker
        self._check_open()
        node_names = set(self.pipeline.get_node_names(nodes))
        target_nodes = [
            i for i in self.pipeline._get_affected_nodes([None])
            if i is not None
            and i in node_names
            and self.pipeline.grps[self.pipeline.nodes[i].grp].role == 'stage'
        ]
        if rebuild:
            self.reset_nodes(target_nodes)
        else:
            target_nodes = [
                i for i in target_nodes
                if self.get_status(i) not in ['built', 'finalized']
            ]
        if not target_nodes:
            self.logger.info("No stage nodes to build")
            return

        self.logger.info(f"Building {len(target_nodes)} node(s)")
        collectors = list(self.collectors.values())
        total = sum(len(of.train_data_flows) for of in self.outer_folds) * len(target_nodes)
        tracker = LoggerExecuteTracker(total, n_jobs, self.logger)

        try:
            if n_jobs > 1:
                errors = _build_flow_multi(self.outer_folds, self.pipeline, target_nodes, n_jobs,
                                           gpu_id_list=gpu_id_list, collectors=collectors,
                                           tracker=tracker)
            else:
                errors = _build_flow_single(self.outer_folds, self.pipeline, target_nodes,
                                            gpu_id_list=gpu_id_list, collectors=collectors,
                                            tracker=tracker)
        finally:
            tracker.close()

        error_nodes = list({n for _, _, n in errors})
        n_ok = len(target_nodes) - len(error_nodes)
        if error_nodes:
            self.logger.info(f"Build complete: {n_ok}/{len(target_nodes)} node(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            self.logger.info(f"Build complete: {len(target_nodes)} node(s)")
    
    def exp(self, nodes=None, finalize=False, n_jobs=1, gpu_id_list=None):
        """Run Head nodes and invoke all matching Collectors.

        Args:
            nodes: Node query — ``None`` (all heads), ``list``, or regex ``str``.
            finalize (bool): If ``True``, finalize after all folds complete.
            n_jobs (int): Number of parallel workers. Default 1 (sequential).
            gpu_id_list (list, optional): GPU IDs to use for GPU-enabled nodes.
        """
        from ._executor import _experiment_single, _experiment_multi
        from ._tracker import LoggerExecuteTracker
        self._check_open()
        node_names = set(self.pipeline.get_node_names(nodes))
        target_nodes = [
            i for i in self.pipeline._get_affected_nodes([None])
            if i is not None
            and i in node_names
            and self.pipeline.grps[self.pipeline.nodes[i].grp].role == 'head'
            and self.get_status(i) not in ['built', 'finalized']
        ]
        if not target_nodes:
            self.logger.info("No head nodes to experiment")
            return

        self.logger.info(f"Experimenting {len(target_nodes)} node(s)")
        collectors = list(self.collectors.values())
        total = sum(len(of.train_data_flows) for of in self.outer_folds) * len(target_nodes)
        tracker = LoggerExecuteTracker(total, n_jobs, self.logger)

        try:
            if n_jobs > 1:
                errors = _experiment_multi(self.outer_folds, self.pipeline, target_nodes, n_jobs,
                                           gpu_id_list=gpu_id_list, collectors=collectors,
                                           tracker=tracker, finalize=finalize)
            else:
                errors = _experiment_single(self.outer_folds, self.pipeline, target_nodes,
                                            gpu_id_list=gpu_id_list, collectors=collectors,
                                            tracker=tracker, finalize=finalize)
        finally:
            tracker.close()

        error_nodes = list({n for _, n in errors})
        n_ok = len(target_nodes) - len(error_nodes)
        if error_nodes:
            self.logger.info(f"Exp complete: {n_ok}/{len(target_nodes)} node(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            self.logger.info(f"Exp complete: {len(target_nodes)} node(s)")
        self._save()

    def collect(self, collector, nodes=None, exist='skip'):
        """Run a Collector ad-hoc over already-built Head nodes.

        Args:
            collector (Collector): Collector instance to run.
            nodes: Node query — ``None`` (all heads), ``list``, or regex ``str``.
            exist (str): ``'skip'`` (default) skips nodes already collected.

        Returns:
            Collector: The same collector after collection.
        """
        from ._executor import _run_collectors
        from ._node_processor import ProgressMonitor

        node_names = set(self.pipeline.get_node_names(nodes))
        target_nodes = [
            name for name in self.pipeline._get_affected_nodes([None])
            if name is not None
            and name in node_names
            and not (exist == 'skip' and collector.has(name))
            and self.get_status(name) == 'built'
            and collector.connector.match(self.pipeline.get_node_attrs(name))
        ]

        if not target_nodes:
            return collector

        collector._setup(len(self.outer_folds), len(self.outer_folds[0].train_data_flows))
        monitor = ProgressMonitor()
        n_total = self.get_n_splits() * len(target_nodes)
        try:
            self.logger.create_session(0)
            self.logger.create_session(1)
            self.logger.start_progress(0, 'Collect', n_total)
            n_done = 0
            for name in target_nodes:
                node_attrs = self.pipeline.get_node_attrs(name)
                edges = node_attrs['edges']
                self.logger.start_progress(1, name)
                for outer_idx, outer_fold in enumerate(self.outer_folds):
                    for inner_idx, (train_flow, artifact_store) in enumerate(
                        zip(outer_fold.train_data_flows, outer_fold.artifact_stores)
                    ):
                        if artifact_store.status(name) != 'built':
                            continue
                        obj, result, info = artifact_store.get_objs(name)
                        train_data = train_flow.get_train(edges)
                        valid_data = train_flow.get_valid(edges)
                        test_data = outer_fold.get_test_data(edges)
                        ext_data = {}
                        if collector.get_properties().get('need_process_data', False):
                            ext_data[collector.name] = train_flow.get_data(collector.get_ext_data(), node_attrs['edges'])
                        _run_collectors(
                            [collector], node_attrs, obj, result, info,
                            train_data, valid_data, test_data, ext_data,
                            outer_idx, inner_idx, monitor
                        )
                    n_done += 1
                    self.logger.update_progress(0, n_done)
                self.logger.end_progress(1)
            self.logger.end_progress(0, n_total)
        finally:
            self.logger.remove_session(1)
            self.logger.remove_session(0)
        return collector

    def process_ext(self, data, node_name, outer_idx):
        node_attrs = self.pipeline.get_node_attrs(node_name)
        edges = node_attrs['edges']
        ext_wrapped = wrap(data)
        for train_flow in self.outer_folds[outer_idx].train_data_flows:
            yield train_flow.get_data(ext_wrapped, edges)

    def get_train_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].train_data_flows[i_idx].get_train(edges)

    def get_valid_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].train_data_flows[i_idx].get_valid(edges)

    def get_test_data(self, edges, o_idx=0, i_idx=0):
        return self.outer_folds[o_idx].get_test_data(edges, i_idx)

    def get_node_train_data(self, node, o_idx=0, i_idx=0):
        edges = self.pipeline.get_node_attrs(node)['edges']
        return self.get_train_data(edges, o_idx, i_idx)

    def get_node_valid_data(self, node, o_idx=0, i_idx=0):
        edges = self.pipeline.get_node_attrs(node)['edges']
        return self.get_valid_data(edges, o_idx, i_idx)

    def get_node_test_data(self, node, o_idx=0, i_idx=0):
        edges = self.pipeline.get_node_attrs(node)['edges']
        return self.get_test_data(edges, o_idx, i_idx)


    def get_node_info(self):
        lines = [f"# Experiment Pipeline Summary\n"]
        lines.append(f"- **DataSource**\n")

        for name in self.pipeline.nodes.keys():
            if name is None:
                continue
            node = self.pipeline.get_node(name)
            node_attrs = node.get_attrs(self.pipeline.grps)
            processor_name = node_attrs['processor'].__name__ if node_attrs['processor'] else 'None'
            edges_info_parts = []
            for key, edge_list in node_attrs['edges'].items():
                edge_strs = [f"{n or 'DataSource'}{f'[{v}]' if v else ''}" for n, v in edge_list]
                edges_info_parts.append(f"{key}: [{', '.join(edge_strs)}]")
            edges_info = ", ".join(edges_info_parts)
            lines.append(f"## {name}")
            lines.append(f"- **Processor**: {processor_name}")
            lines.append(f"- **Method**: {node_attrs['method']}")
            lines.append(f"- **Edges**: {edges_info}")

            descendants = self.pipeline._find_descendants(name)
            if descendants:
                lines.append(f"- **Descendants**: {sorted(descendants)}")
            lines.append("")

        return "\n".join(lines)

    def get_objs(self, node_name, outer_idx = 0, inner_idx = 0):
        node = self.pipeline.get_node(node_name)
        node_attrs = node.get_attrs(self.pipeline.grps)
        fold = self.outer_folds[outer_idx]
        if node_attrs['role'] == 'head':
            return fold.artifact_stores[inner_idx].get_objs(node_name)
        else:
            return fold.train_data_flows[inner_idx].get_objs(node_name)

    def _save(self, filepath=None):
        if filepath is None:
            filepath = self.path / '__exp.pkl'

        save_data = {
            'data_key': self.data_key,
            'title': self.title,
            'sp': self.sp,
            'sp_v': self.sp_v,
            'splitter_params': self.splitter_params,
            'cache_maxsize': self.cache_maxsize,
            'exp_id': self.exp_id,
            'pipeline': self.pipeline,
            'collector_keys': {name: type(c).__name__ for name, c in self.collectors.items()},
            'trainer_keys': list(self.trainers.keys()),
            'status': self.status
        }

        with open(filepath, 'wb') as f:
            pkl.dump(save_data, f)

    @staticmethod
    def load(filepath, data, data_key=None, aug_data=None, logger = DefaultLogger(level=['info', 'progress'])):
        """Load a saved Experimenter from disk.

        Args:
            filepath (str | Path): Path to the experiment directory
                (contains ``__exp.pkl``).
            data: Dataset to attach. Must match the original data shape.
            data_key (str, optional): If the saved experiment has a ``data_key``,
                this must match.

        Returns:
            Experimenter: Restored experimenter with all nodes, collectors, and
            trainers reloaded.

        Raises:
            ValueError: If ``data_key`` does not match the saved value.
        """
        COLLECTOR_TYPES = {
            'MetricCollector': MetricCollector,
            'StackingCollector': StackingCollector,
            'ModelAttrCollector': ModelAttrCollector,
            'SHAPCollector': SHAPCollector,
            'OutputCollector': OutputCollector,
            'ProcessCollector': ProcessCollector,
        }

        filepath = Path(filepath)
        with open(filepath / '__exp.pkl', 'rb') as f:
            save_data = pkl.load(f)

        saved_data_key = save_data.get('data_key')
        if saved_data_key is not None and saved_data_key != data_key:
            raise ValueError(
                f"data_key mismatch: saved='{saved_data_key}', provided='{data_key}'"
            )

        exp = Experimenter(
            data=data,
            path=filepath,
            sp=save_data['sp'],
            sp_v=save_data['sp_v'],
            splitter_params=save_data['splitter_params'],
            title=save_data['title'],
            data_key=saved_data_key,
            cache_maxsize=save_data.get('cache_maxsize', 4 * 1024 ** 3),
            aug_data=aug_data,
            _save=False,
            logger = logger
        )
        exp.exp_id = save_data['exp_id']
        exp.pipeline = save_data['pipeline']
        exp.status = save_data['status']

        # Collector 복원
        collector_keys = save_data.get('collector_keys', {})
        for coll_name, type_name in collector_keys.items():
            cls = COLLECTOR_TYPES.get(type_name)
            if cls is None:
                continue
            coll_path = filepath / '__collector' / coll_name
            if (coll_path / '__config.pkl').exists():
                collector = cls.load(coll_path)
                exp.collectors[coll_name] = collector

        # Trainer 복원
        from ._trainer import Trainer
        for trainer_name in save_data.get('trainer_keys', []):
            trainer_path = filepath / '__trainer' / trainer_name
            if (trainer_path / '__trainer.pkl').exists():
                trainer = Trainer._load(
                    trainer_path,
                    pipeline=exp.pipeline,
                    data=exp.data,
                    cache=exp.cache,
                    logger=exp.logger,
                )
                exp.trainers[trainer_name] = trainer

        exp.logger.info(f"Loaded: {len(exp.pipeline.nodes) - 1} node(s), {len(exp.pipeline.grps)} group(s), {len(exp.outer_folds)} fold(s)")
        return exp
    
    def export_pipeline(self):
        return self.pipeline.copy()

    def import_pipeline(self, pipeline):
        if len(self.pipeline.nodes) > 0 or len(self.pipeline.grps) > 0:
            raise RuntimeError("")
        self.pipeline = pipeline.copy()
    
    def desc_status(self):
        return desc_status(self)

    def desc_spec(self):
        return desc_spec(self)

    def desc_obj_vars(self, node_name, idx):
        obj_vars = self.get_obj_vars(node_name, idx)
        return desc_obj_vars(self, obj_vars[0])

    def desc_pipeline(self, max_depth=None, direction='TD'):
        return self.pipeline.desc_pipeline(max_depth, direction)

    def desc_node(self, node_name, direction='TD', show_params=False):
        return self.pipeline.desc_node(node_name, direction, show_params)