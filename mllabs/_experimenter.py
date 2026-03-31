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
from ._store import ArtifactStore
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

    def __init__(self, path, data, test_idx, train_idx_list, cache=None, aug_data=None):
        self.path = Path(path)
        self.test_idx = test_idx
        self.data = data
        self.train_data_flows = [
            TrainDataFlow(
                path=self.path / str(j),
                data_source=DataWrapperProvider(data, train_idx, valid_idx=valid_idx, aug_data=aug_data),
                cache=cache,
                cache_key=(int(self.path.name), j),
            )
            for j, (train_idx, valid_idx) in enumerate(train_idx_list)
        ]
        self.artifact_stores = [
            ArtifactStore(path=self.path / str(j))
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

    def get_data(self, node, typ, idx):
        key = (node, typ, idx)
        return self.cache_dic.get(key, None)

    def put_data(self, node, typ, idx, data):
        key = (node, typ, idx)
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
        node_objs (dict): ``{node_name: StageObj}`` — stage nodes only; head node state is checked on-demand via :func:`get_head_status`.
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
                inner_folds = [(train_idx, None)]
            raw_splits.append((test_idx, inner_folds))

        self.pipeline = Pipeline()
        self.cache = DataCache(maxsize=cache_maxsize)

        self.outer_folds = [
            OuterFold(
                path=self.path / '__folds' / str(i),
                data=self.data,
                test_idx=test_idx,
                train_idx_list=inner_folds,
                cache=self.cache,
                aug_data=self.aug_data,
            )
            for i, (test_idx, inner_folds) in enumerate(raw_splits)
        ]
        self.grps = {}
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
            del self.collectors[name]
            self._save()

    def add_collector(self, collector, exist = 'skip'):
        """Register a Collector and immediately collect from built Head nodes.

        Args:
            collector (Collector): Collector instance to register.
            exist (str): ``'skip'`` (default) returns existing if already registered;
                ``'error'`` raises.

        Returns:
            Collector: The registered collector.
        """
        if collector.name in self.collectors:
            if exist == 'skip':
                return self.collectors[collector.name]
            elif exist == 'error':
                raise RuntimeError("")

        self._check_open()
        collector.path = self.path / '__collector' / collector.name
        collector.save()
        self.collectors[collector.name] = collector
        self.collect(collector)
        self._save()
        return collector

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
                grp_path = self.get_grp_path(node.grp)
                if get_head_status(grp_path, i) == 'built':
                    self.logger.info(f"Finalize '{i}'")
                    finalize_head(grp_path, i)

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
                grp_path = self.get_grp_path(node.grp)
                finalized_pkl = grp_path / i / 'finalized.pkl'
                if finalized_pkl.exists():
                    finalized_pkl.unlink()
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
            if grp.role == 'head':
                grp_path = self.get_grp_path(node.grp)
                if get_head_status(grp_path, name) == 'built':
                    self.logger.info(f"Finalize '{name}'")
                    finalize_head(grp_path, name)
        self.status = "closed"
        self._save()

    def reopen_exp(self):
        """Reopen a closed experiment and rebuild Stage nodes.

        Clears all node objects, sets status back to ``'open'``, then calls
        :meth:`build`.
        """
        if self.status != "closed":
            raise RuntimeError("")
        for k in list(self.node_objs.keys()):
            self.logger.info(f"Initialize '{k}'")
            del self.node_objs[k]
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
        for i in nodes:
            if i in self.node_objs:
                node_obj = self.node_objs[i]
                if node_obj.status == 'built':
                    node_obj.finalize()
                del self.node_objs[i]
            else:
                node = self.pipeline.get_node(i)
                if node is not None:
                    grp = self.pipeline.get_grp(node.grp)
                    if grp.role == 'head':
                        node_path = self.get_node_path(i)
                        if os.path.isdir(node_path):
                            shutil.rmtree(node_path)

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
            if grp.role == 'stage':
                if n in self.node_objs and self.node_objs[n].status == 'error':
                    error_nodes.append(n)
            else:
                if get_head_status(self.get_node_path(n).parent, n) == 'error':
                    error_nodes.append(n)
        if not error_nodes:
            self.logger.info("No error nodes found")
            return
        for n in error_nodes:
            node = self.pipeline.get_node(n)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'stage':
                err = self.node_objs[n].error
            else:
                err = get_head_error(self.get_node_path(n).parent, n)
            if traceback:
                self.logger.info(f"[{n}] {err['type']}: {err['message']}\n{err['traceback']}")
            else:
                self.logger.info(f"[{n}] {err['type']}: {err['message']}")

    def build(self, nodes = None, rebuild = False):
        """Build Stage nodes.

        Args:
            nodes: Node query — ``None`` (all stages), ``list``, or regex ``str``.
            rebuild (bool): If ``True``, rebuild already-built nodes.
        """
        self._check_open()
        node_names = self.pipeline.get_node_names(nodes)
        target_nodes = list()
        for i in self.pipeline._get_affected_nodes([None]):
            node = self.pipeline.get_node(i)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'stage':
                if i not in self.node_objs:
                    target_nodes.append(i)
                elif rebuild or self.node_objs[i].status != 'built':
                    target_nodes.append(i)

        self.logger.info(f"Building {len(target_nodes)} node(s)")
        for node in target_nodes:
            if node in self.node_objs:
                node_obj = self.node_objs[node]
            else:
                node_obj = StageObj(self.get_node_path(node))
                self.node_objs[node] = node_obj
            node_obj.start_build()
        
        n_splits = self.get_n_splits()
        self.logger.clear_progress()
        self.logger.start_progress("Build", n_splits)
        for i in range(n_splits):
            self.logger.update_progress(i)
            self.logger.start_progress("Node", len(target_nodes))
            for ni, node in enumerate(target_nodes):
                self.logger.update_progress(ni)
                self.logger.rename_progress(node)
                node_obj = self.node_objs[node]
                if node_obj.status == 'error':
                    continue
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        node_attrs = self.pipeline.get_node_attrs(node)
                        node_obj.build_idx(
                            i, node_attrs, self.get_node_data(node, i), [], self.logger
                        )
                        for w in caught:
                            self.logger.warning(f"[{node}] fold {i}: {w.category.__name__}: {w.message}")
                except Exception as e:
                    node_obj.set_error({
                        'type': type(e).__name__,
                        'message': str(e),
                        'traceback': traceback.format_exc(),
                        'fold': i,
                    })
                    self.logger.info(f"[{node}] Build error at fold {i}: {type(e).__name__}: {e}")
                    self.logger.info(traceback.format_exc())
            self.logger.end_progress(len(target_nodes))
        self.logger.end_progress(n_splits)

        error_nodes = [n for n in target_nodes if self.node_objs[n].status == 'error']
        for node in target_nodes:
            node_obj = self.node_objs[node]
            if node_obj.status != 'error':
                node_obj.end_build()
        if error_nodes:
            self.logger.info(f"Build complete: {len(target_nodes) - len(error_nodes)}/{len(target_nodes)} node(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            self.logger.info(f"Build complete: {len(target_nodes)} node(s)")
    
    def exp(self, nodes=None, finalize=False, include_train=True):
        """Run Head nodes and invoke all matching Collectors.

        Args:
            nodes: Node query — ``None`` (all heads), ``list``, or regex ``str``.
            finalize (bool): If ``True``, save ``finalized.pkl`` with specs after
                all folds complete and remove per-fold pkl files.
            include_train (bool): If ``False``, skip computing train output
                (``output_train`` will be absent from collector context).
        """
        self._check_open()
        node_names = set(self.pipeline.get_node_names(nodes))
        target_nodes = list()
        for i in self.pipeline._get_affected_nodes([None]):
            node = self.pipeline.get_node(i)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'head' and i in node_names:
                grp_path = self.get_grp_path(node.grp)
                if get_head_status(grp_path, i) not in ['built', 'finalized']:
                    target_nodes.append(i)

        self.logger.info(f"Experimenting {len(target_nodes)} node(s)")

        # node_attrs with path injected
        node_attrs_cache = {}
        for n in target_nodes:
            attrs = self.pipeline.get_node_attrs(n)
            attrs['path'] = self.get_grp_path(self.pipeline.get_node(n).grp)
            node_attrs_cache[n] = attrs

        # matched collectors per node
        node_matched = {
            node: [c for c in self.collectors.values()
                   if c.connector.match(node, node_attrs_cache[node])]
            for node in target_nodes
        }

        error_nodes_set = set()
        n_splits = self.get_n_splits()
        self.logger.clear_progress()
        self.logger.start_progress("Exp", n_splits)
        for i in range(n_splits):
            self.logger.update_progress(i)
            self.logger.start_progress("Node", len(target_nodes))
            for ni, node in enumerate(target_nodes):
                self.logger.update_progress(ni)
                self.logger.rename_progress(node)
                if node in error_nodes_set:
                    continue
                success = _head_exp_node(
                    node_attrs_cache[node]['path'], node_attrs_cache[node],
                    i, self.get_node_data(node, i),
                    list(self.collectors.values()),
                    self.logger,
                    finalize=finalize, include_train=include_train,
                )
                if not success:
                    error_nodes_set.add(node)
                    for collector in node_matched[node]:
                        collector.reset_nodes([node])

            self.logger.end_progress(len(target_nodes))
        self.logger.end_progress(n_splits)

        error_nodes = list(error_nodes_set)
        for node in target_nodes:
            if node not in error_nodes_set:
                if finalize:
                    grp_path = node_attrs_cache[node]['path']
                    finalize_head(grp_path, node)
                node_path = self.get_node_path(node)
                for c in node_matched[node]:
                    c._node_paths[node] = node_path

        if error_nodes:
            self.logger.info(f"Experimentation complete: {len(target_nodes) - len(error_nodes)}/{len(target_nodes)} node(s), {len(error_nodes)} error(s): {error_nodes}")
        else:
            self.logger.info(f"Experimentation complete: {len(target_nodes)} node(s)")
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
        node_names = set(self.pipeline.get_node_names(nodes))
        # built head 노드 중 connector 매칭
        target_nodes = []
        node_attrs_cache = {}
        for name in self.pipeline._get_affected_nodes([None]):
            node = self.pipeline.get_node(name)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role != 'head':
                continue
            if name not in node_names:
                continue
            if exist == 'skip' and collector.has(name):
                continue
            grp_path = self.get_grp_path(node.grp)
            if get_head_status(grp_path, name) != 'built':
                continue
            node_attrs = self.pipeline.get_node_attrs(name)
            node_attrs['path'] = grp_path
            if collector.connector.match(name, node_attrs) and not collector.has_node(name):
                target_nodes.append(name)
                node_attrs_cache[name] = node_attrs

        n_splits = self.get_n_splits()

        self.logger.start_progress("Collect", n_splits)
        for idx in range(n_splits):
            self.logger.update_progress(idx)
            self.logger.start_progress("Node", len(target_nodes))
            for ni, node in enumerate(target_nodes):
                self.logger.update_progress(ni)
                self.logger.rename_progress(node)
                _head_exp_node(
                    node_attrs_cache[node]['path'], node_attrs_cache[node],
                    idx, self.get_node_data(node, idx),
                    [collector], self.logger,
                )
            self.logger.end_progress(len(target_nodes))
        self.logger.end_progress(n_splits)

        for node in target_nodes:
            node_path = self.get_node_path(node)
            collector._node_paths[node] = node_path

        return collector

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

    def get_objs(self, node_name, idx):
        if node_name not in self.node_objs or node_name is None:
            raise ValueError(f"Node '{node_name}' objects not found")

        obj = self.node_objs[node_name]

        # 노드가 빌드되지 않았으면 에러
        if obj.status != 'built':
            raise ValueError(f"Node '{node_name}' status should be built")

        # 외부 fold의 내부 fold들: [(processor, train_v, info), ...]
        return obj.get_objs(idx)
    
    def get_obj_vars(self, node_name, idx):
        if node_name not in self.node_objs:
            raise ValueError(f"Node '{node_name}' has no object")

        # 외부 fold의 내부 fold들: [(processor, train_v, info), ...]
        objs_ = self.node_objs[node_name].get_objs(idx)

        # (입력변수 튜플, 출력변수 튜플) -> 내부 fold index 리스트
        var_map = {}
        for inner_idx, (processor, train_v, info) in enumerate(objs_):
            # 입력 변수와 출력 변수 가져오기
            input_vars = tuple(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else ()
            output_vars = tuple(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else ()

            # 튜플 키 생성
            key = (input_vars, output_vars)

            if key not in var_map:
                var_map[key] = []
            var_map[key].append(inner_idx)

        # 결과 리스트 생성: [(입력변수 리스트, 출력변수 리스트, 내부 폴드 index 리스트), ...]
        result = []
        for (input_vars, output_vars), fold_indices in var_map.items():
            result.append((list(input_vars), list(output_vars), fold_indices))

        # 등장 빈도(내부 폴드 개수)의 내림차순으로 정렬
        result.sort(key=lambda x: len(x[2]), reverse=True)
        return result

    def get_edges_var(self, edges):
        class _ColHolder:
            def __init__(self, columns):
                self._columns = columns
            def get_columns(self):
                return self._columns

        var_map = {}

        for idx in range(self.get_n_splits()):
            n_inner = len(self.outer_folds[idx].train_data_flows)
            # edges는 dict: {key: [(node_name, var), ...], ...}
            edge_objs = {}
            for key, edge_list in edges.items():
                edge_objs[key] = []
                for node_name, var in edge_list:
                    if node_name is None:
                        edge_objs[key].append((None, var, None))
                    else:
                        node_obj = self.node_objs[node_name]
                        edge_objs[key].append((node_name, var, node_obj.get_objs(idx)))

            for inner_idx in range(n_inner):
                collected = {}
                for key, objs_list in edge_objs.items():
                    collected[key] = []
                    for node_name, var, objs in objs_list:
                        if node_name is None:
                            cols = self.data.get_columns()
                            proc = None
                        else:
                            obj = next(objs)
                            proc = obj[0]
                            cols = list(proc.output_vars) if proc.output_vars is not None else []

                        if var is not None:
                            cols = resolve_columns(_ColHolder(cols), var, processor=proc)

                        collected[key].extend(cols)

                # key를 정렬된 형태로 튜플화
                key_tuple = tuple((k, tuple(v)) for k, v in sorted(collected.items()))
                if key_tuple not in var_map:
                    var_map[key_tuple] = []
                var_map[key_tuple].append((idx, inner_idx))

        result = []
        for vars_tuple, fold_indices in var_map.items():
            # dict로 복원
            vars_dict = {k: list(v) for k, v in vars_tuple}
            result.append((vars_dict, fold_indices))

        result.sort(key=lambda x: len(x[1]), reverse=True)

        return result

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
            'node_obj_keys': list(self.node_objs.keys()),
            'collector_keys': {name: type(c).__name__ for name, c in self.collectors.items()},
            'trainer_keys': list(self.trainers.keys()),
            'status': self.status
        }

        with open(filepath, 'wb') as f:
            pkl.dump(save_data, f)

    @staticmethod
    def load(filepath, data, data_key=None, aug_data=None):
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
        )
        exp.exp_id = save_data['exp_id']
        exp.pipeline = save_data['pipeline']
        exp.status = save_data['status']

        # node_objs 복원 (stage 노드만)
        for node_name in save_data['node_obj_keys']:
            node = exp.pipeline.get_node(node_name)
            grp = exp.pipeline.get_grp(node.grp)
            node_path = exp.get_node_path(node_name)

            if grp.role == 'stage':
                node_obj = StageObj(node_path)
                node_obj.load()
                exp.node_objs[node_name] = node_obj

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