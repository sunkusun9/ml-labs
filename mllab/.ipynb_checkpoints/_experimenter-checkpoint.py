import re
import os
import uuid
import pickle as pkl
import shutil
import traceback
import warnings
from pathlib import Path

import pandas as pd

from sklearn.model_selection import ShuffleSplit

from ._data_wrapper import wrap, unwrap
from ._node import NodeGroup, Node, RootNode
from ._describer import desc_spec, desc_pipeline, desc_node, desc_node_vars
from ._metric import Metric
from ._stacking import Stacking
from ._logger import DefaultLogger

class Experimenter():
    def __init__(
            self, data, path, data_names = None, sp = ShuffleSplit(n_splits=1, random_state=1), sp_v=None, splitter_params=None, title=None, data_key=None,
            logger = DefaultLogger(level=['info', 'progress'])
        ):
        self.logger = logger
        self.path = Path(path)
        if not os.path.exists(path):
            self.path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Created directory: {self.path}")
        self.train_idx_list = list()
        self.valid_idx_list = list()
        data_native = data
        data = wrap(data)
        self.root = data

        # 실험 타이틀 저장
        self.title = title

        # data 식별자 (load 시 검증용)
        self.data_key = data_key

        # splitter 설정 저장
        self.sp = sp
        self.sp_v = sp_v
        self.splitter_params = splitter_params if splitter_params is not None else {}
        self.exp_id = str(uuid.uuid4())

        split_params = {}

        if data_names is None:
            data_names = data.get_columns()
        for k, v in self.splitter_params.items():
            split_params[k] = unwrap(data.select_columns(v))

        for train_idx, valid_idx in sp.split(data_native, **split_params):
            if sp_v is not None:
                train_data = data.iloc(train_idx)
                train_data_native = unwrap(train_data)

                inner_split_params = {'X': train_data_native}
                for k, v in self.splitter_params.items():
                    inner_split_params[k] = unwrap(train_data.select_columns(v))

                self.train_idx_list.append([
                    (train_idx[train_v_idx], train_idx[valid_v_idx])
                    for train_v_idx, valid_v_idx in sp_v.split(**inner_split_params)
                ])
            else:
                self.train_idx_list.append([
                    (train_idx, None)
                ])
            self.valid_idx_list.append(valid_idx)
        self.nodes = {None: RootNode(self, data)}
        self.grps = {}
        self.metric = {}
        self.stacking = {}

    @staticmethod
    def create(data, path, data_names=None, sp=ShuffleSplit(n_splits=1, random_state=1), sp_v=None, splitter_params=None, title=None, data_key=None,
            logger = DefaultLogger(level=['info', 'progress'])):
        
        if os.path.exists(path):
            raise RuntimeError(f"Exists: {self.path}")
        return Experimenter(
            data, path, data_names, sp=sp, sp_v=sp_v, splitter_params=splitter_params, title=title, data_key=data_key,
            logger = logger)

    def get_n_splits(self):
        return len(self.train_idx_list)

    def add_metric(self, name, target_edges, output_var, metric_func, include_train=False):
        """Metric 인스턴스를 생성하여 추가

        Args:
            name: metric 이름
            target_edges: 타겟 edges
            output_var: 출력 변수
            metric_func: metric 함수
            include_train: train 결과 포함 여부 (기본값: False)

        Returns:
            Metric: 생성된 Metric 인스턴스
        """
        # __metric 폴더 생성 (최초 추가 시)
        metric_dir = self.path / "__metric"
        if not metric_dir.exists():
            metric_dir.mkdir(parents=True, exist_ok=True)

        metric = Metric(
            name=name,
            experimenter=self,
            target_edges=target_edges,
            output_var=output_var,
            metric_func=metric_func,
            include_train=include_train
        )
        self.metric[name] = metric
        self._save()
        return metric

    def add_stacking(self, name, target_edges, output_var, method='mean', include_target=True):
        """Stacking 인스턴스를 생성하여 추가

        Args:
            name: stacking 이름
            target_edges: 타겟 edges
            output_var: 출력 변수
            method: 집계 방법 (기본값: 'mean')
            include_target: 타겟 포함 여부 (기본값: True)

        Returns:
            Stacking: 생성된 Stacking 인스턴스
        """

        stacking = Stacking(
            experimenter=self,
            target_edges=target_edges,
            output_var=output_var,
            method=method,
            include_target=include_target
        )
        stacking.name = name
        stacking.save_config()
        self.stacking[name] = stacking
        self._save()
        return stacking

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

    def _find_descendants(self, node_name):
        """특정 노드에 의존하는 모든 하위 노드들을 찾음 (BFS)

        output_edges를 활용하여 효율적으로 탐색
        """
        descendants = set()
        queue = [node_name]

        while queue:
            current = queue.pop(0)

            if current not in self.nodes:
                continue

            # output_edges: 이 노드를 입력으로 사용하는 노드 이름 리스트
            for child_name in self.nodes[current].output_edges:
                if child_name not in descendants:
                    descendants.add(child_name)
                    queue.append(child_name)

        return descendants

    def _check_cycle(self, node_name, new_edges):
        """특정 노드에 새로운 edges를 추가했을 때 사이클이 생기는지 체크

        Args:
            node_name: 체크할 노드 이름
            new_edges: 추가할 edges 리스트 [(edge_name, var), ...]

        Returns:
            tuple: (has_cycle: bool, cycle_edges: list)
                - has_cycle: 사이클이 있으면 True, 없으면 False
                - cycle_edges: 사이클을 만드는 edge 이름들 리스트
        """
        # node_name의 descendants를 먼저 구함
        descendants = self._find_descendants(node_name)

        cycle_edges = []
        for edge_name, _ in new_edges:
            # Root(None)로의 edge는 사이클을 만들지 않음
            if edge_name is None:
                continue

            # edge_name이 실제 노드인지 확인
            if edge_name not in self.nodes:
                continue

            # edge_name이 node_name의 descendants에 있으면 사이클
            # node_name -> ... -> edge_name (이미 존재)
            # node_name -> edge_name (새로 추가)
            # 이면 node_name -> edge_name -> ... -> node_name 사이클이 생김
            if edge_name in descendants:
                cycle_edges.append(edge_name)

        if cycle_edges:
            return True, cycle_edges
        return False, []
    
    def _check_edges(self, edges):
        if edges is None:
            return False
        for name, _ in edges:
            if name is None:
                continue
            if name not in self.nodes:
                raise ValueError(f"Edge node '{name}' not found")
            if self.nodes[name].grp.role != 'pipe':
                raise ValueError(f"Edge node '{name}' must be a pipe node, got '{self.nodes[name].grp.role}'")
        return True

    def _get_all_nodes_in_grp(self, grp):
        """그룹과 하위 그룹의 모든 노드 이름을 수집"""
        result = list(grp.nodes)
        for child_grp in grp.child_grps:
            result.extend(self._get_all_nodes_in_grp(child_grp))
        return result

    def _compute_node_edges(self, node_name, new_grp_edges=None):
        """노드의 최종 edges 계산 (그룹 상속 포함)

        Args:
            node_name: 노드 이름
            new_grp_edges: 새로 적용할 그룹 edges (None이면 현재 그룹 attrs 사용)
        """
        if node_name not in self.nodes:
            return []

        node = self.nodes[node_name]
        node_own_edges = list(node.org_attr['edges']) if node.org_attr and node.org_attr['edges'] else []

        if new_grp_edges is not None:
            # 새 그룹 edges + 노드 자체 edges
            return new_grp_edges + node_own_edges
        else:
            # 현재 그룹 attrs에서 edges 가져오기
            grp_attrs = node.grp.get_attrs() if node.grp else {}
            grp_edges = grp_attrs.get('edges', [])
            return grp_edges + node_own_edges

    def set_grp(self, name, role=None, processor=None, edges=[], X=None, y=None, method=None, parent_grp=None, adapter=None, params=None):
        self._validate_name(name)
        self._check_edges(edges)
        if name in self.nodes:
            raise ValueError(f"Name '{name}' already exists as a node")

        # parent_grp가 문자열이면 grps에서 찾기
        if parent_grp is not None:
            if parent_grp not in self.grps:
                raise ValueError(f"Parent group '{parent_grp}' not found")
            parent_grp = self.grps.get(parent_grp)
            if role is None:
                role = parent_grp.role
        if role not in ['pipe', 'exp']:
            raise ValueError(f"Role must be 'pipe' or 'exp', got '{role}'")
        # 1. 새로운 그룹일 경우 추가
        if name not in self.grps:
            self._check_edges(edges)
            # NodeGroup 생성
            grp = NodeGroup(self, name, role, processor=processor, edges=edges, X=X, y=y, method=method, parent_grp=parent_grp, adapter=adapter, params=params)

            # parent의 child_grps에 추가
            if parent_grp is not None:
                parent_grp.child_grps.append(grp)

            # grps 딕셔너리에 등록
            self.grps[name] = grp

            # 디렉터리 생성
            if grp.path is not None and not grp.path.exists():
                grp.path.mkdir(parents=True, exist_ok=True)
            grp.save_info()
            self._save()
            return grp

        grp = self.grps[name]
        if grp.role != role:
            raise ValueError(f"Cannot change role of group '{name}': existing '{grp.role}', requested '{role}'")
        old_grp_path = grp.path
        # 3. edges 변경 시 순환 구조 체크 (변경 전 검증)
        if edges is not None:
            new_edges = edges if isinstance(edges, list) else [edges]

            # 이 그룹과 하위 그룹의 모든 노드 수집
            all_affected_nodes = self._get_all_nodes_in_grp(grp)

            # 각 노드에 대해 새 edges로 순환 구조 체크
            for node_name in all_affected_nodes:
                if node_name not in self.nodes:
                    continue

                node = self.nodes[node_name]
                # 노드의 그룹 계층에서 현재 grp의 위치를 고려하여 최종 edges 계산
                # 부모 그룹의 edges + 새 edges + 자식 그룹의 edges + 노드 자체 edges
                node_own_edges = list(node.org_attr['edges']) if node.org_attr and node.org_attr['edges'] else []

                # 그룹 계층에서 edges 수집 (현재 grp는 new_edges로 대체)
                grp_edges = []
                current_grp = node.grp
                while current_grp is not None:
                    if current_grp.name == name:
                        # 변경 대상 그룹: 새 edges 사용
                        grp_edges = new_edges + grp_edges
                    else:
                        grp_edges = current_grp.edges + grp_edges
                    current_grp = current_grp.parent_grp

                final_edges = grp_edges + node_own_edges

                # 사이클 체크
                has_cycle, cycle_edges = self._check_cycle(node_name, final_edges)
                if has_cycle:
                    cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
                    raise ValueError(f"Cannot update group '{name}': node '{node_name}' would create cycle through edge(s) {cycle_info}")

        # 4. 검증 통과 - 실제 변경 수행

        # parent_grp 변경 처리
        parent_changed = False
        new_parent = parent_grp
        if new_parent is not None and grp.parent_grp != new_parent:
            parent_changed = True
            # 이전 parent의 child_grps에서 제거
            if grp.parent_grp is not None:
                grp.parent_grp.child_grps.remove(grp)
            # 새로운 parent의 child_grps에 추가
            grp.parent_grp = new_parent
            if new_parent is not None:
                new_parent.child_grps.append(grp)

        # 그룹 속성 업데이트
        if processor is not None:
            grp.processor = processor
        if edges is not None:
            grp.edges = edges if isinstance(edges, list) else [edges]
        if X is not None:
            grp.X = X
        if y is not None:
            grp.y = y
        if method is not None:
            grp.method = method
        if adapter is not None:
            grp.adapter = adapter
        if params is not None:
            grp.params.update(params)

        # parent 변경 시 디렉터리 구조 업데이트
        if parent_changed:
            self._ensure_grp_directories(grp)

        # 5. 영향받는 노드들 초기화
        all_affected_nodes = self._get_all_nodes_in_grp(grp)
        if len(all_affected_nodes) == 0:
            self.logger.info(f"Group '{name}' updated (no nodes to rebuild)")
            return grp
        
        node_to_initialize = self._get_effected_nodes(all_affected_nodes)
        for node in node_to_initialize:
            node.initialize()

        for v in self.metric.values():
            v.reset_nodes(node_to_initialize)
        
        for v in self.stacking.values():
            v.reset_nodes(node_to_initialize)

        new_grp_path = grp.path
        if old_grp_path != new_grp_path:
            os.makedirs(dst_dir, exist_ok=True)
            for name in os.listdir(old_grp_path):
                src_path = os.path.join(old_grp_path, name)
                dst_path = os.path.join(dst_dir, name)
                shutil.move(src_path, dst_path)
        grp.save_info()
        self.logger.info(f"Group '{name}' updated, {len(node_to_initialize)} node(s) affected")
        self._save()
        return grp

    def rename_grp(self, name_from, name_to):
        self._validate_name(name_to)

        if name_from not in self.grps:
            raise ValueError(f"Group '{name_from}' not found")
        if name_to in self.grps:
            raise ValueError(f"Group '{name_to}' already exists")
            
        grp = self.grps[name_from]
        old_grp_path = grp.path
        grp.name = name_to
        if grp.parent_grp is not None:
            # 이전 parent의 child_grps에서 제거
            if grp.parent_grp is not None:
                grp.parent_grp.child_grps.remove(name_from)
                grp.parent_grp.child_grps.append(name_to)
        new_grp_path = grp.path
        os.makedirs(new_grp_path, exist_ok=True)
        for name in os.listdir(old_grp_path):
            src_path = os.path.join(old_grp_path, name)
            dst_path = os.path.join(new_grp_path, name)
            shutil.move(src_path, dst_path)
        shutil.rmtree(old_grp_path)
        del self.grps[name_from]
        self.grps[name_to] = grp
        self._save()
        
    def _get_effected_nodes(self, nodes):
        # 우선순위 알고리즘: BFS로 노드들의 빌드 우선순위 결정
        priorities = {}
        queue = []

        # 변경된 그룹의 노드들을 Root로 우선순위 1 할당
        for node_name in nodes:
            priorities[node_name] = 1
            queue.append((node_name, 1))

        # BFS로 하위 노드들 탐색
        while queue:
            current_node, current_priority = queue.pop(0)

            # 현재 노드에 의존하는 하위 노드들 찾기
            descendants = self._find_descendants(current_node)

            for desc_node in descendants:
                new_priority = current_priority + 1
                # 가장 마지막에 배정된 우선순위가 최종 우선순위
                if desc_node not in priorities or priorities[desc_node] < new_priority:
                    priorities[desc_node] = new_priority
                    queue.append((desc_node, new_priority))
        # 우선순위 순으로 정렬 (낮은 숫자가 먼저)
        sorted_nodes = sorted(priorities.items(), key=lambda x: x[1])
        return [self.nodes[i[0]] for i in sorted_nodes]
    
    def remove_grp(self, name):
        if name not in self.grps:
            raise ValueError(f"Group '{name}' not found")

        grp = self.grps[name]

        # child group이 있으면 제거 불가
        if len(grp.child_grps) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.child_grps)} child group(s)")

        # 소속 Node가 있으면 제거 불가
        if len(grp.nodes) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.nodes)} node(s)")

        # parent의 child_grps에서 제거
        if grp.parent_grp is not None:
            grp.parent_grp.child_grps.remove(grp)

        # grps 딕셔너리에서 제거
        del self.grps[name]

        self.logger.info(f"Group '{name}' removed")
        self._save()

    def get_parents(self, node_name):
        if node_name not in self.nodes:
            return []

        node = self.nodes[node_name]
        if node.grp_name is None:
            return []

        result = []
        current_grp = self.grps.get(node.grp_name)

        while current_grp is not None:
            result.append(current_grp.name)
            current_grp = current_grp.parent_grp

        return result

    def get_node_names(self, query):
        if isinstance(query, str):
            if query not in self.grps:
                return []

            result = []
            def collect_nodes(grp):
                result.extend(grp.nodes)
                for child_grp in grp.child_grps:
                    collect_nodes(child_grp)

            collect_nodes(self.grps[query])
            return result

        elif isinstance(query, re.Pattern):
            return [name for name in self.nodes.keys() if name is not None and query.search(name)]

        else:
            raise ValueError(f"query must be str or re.Pattern, got {type(query)}")

    def remove_node(self, name):
        """노드를 제거

        Args:
            name: 제거할 노드 이름

        Raises:
            ValueError: 노드가 존재하지 않거나, 자식 노드가 있는 경우
        """
        # 노드가 존재하는지 확인
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found")

        # Root 노드는 제거 불가
        if name is None:
            raise ValueError("Cannot remove Root node")

        # 자식 노드(descendants)가 있는지 확인
        descendants = self._find_descendants(name)
        if descendants:
            descendants_list = sorted(descendants)
            raise ValueError(f"Cannot remove node '{name}': has {len(descendants)} dependent node(s): {descendants_list}")

        node = self.nodes[name]

        # output_edges 무결성 유지: 부모 노드들의 output_edges에서 제거
        self._update_output_edges(name, node.edges, None)

        # 그룹에 속해있으면 그룹의 nodes 리스트에서 제거
        grp_name = node.grp.name if node.grp is not None else None
        if grp_name is not None and grp_name in self.grps:
            grp = self.grps[grp_name]
            if name in grp.nodes:
                grp.nodes.remove(name)
                self.logger.info(f"Removed '{name}' from group '{grp_name}'")

        node.remove()
        # nodes 딕셔너리에서 제거
        del self.nodes[name]

        self.logger.info(f"Node '{name}' removed")
        self._save()

    def finalize(self, nodes):
        if nodes is None:
            # 기존 동작: 모든 root group의 노드
            node_names = list(self.nodes.keys())
        elif isinstance(nodes, list):
            node_names = [n for n in nodes if n in self.nodes]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            node_names = [k for k in self.nodes.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        target_nodes = list()
        for i in node_names:
            node = self.nodes[i]
            if type(node) != RootNode and node.grp.role == 'exp' and node.status == 'built':
                self.logger.info(f"Finalize '{i}'")
                node.finalize()

    def reinitialize(self, nodes):
        if nodes is None:
            # 기존 동작: 모든 root group의 노드
            node_names = list(self.nodes.keys())
        elif isinstance(nodes, list):
            node_names = [n for n in nodes if n in self.nodes]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            node_names = [k for k in self.nodes.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        target_nodes = list()
        for i in node_names:
            node = self.nodes[i]
            if type(node) != RootNode and node.status == 'finalized':
                self.logger.info(f"reinitialize '{i}'")
                node.initialize()

    def close_exp(self):
        for k, node in self.nodes.items():
            if type(node) != RootNode and node.status == 'built':
                self.logger.info(f"Finalize '{k}'")
                node.finalize()
    
    def _update_output_edges(self, node_name, old_edges, new_edges):
        """output_edges 무결성 유지

        Args:
            node_name: 현재 노드 이름
            old_edges: 이전 edges 리스트 (None이면 제거만 스킵)
            new_edges: 새 edges 리스트 (None이면 추가만 스킵)
        """
        # 이전 edges에서 현재 노드 제거
        if old_edges is not None:
            for edge_name, _ in old_edges:
                if edge_name in self.nodes:
                    parent_node = self.nodes[edge_name]
                    if node_name in parent_node.output_edges:
                        parent_node.output_edges.remove(node_name)

        # 새 edges에 현재 노드 추가
        if new_edges is not None:
            for edge_name, _ in new_edges:
                if edge_name in self.nodes:
                    parent_node = self.nodes[edge_name]
                    if node_name not in parent_node.output_edges:
                        parent_node.output_edges.append(node_name)

    def set_node(
        self, name, grp, processor = None, edges = list(), X = None, y = None,
        method = None, adapter = 'default', params = None
    ):
        self._validate_name(name)

        if name in self.grps:
            raise ValueError(f"Name '{name}' already exists as a group")

        if grp not in self.grps:
            raise ValueError(f"Group '{grp}' not found")
        
        self._check_edges(edges)

        # 기존 노드가 있는지 확인
        is_update = name in self.nodes
        old_edges = None
        old_output_edges = None
        if is_update:
            old_edges = self.nodes[name].edges
            old_output_edges = self.nodes[name].output_edges

        # params 기본값 처리
        if params is None:
            params = {}

        # org_attr 생성 (원본 파라미터 저장)
        org_attr = {
            'processor': processor,
            'edges': edges,
            'X': X,
            'y': y,
            'method': method,
            'adapter': adapter,
            'params': params
        }

        # grp 이름 저장
        grp_name = grp
        grp_obj = self.grps.get(grp, None)
        if grp_obj is None:
            raise ValueError(f"Group '{grp}' not found")

        # grp의 attrs를 가져와서 기본값으로 사용
        grp_attrs = grp_obj.get_attrs()

        # 파라미터로 넘어온 값이 None이 아니면 override
        if processor is None:
            processor = grp_attrs.get('processor', None)
        if len(grp_attrs['edges']) > 0:
            edges = edges + grp_attrs['edges']
        if X is None:
            X = grp_attrs['X']
        if y is None:
            y = grp_attrs['y']
        if method is None:
            method = grp_attrs.get('method', None)
        if adapter is None:
            adapter = grp_attrs.get('adapter', None)

        # params는 grp의 params를 가져와서 현재 params로 override
        merged_params = {**grp_attrs['params'], **params}

        # processor 체크
        if processor is None:
            raise ValueError(f"Cannot create node '{name}': processor is required")

        # method가 None이면 기본값 설정
        if method is None:
            raise ValueError(f"Cannot create node '{name}': method is required")

        # edges를 리스트로 정규화
        if not isinstance(edges, list):
            edges = [edges]

        # 사이클 체크
        has_cycle, cycle_edges = self._check_cycle(name, edges)
        if has_cycle:
            cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
            raise ValueError(f"Cannot add node '{name}': would create cycle through edge(s) {cycle_info}")

        # output_edges 무결성 업데이트
        self._update_output_edges(name, old_edges, edges)

        node = Node(self, name, processor, edges, X = X, y = y, method = method, grp = grp_obj, adapter = adapter, org_attr = org_attr, params = merged_params)
        if old_output_edges is not None:
            node.output_edges = old_output_edges
        # grp에 노드 추가
        if grp_obj is not None:
            if name not in grp_obj.nodes:
                grp_obj.nodes.append(name)

        # 기존 노드를 업데이트한 경우, 하위 노드들도 재빌드
        if is_update:
            descendants = self._find_descendants(name)
            if descendants:
                self.logger.info(f"Effected {len(descendants)} dependent node(s): {sorted(descendants)}")
                for i in descendants:
                    self.nodes[i].initialize()

                for v in self.metric.values():
                    v.reset_nodes(descendants)
                
                for v in self.stacking.values():
                    v.reset_nodes(descendants)

        # 그룹이 변경된 경우 이전 그룹에서 노드 제거
        if is_update and self.nodes[name].grp.name != grp_name:
            old_grp_name = self.nodes[name].grp.name
            if old_grp_name is not None and old_grp_name in self.grps:
                old_grp = self.grps[old_grp_name]
                if name in old_grp.nodes:
                    old_grp.nodes.remove(name)
                    self.logger.info(f"Removed '{name}' from group '{old_grp_name}'")
            if grp_name is not None:
                self.logger.info(f"Moved '{name}' to group '{grp_name}'")

        self.nodes[name] = node
        self._save()
        return node

    def build(self, nodes = None, rebuild = False):
        if nodes is None:
            # 기존 동작: 모든 root group의 노드
            node_names = list(self.nodes.keys())
        elif isinstance(nodes, list):
            node_names = [n for n in nodes if n in self.nodes]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            node_names = [k for k in self.nodes.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        target_nodes = [
            i for i in self._get_effected_nodes([None]) if type(i) != RootNode and i.grp.role == 'pipe' and (i.name in node_names and (i.status is None or rebuild))
        ]
        self.logger.info(f"Building {len(target_nodes)} node(s)")
        for node in target_nodes:
            node.start_build()
        n_splits = self.get_n_splits()
        self.logger.start_progress("Build", n_splits)
        try:
            for i in range(n_splits):
                self.logger.update_progress(i)
                self.logger.start_progress("Node", len(target_nodes))
                for ni, node in enumerate(target_nodes):
                    self.logger.update_progress(ni)
                    self.logger._progress[-1][0] = node.name
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        node.build_idx(i)
                        for w in caught:
                            self.logger.warning(f"[{node.name}] fold {i}: {w.category.__name__}: {w.message}")
                self.logger.end_progress(len(target_nodes))
            self.logger.end_progress(n_splits)
        except Exception as e:
            self.logger.clear_progress()
            self.logger.info(f"Build failed at fold {i}, node '{node.name}': {type(e).__name__}: {e}")
            self.logger.info(traceback.format_exc())
            raise
        for node in target_nodes:
            node.end_build()
        self.logger.info(f"Build complete: {len(target_nodes)} node(s)")
    
    def exp(self, nodes = None):
        if nodes is None:
            # 기존 동작: 모든 root group의 노드
            node_names = list(self.nodes.keys())
        elif isinstance(nodes, list):
            node_names = [n for n in nodes if n in self.nodes]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            node_names = [k for k in self.nodes.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        target_nodes = [
            i for i in self._get_effected_nodes([None]) if type(i) != RootNode and i.grp.role == 'exp' and (i.name in node_names and i.status is None)
        ]
        self.logger.info(f"Experimenting {len(target_nodes)} node(s)")

        # start_experiment for all nodes
        for node in target_nodes:
            node.start_experiment()

        # _start for metrics and stackings
        for v in self.metric.values():
            for node in target_nodes:
                v._start(node.name)
        for v in self.stacking.values():
            for node in target_nodes:
                v._start(node.name)

        # experiment loop
        n_splits = self.get_n_splits()
        self.logger.start_progress("Exp", n_splits)
        try:
            for i in range(n_splits):
                self.logger.update_progress(i)
                # prepare target metrics data
                target_metrics = {
                    k: v._get_data(i) for k, v in self.metric.items()
                }

                self.logger.start_progress("Node", len(target_nodes))
                for ni, node in enumerate(target_nodes):
                    self.logger.update_progress(ni)
                    self.logger._progress[-1][0] = node.name
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        result_iter = node.experiment(i)

                        stacks = {k: list() for k in self.stacking.keys()}
                        sub_metrics = {k: list() for k in self.metric.keys()}

                        for n, result_data in enumerate(result_iter):
                            # collect metrics
                            for k, v in self.metric.items():
                                sub_metric = v._get_metric(target_metrics[k][n], result_data)
                                sub_metric = {k_sub: v_sub for k_sub, v_sub in sub_metric.items()}
                                sub_metrics[k].append(sub_metric)
                            # collect stacking data
                            for k, v in self.stacking.items():
                                _valid = v._get_valid(result_data)
                                if _valid is not None:
                                    stacks[k].append(_valid)

                        for w in caught:
                            self.logger.warning(f"[{node.name}] fold {i}: {w.category.__name__}: {w.message}")

                    # set metrics
                    for k, v in self.metric.items():
                        v._set_metric(node.name, i, sub_metrics[k])
                    # aggregate and stack
                    for k, v in self.stacking.items():
                        if len(stacks[k]) > 0:
                            stk = v._aggregate(iter(stacks[k]))
                            v._stack(node.name, i, stk)
                self.logger.end_progress(len(target_nodes))
            self.logger.end_progress(n_splits)
        except Exception as e:
            self.logger.clear_progress()
            self.logger.info(f"Exp failed at fold {i}, node '{node.name}': {type(e).__name__}: {e}")
            self.logger.info(traceback.format_exc())
            raise

        # end_experiment for all nodes
        for node in target_nodes:
            node.end_experiment()

        # _end for metrics and stackings
        for v in self.metric.values():
            for node in target_nodes:
                v._end(node.name)
        for v in self.stacking.values():
            for node in target_nodes:
                v._end(node.name)

        self.logger.info(f"Experimentation complete: {len(target_nodes)} node(s)")

    def get_data(self, idx, edges):
        def ret_data_func(data_list):
            for z in zip(*data_list):
                train_sub, valid_sub, outer_valid_sub = list(), list(), list()
                for (train_data, train_v_data), outer_valid_data in z:
                    train_sub.append(train_data)
                    if train_v_data is not None:
                        valid_sub.append(train_v_data)
                    outer_valid_sub.append(outer_valid_data)

                train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                if len(valid_sub) > 0:
                    valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                    yield (train_concat, valid_concat), outer_concat
                else:
                    yield (train_concat, None), outer_concat

        data_list = list()
        for node_name, var in edges:
            data_list.append(self.nodes[node_name].get_data(idx, var))
        return ret_data_func(data_list)
    
    def get_data_train(self, idx, edges):
        def ret_data_func(data_list):
            for z in zip(*data_list):
                train_sub, valid_sub = list(), list()
                for train_data, train_v_data in z:
                    train_sub.append(train_data)
                    if train_v_data is not None:
                        valid_sub.append(train_v_data)
                train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                if len(valid_sub) > 0:
                    valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                    yield train_concat, valid_concat
                else:
                    yield train_concat, None

        data_list = list()
        for node_name, var in edges:
            data_list.append(self.nodes[node_name].get_data_train(idx, var))
        return ret_data_func(data_list)
    
    def get_data_valid(self, idx, edges):
        """외부 검증 데이터에 대한 처리 결과를 가져옴

        Args:
            idx: outer fold 인덱스
            edges: [(node_name, var), ...] 형태의 edge 리스트

        Yields:
            valid_concat: 각 inner fold 모델로 처리된 외부 검증 데이터 결과 (concat)
        """
        def ret_data_func(data_list):
            for z in zip(*data_list):
                outer_valid_sub = list()
                for outer_valid_data in z:
                    outer_valid_sub.append(outer_valid_data)

                # DataWrapper의 concat 사용
                outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                yield outer_concat

        data_list = list()
        for node_name, var in edges:
            data_list.append(self.nodes[node_name].get_data_valid(idx, var))
        return ret_data_func(data_list)

    def split(self, edges):
        for idx in range(len(self.train_idx_list)):
            yield self.get_data(idx, edges)
    
    def get_node_output(self, idx, node, var = None):
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found")
        return self.nodes[node].get_data(idx, var)

    def get_node_train_output(self, idx, node, var=None):
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found")
        return self.nodes[node].get_data_train(idx, var)

    def get_node_valid_output(self, idx, node, var=None):
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found")
        return self.nodes[node].get_data_valid(idx, var)

    def get_node_info(self):
        lines = [f"# Experiment Pipeline Summary\n"]
        lines.append(f"- **Root**: {type(self.root).__name__}\n")

        for name, node in self.nodes.items():
            if name is None:
                continue
            processor_name = node.processor.__name__
            edges_info = ", ".join([
                f"{n or 'Root'}{f'[{v}]' if v else ''}"
                for n, v in node.edges
            ])
            lines.append(f"## {name}")
            lines.append(f"- **Processor**: {processor_name}")
            lines.append(f"- **Method**: {node.method}")
            lines.append(f"- **Edges**: {edges_info}")

            descendants = self._find_descendants(name)
            if descendants:
                lines.append(f"- **Descendants**: {sorted(descendants)}")
            lines.append("")

        return "\n".join(lines)

    def desc_spec(self):
        """실험 스펙을 Markdown으로 반환"""
        return desc_spec(self)

    def desc_pipeline(self, max_depth=None, direction='TD'):
        """파이프라인 구조를 Mermaid Markdown으로 반환

        Args:
            max_depth: 최대 표시 깊이 (None이면 무제한)
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
        """
        return desc_pipeline(self, max_depth, direction)

    def desc_node(self, node_name, direction='TD', show_params=False):
        """특정 노드까지의 연결 구조를 Mermaid Markdown으로 반환

        Args:
            node_name: 대상 노드 이름
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
            show_params: True이면 노드의 파라미터 정보를 표시 (default: False)
        """
        return desc_node(self, node_name, direction, show_params)

    def desc_node_vars(self, node_name, idx):
        """특정 노드의 입력/출력 변수를 분석

        Args:
            node_name: 대상 노드 이름
            idx: 외부 fold 인덱스

        Returns:
            list: [(입력변수 리스트, 출력변수 리스트, 해당 내부 폴드 index 리스트), ...]
                  등장 빈도의 내림차순으로 정렬
        """
        return desc_node_vars(self, node_name, idx)

    def get_node_vars(self, node_name, idx):
        """특정 노드의 입력/출력 변수를 가져옴

        Args:
            node_name: 대상 노드 이름
            idx: 외부 fold 인덱스

        Returns:
            list: [(입력변수 리스트, 출력변수 리스트, 해당 내부 폴드 index 리스트), ...]
                  등장 빈도의 내림차순으로 정렬
        """
        if node_name not in self.nodes or node_name is None:
            raise ValueError(f"Node '{node_name}' not found")

        node = self.nodes[node_name]

        # 노드가 빌드되지 않았으면 에러
        if node.status != 'built':
            raise ValueError(f"Node '{node_name}' status should be built")

        # 외부 fold의 내부 fold들: [(processor, train_v, info), ...]
        inner_folds = node.get_exp_obj(idx)

        # (입력변수 튜플, 출력변수 튜플) -> 내부 fold index 리스트
        var_map = {}

        for inner_idx, (processor, train_v, info) in enumerate(inner_folds):
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

    def _get_grp_load_order(self):
        """NodeGroup 로딩 순서를 BFS로 계산 (parent가 없는 그룹부터 시작)

        Returns:
            list: 로딩 순서에 맞는 (grp_name, parent_grp_name) 튜플 리스트
        """
        result = []
        # 최상위 그룹 찾기 (parent_grp가 None인 그룹)
        queue = [(grp.name, None) for grp in self.grps.values() if grp.parent_grp is None]

        while queue:
            grp_name, parent_name = queue.pop(0)
            result.append((grp_name, parent_name))
            grp = self.grps[grp_name]
            # 자식 그룹을 큐에 추가
            for child_grp in grp.child_grps:
                queue.append((child_grp.name, grp_name))

        return result

    def _get_node_load_order(self):
        """Node 로딩 순서를 계산 (_get_effected_nodes 사용)

        Returns:
            list: 로딩 순서에 맞는 (node_name, grp_name) 튜플 리스트
        """
        effected = self._get_effected_nodes([None])
        result = []
        for node in effected:
            if type(node) != RootNode:
                result.append((node.name, node.grp.name))
        return result

    def _save(self, filepath=None):
        """Experimenter 객체를 파일로 저장

        Args:
            filepath: 저장할 파일 경로 (None이면 self.path / '__exp.pkl')
        """
        if filepath is None:
            filepath = self.path / '__exp.pkl'

        # 저장할 데이터 구성 (data는 저장하지 않음)
        save_data = {
            'data_key': self.data_key,
            'title': self.title,
            'sp': self.sp,
            'sp_v': self.sp_v,
            'splitter_params': self.splitter_params,
            'exp_id': self.exp_id,
            'grp_load_order': self._get_grp_load_order(),
            'node_load_order': self._get_node_load_order(),
            'metric_keys': list(self.metric.keys()),
            'stacking_keys': list(self.stacking.keys())
        }

        # print(f"💾 Saving Experimenter to {filepath}...")
        with open(filepath, 'wb') as f:
            pkl.dump(save_data, f)
        """
        print(f"✅ Experimenter saved successfully")
        print(f"   - {len(save_data['node_load_order'])} node(s)")
        print(f"   - {len(save_data['grp_load_order'])} group(s)")
        print(f"   - {len(save_data['metric_keys'])} metric(s)")
        print(f"   - {len(save_data['stacking_keys'])} stacking(s)")
        """

    @staticmethod
    def load(filepath, data, data_key=None):
        """파일에서 Experimenter 객체를 불러옴

        Args:
            filepath: 불러올 파일 경로
            data: 실험에 사용할 데이터
            data_key: 데이터 식별자 (저장된 data_key와 비교하여 검증)

        Returns:
            Experimenter: 불러온 Experimenter 객체

        Raises:
            ValueError: 저장된 data_key와 전달된 data_key가 일치하지 않는 경우
        """
        from ._metric import Metric
        from ._stacking import Stacking

        filepath = Path(filepath)
        with open(filepath / '__exp.pkl', 'rb') as f:
            save_data = pkl.load(f)

        # data_key 검증 (저장된 data_key가 None이 아닌 경우에만)
        saved_data_key = save_data.get('data_key')
        if saved_data_key is not None and saved_data_key != data_key:
            raise ValueError(
                f"data_key mismatch: saved='{saved_data_key}', provided='{data_key}'"
            )

        # Experimenter 생성자 활용
        exp = Experimenter(
            data=data,
            path=filepath,
            sp=save_data['sp'],
            sp_v=save_data['sp_v'],
            splitter_params=save_data['splitter_params'],
            title=save_data['title'],
            data_key=saved_data_key
        )
        # exp_id를 저장된 값으로 복원
        exp.exp_id = save_data['exp_id']

        # NodeGroup 복원 (로딩 순서대로)
        for grp_name, parent_grp_name in save_data['grp_load_order']:
            parent_grp = exp.grps.get(parent_grp_name) if parent_grp_name else None
            grp = NodeGroup.load(exp, grp_name, parent_grp)
            exp.grps[grp_name] = grp

        # Node 복원 (로딩 순서대로)
        for node_name, grp_name in save_data['node_load_order']:
            grp = exp.grps[grp_name]
            node = Node.load(exp, grp, node_name)
            exp.nodes[node_name] = node

        # output_edges 재구성
        for node_name, node in exp.nodes.items():
            if node_name is None:
                continue
            for edge_name, _ in node.edges:
                if edge_name in exp.nodes:
                    parent_node = exp.nodes[edge_name]
                    if node_name not in parent_node.output_edges:
                        parent_node.output_edges.append(node_name)

        # Metric 복원
        for metric_name in save_data['metric_keys']:
            metric = Metric.load_from_file(exp, metric_name)
            exp.metric[metric_name] = metric

        # Stacking 복원
        for stacking_name in save_data['stacking_keys']:
            stacking = Stacking.load_from_file(exp, stacking_name)
            exp.stacking[stacking_name] = stacking

        exp.logger.info(f"Loaded: {len(exp.nodes) - 1} node(s), {len(exp.grps)} group(s), {len(exp.train_idx_list)} fold(s)")

        return exp

    def get_result(self, node, idx, result, params = {}):
        return self.nodes[node].get_result(idx, result, params)

    def get_results(self, node, result, params = {}):
        for i in range(self.get_n_splits()):
            yield list(
                self.get_result(node, i, result, params)
            )

    def get_results_agg(self, node, result, params = {}, agg_inner = True, agg_outer = True):
        if agg_outer and not agg_inner:
            raise ValueError("agg_outer requires agg_inner to be True")
        if not self.nodes[node].adapter_.result_objs[result][1]:
            raise ValueError(f"Result '{result}' is not mergeable across folds")
        l = list()
        for no, i in enumerate(self.get_results(node, result, params)):
            l.append(pd.concat([j.rename(no_i) for no_i, j in enumerate(i)], axis = 1).stack().rename(no))
        df = pd.concat(l, axis=1)
        if agg_inner:
            df = df.groupby(level=[i for i in range(len(df.index.levels) - 1)]).mean()
            if agg_outer:
                return df.mean(axis=1)
        return df

def create_like(exp, data, path, data_names=None, sp=None, sp_v=None, splitter_params=None, title=None, data_key=None):
    """기존 Experimenter의 구조를 복제하여 새로운 Experimenter 생성

    Args:
        exp: 구조를 복제할 원본 Experimenter
        data: 새로운 데이터
        path: 작업 디렉토리 경로
        data_names: 새로운 데이터의 컬럼명 (None이면 자동)
        sp: 외부 splitter (None이면 원본과 동일)
        sp_v: 내부 splitter (None이면 원본과 동일, "remove"이면 제거)
        splitter_params: splitter에 전달할 파라미터 (None이면 원본과 동일)
        title: 실험 타이틀 (None이면 원본과 동일)
        data_key: 데이터 식별자 (None이면 원본과 동일)

    Returns:
        Experimenter: 새로 생성된 Experimenter 인스턴스
    """
    exp.logger.info("Creating new Experimenter with same structure...")

    if sp is None:
        sp = exp.sp
    if sp_v is None:
        sp_v = exp.sp_v
    elif sp_v == "remove":
        sp_v = None
    if splitter_params is None:
        splitter_params = exp.splitter_params.copy() if exp.splitter_params else None
    if title is None:
        title = exp.title
    if data_key is None:
        data_key = exp.data_key

    # 새 Experimenter 생성
    new_exp = Experimenter(
        data,
        path=path,
        data_names=data_names,
        sp=sp,
        sp_v=sp_v,
        splitter_params=splitter_params,
        title=title,
        data_key=data_key
    )
    new_exp.logger.info(f"Created base Experimenter with {len(new_exp.train_idx_list)} fold(s)")

    # 그룹 복제 (부모-자식 관계를 유지하기 위해 위상 정렬)
    # 1. 최상위 그룹부터 BFS로 복제
    grp_mapping = {}  # 원본 그룹명 -> 새 그룹 객체

    # 최상위 그룹 찾기 (parent_grp가 None인 그룹)
    top_level_grps = [grp for grp in exp.grps.values() if grp.parent_grp is None]

    def clone_group_recursive(orig_grp, parent_grp_name=None):
        """그룹을 재귀적으로 복제"""
        new_grp = new_exp.set_grp(
            name=orig_grp.name,
            role=orig_grp.role,
            processor=orig_grp.processor,
            edges=orig_grp.edges[:],  # 리스트 복사
            X=orig_grp.X,
            y=orig_grp.y,
            method=orig_grp.method,
            parent_grp=parent_grp_name,
            adapter=orig_grp.adapter,
            params=orig_grp.params.copy() if orig_grp.params else {}
        )
        grp_mapping[orig_grp.name] = new_grp

        # 자식 그룹들도 복제
        for child_grp in orig_grp.child_grps:
            clone_group_recursive(child_grp, parent_grp_name=orig_grp.name)

    # 최상위 그룹부터 재귀적으로 복제
    for grp in top_level_grps:
        clone_group_recursive(grp)

    new_exp.logger.info(f"Cloned {len(exp.grps)} group(s)")

    # 노드 복제 (위상 정렬: Root부터 BFS)
    # 1. 노드의 우선순위 계산 (BFS)
    node_priorities = {}
    queue = [(None, 0)]  # Root부터 시작

    while queue:
        current_node, priority = queue.pop(0)

        if current_node in node_priorities:
            continue

        node_priorities[current_node] = priority

        # current_node를 edge로 참조하는 child 노드들 찾기
        for name in current_node.output_edges :
            if name is not None and name not in node_priorities:
                queue.append((name, priority + 1))
                break

    # 우선순위 순으로 노드 정렬 (Root 제외)
    sorted_nodes = sorted(
        [(name, node) for name, node in exp.nodes.items() if name is not None],
        key=lambda x: node_priorities.get(x[0], float('inf'))
    )

    # 노드 복제
    for name, orig_node in sorted_nodes:
        if orig_node.org_attr is not None:
            org = orig_node.org_attr
            new_exp.set_node(
                name,
                grp=orig_node.grp.name,
                processor=org['processor'],
                edges=org['edges'][:] if isinstance(org['edges'], list) else [org['edges']] if org['edges'] else [],
                X=org['X'],
                y=org['y'],
                method=org['method'],
                adapter=org['adapter'],
                params=org['params'].copy() if org['params'] else {}
            )

    new_exp.logger.info(f"Structure cloning complete: {len(exp.grps)} group(s), {len(sorted_nodes)} node(s)")

    return new_exp