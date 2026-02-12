import re
import pickle
import pandas as pd
from ._node_processor import resolve_columns

class Metric:
    def __init__(
        self, name, experimenter, target_vars, output_var, metric_func, include_train = False
    ):
        self.experimenter = experimenter
        self.name = name
        self.target_vars = target_vars
        self.output_var = output_var
        self.include_train = include_train
        self.metric_func = metric_func
        self.metrics = dict()

    @property
    def metric_dir(self):
        """metric 데이터를 저장할 디렉토리 경로"""
        return self.experimenter.path / "__metric"

    @property
    def metric_path(self):
        """이 metric의 저장 파일 경로"""
        return self.metric_dir / f"{self.name}.pkl"

    def _ensure_metric_dir(self):
        """metric 디렉토리가 존재하는지 확인하고 없으면 생성"""
        if not self.metric_dir.exists():
            self.metric_dir.mkdir(parents=True, exist_ok=True)
        return self.metric_dir

    def save(self):
        """name, target_vars, output_var, include_train, metric_func, metrics를 파일로 저장"""
        self._ensure_metric_dir()
        data = {
            'name': self.name,
            'target_vars': self.target_vars,
            'output_var': self.output_var,
            'include_train': self.include_train,
            'metric_func': self.metric_func,
            'metrics': self.metrics
        }
        with open(self.metric_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        """파일에서 name, target_vars, output_var, include_train, metric_func, metrics를 로드"""
        if not self.metric_path.exists():
            raise FileNotFoundError(f"Metric data not found: {self.metric_path}")
        with open(self.metric_path, 'rb') as f:
            data = pickle.load(f)
        self.name = data['name']
        self.target_vars = data['target_vars']
        self.output_var = data['output_var']
        self.include_train = data['include_train']
        self.metric_func = data['metric_func']
        self.metrics = data['metrics']

    @classmethod
    def load_from_file(cls, experimenter, name):
        """저장된 Metric 정보를 불러와서 Metric 인스턴스 생성

        Args:
            experimenter: Experimenter 인스턴스
            name: metric 이름

        Returns:
            Metric: 복원된 Metric 인스턴스
        """
        filepath = experimenter.path / "__metric" / f"{name}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Metric data not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Metric 인스턴스 생성
        metric = cls(
            name=data['name'],
            experimenter=experimenter,
            target_vars=data['target_vars'],
            output_var=data['output_var'],
            metric_func=data['metric_func'],
            include_train=data['include_train']
        )
        metric.metrics = data['metrics']

        return metric

    def _get_data(self, idx):
        # target_vars를 임시 key로 감싸서 get_data 호출
        temp_edges = {'_target': self.target_vars}
        result = list(self.experimenter.get_data(idx, temp_edges))
        # '_target' key의 데이터만 추출
        return [data_dict['_target'] for data_dict in result]

    def _get_metric(self, target_data, result_data):
        (true_train_t, true_train_v), true_valid = target_data
        selected_cols = resolve_columns(result_data['output_valid'], self.output_var)
        if len(selected_cols) == 0:
            return None
        prd = result_data['output_valid'].select_columns(selected_cols)
        result = {
            'valid': self.metric_func(true_valid.data, prd.data)
        }
        if self.include_train:
            prd = result_data['output_train'][0].select_columns(selected_cols)
            result['train_sub'] = self.metric_func(true_train_t.data, prd.data)
            if true_train_v is not None:
                prd = result_data['output_train'][1].select_columns(selected_cols)
                result['valid_sub'] = self.metric_func(true_train_v.data, prd.data)
        return result

    def _start(self, node):
        self.metrics[node] = list()

    def _set_metric(self, node, idx, metric):
        l = self.metrics[node]
        if len(l) != idx:
            raise RuntimeError("")
        l.append(metric)

    def _end(self, node):
        if len(self.metrics[node]) == 0:
            del self.metrics[node]
        self.save()

    def _get_nodes(self, nodes):
        if nodes is None:
            # 기존 동작: 모든 노드
            node_names = list(self.metrics.keys())
        elif isinstance(nodes, list):
            node_names = [n for n in nodes if n in self.metrics]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            node_names = [k for k in self.metrics.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        return node_names
    
    def reset_nodes(self, nodes):
        for node in nodes:
            if node not in self.metrics:
                continue
            del self.metrics[node]
        self.save()

    def get_metric(self, node):
        l = list()
        for i, sub in enumerate(self.metrics[node]):
            l.append(
                pd.concat([pd.Series(j, name = str(no)) for no, j in enumerate(sub)], axis=1).unstack()
            )
        return pd.concat(l, axis=1).unstack(level=[0, 1]).rename(node)

    def get_metrics(self, nodes):
        node_names = self._get_nodes(nodes)
        return pd.concat([self.get_metric(node) for node in node_names], axis=1).T

    def get_metrics_agg(self, nodes, inner_fold = True, outer_fold = True, include_std = False):
        if outer_fold and not inner_fold:
            raise ValueError("")
        df = self.get_metrics(nodes)
        if inner_fold:
            df_agg_mean = df.stack(level = 1).groupby(level=0).mean()
            if include_std:
                df_agg_std = df.stack(level = 1).groupby(level=0).std()
            else:
                df_agg_std = None
            if outer_fold:
                df_agg_mean = df_agg_mean.stack(level=0).groupby(level=0).mean()
                if include_std:
                    df_agg_std = df_agg_std.stack(level = 0).groupby(level=0).mean()
            return df_agg_mean, df_agg_std
        return df