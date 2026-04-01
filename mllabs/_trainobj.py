import os
import time
import uuid
import pickle as pkl

from ._node_processor import TransformProcessor, PredictProcessor


def _train_build(node_attrs, data_dict, logger):
    method = node_attrs['method']
    if method in ['transform', 'fit_transform']:
        fit_process = method == 'fit_transform'
        obj = TransformProcessor(
            node_attrs['name'], node_attrs['processor'],
            node_attrs['adapter'], node_attrs['params'],
        )
    elif method in ['predict', 'predict_proba', 'fit_predict']:
        fit_process = method == 'fit_predict'
        obj = PredictProcessor(
            node_attrs['name'], node_attrs['processor'],
            method, node_attrs['adapter'], node_attrs['params'],
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    start_time = time.time()
    if fit_process:
        result = obj.fit_process(data_dict, logger=logger)
    else:
        obj.fit(data_dict, logger=logger)
        result = None
    elapsed_time = time.time() - start_time

    _ref_key = 'X' if 'X' in data_dict else 'y'
    train_, train_v_ = data_dict[_ref_key]
    info = {
        'build_id': str(uuid.uuid4()),
        'fit_time': elapsed_time,
        'train_shape': train_.get_shape() if train_ is not None else None,
        'train_v_shape': train_v_.get_shape() if train_v_ is not None else None,
    }
    return obj, result, info


class TrainStageObj:

    def __init__(self, path):
        self.path = path
        self.status = None
        self.objs_ = None

    def load(self):
        if not os.path.isdir(self.path):
            self.objs_ = None
            return
        self.objs_ = {}
        split_idx = 0
        while True:
            obj_file = self.path / f'obj{split_idx}.pkl'
            if not os.path.isfile(obj_file):
                break
            with open(obj_file, 'rb') as f:
                obj = pkl.load(f)
            with open(self.path / f'result{split_idx}.pkl', 'rb') as f:
                result = pkl.load(f)
            with open(self.path / f'info{split_idx}.pkl', 'rb') as f:
                info = pkl.load(f)
            self.objs_[split_idx] = (obj, result, info)
            split_idx += 1
        if self.objs_:
            self.status = 'built'

    def start_build(self):
        self.objs_ = {}
        os.makedirs(self.path, exist_ok=True)

    def build_split(self, split_idx, node_attrs, data_dict, logger):
        obj, result, info = _train_build(node_attrs, data_dict, logger)
        info_with_status = {**info, 'status': 'built'}
        with open(self.path / f'obj{split_idx}.pkl', 'wb') as f:
            pkl.dump(obj, f)
        with open(self.path / f'result{split_idx}.pkl', 'wb') as f:
            pkl.dump(result, f)
        with open(self.path / f'info{split_idx}.pkl', 'wb') as f:
            pkl.dump(info_with_status, f)
        self.objs_[split_idx] = (obj, result, info_with_status)

    def end_build(self):
        self.status = 'built'

    def get_obj(self):
        for i in self.objs_:
            yield self.objs_[i]


class TrainHeadObj:

    def __init__(self, path):
        self.path = path
        self.status = None

    def load(self):
        if not os.path.isdir(self.path):
            return
        if os.path.isfile(self.path / 'obj0.pkl'):
            self.status = 'built'

    def start_build(self):
        os.makedirs(self.path, exist_ok=True)

    def build_split(self, split_idx, node_attrs, data_dict, logger):
        obj, result, info = _train_build(node_attrs, data_dict, logger)
        info_with_status = {**info, 'status': 'built'}
        with open(self.path / f'obj{split_idx}.pkl', 'wb') as f:
            pkl.dump(obj, f)
        with open(self.path / f'result{split_idx}.pkl', 'wb') as f:
            pkl.dump(result, f)
        with open(self.path / f'info{split_idx}.pkl', 'wb') as f:
            pkl.dump(info_with_status, f)

    def end_build(self):
        self.status = 'built'

    def get_obj(self):
        no = 0
        while True:
            obj_file = self.path / f'obj{no}.pkl'
            if not os.path.isfile(obj_file):
                break
            with open(obj_file, 'rb') as f:
                obj = pkl.load(f)
            with open(self.path / f'result{no}.pkl', 'rb') as f:
                result = pkl.load(f)
            with open(self.path / f'info{no}.pkl', 'rb') as f:
                info = pkl.load(f)
            yield obj, result, info
            no += 1
