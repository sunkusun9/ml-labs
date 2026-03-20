import uuid
import json
import os
import shutil
import time
import pickle as pkl
import traceback
import warnings


def _get_process_inputs(data_dict):
    key = 'X' if 'X' in data_dict else 'y'
    (train, train_v), valid = data_dict[key]
    return train, train_v, valid


def _build_sub(node_attrs, data_dict, fit_process, logger):
    from ._node_processor import TransformProcessor, PredictProcessor
    method = node_attrs['method']
    if method in ['transform', 'fit_transform']:
        obj = TransformProcessor(node_attrs['name'], node_attrs['processor'], node_attrs['adapter'], node_attrs['params'])
    else:
        obj = PredictProcessor(node_attrs['name'], node_attrs['processor'], method, node_attrs['adapter'], node_attrs['params'])

    fit_data = {key: val[0] for key, val in data_dict.items()}

    start_time = time.time()
    if fit_process:
        result = obj.fit_process(fit_data, logger=logger)
    else:
        result = None
        obj.fit(fit_data, logger=logger)
    elapsed_time = time.time() - start_time

    _ref_key = 'X' if 'X' in data_dict else 'y'
    train_, train_v_ = fit_data[_ref_key]
    info = {
        'build_id': str(uuid.uuid4()),
        'fit_time': elapsed_time,
        'train_shape': train_.get_shape() if train_ is not None else None,
        'train_v_shape': train_v_.get_shape() if train_v_ is not None else None
    }
    return obj, result, info


def _build_iter(node_attrs, data_dict_it, logger):
    method = node_attrs['method']
    if method in ['transform', 'predict', 'predict_proba']:
        fit_process = False
    elif method in ['fit_transform', 'fit_predict']:
        fit_process = True
    else:
        raise ValueError(f"Unknown processor_type: {method}")

    for data_dict in data_dict_it:
        yield _build_sub(node_attrs, data_dict, fit_process, logger), data_dict


def _build_iter_output(node_attrs, data_dict_it, logger, include_train=True):
    method = node_attrs['method']
    if method in ['transform', 'predict', 'predict_proba']:
        fit_process = False
    elif method in ['fit_transform', 'fit_predict']:
        fit_process = True
    else:
        raise ValueError(f"Unknown processor_type: {method}")

    for data_dict in data_dict_it:
        obj, result, info = _build_sub(node_attrs, data_dict, fit_process, logger)
        train_X, train_v_X, valid_X = _get_process_inputs(data_dict)
        if include_train:
            if result is None:
                train_result = obj.process(train_X)
            else:
                train_result = result
            train_v_result = obj.process(train_v_X) if train_v_X is not None else None
            output_train = (train_result, train_v_result)
        else:
            output_train = None
        output_valid = obj.process(valid_X)
        yield obj, result, info, data_dict, output_train, output_valid


# ---------------------------------------------------------------------------
# Collector safe call
# ---------------------------------------------------------------------------

def _safe_collector_call(collector, node, method, logger, *args):
    try:
        getattr(collector, method)(node, *args)
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"[Collector:{collector.name}] [{node}] {method} failed: {type(e).__name__}: {e}\n{tb}"
        logger.warning(msg)
        collector.warnings.append({
            'method': method, 'node': node,
            'type': type(e).__name__, 'message': str(e), 'traceback': tb,
        })


# ---------------------------------------------------------------------------
# Head node — function-based management
# ---------------------------------------------------------------------------

def get_head_status(path, name):
    """Returns 'built'/'finalized'/'error'/None by checking disk."""
    node_path = path / name
    if not os.path.isdir(node_path):
        return None
    if (node_path / 'error.txt').exists():
        return 'error'
    if (node_path / 'finalized.pkl').exists():
        return 'finalized'
    if (node_path / 'obj0_0.pkl').exists():
        return 'built'
    return None


def get_head_error(path, name):
    error_path = path / name / 'error.txt'
    if error_path.exists():
        with open(error_path) as f:
            return json.load(f)
    return None


def set_head_error(path, name, error_info):
    node_path = path / name
    os.makedirs(node_path, exist_ok=True)
    with open(node_path / 'error.txt', 'w') as f:
        json.dump(error_info, f, ensure_ascii=False, indent=2)


def finalize_head(path, name):
    """Read specs from saved pkls → save finalized.pkl → remove obj pkls."""
    node_path = path / name
    specs = {}
    for fpath in sorted(node_path.glob('obj*.pkl')):
        outer, inner = map(int, fpath.stem[3:].split('_'))
        with open(fpath, 'rb') as f:
            _, _, spec = pkl.load(f)
        specs[(outer, inner)] = spec
    with open(node_path / 'finalized.pkl', 'wb') as f:
        pkl.dump(specs, f)
    for fpath in node_path.glob('obj*.pkl'):
        fpath.unlink()


def exp_node(node_attrs, data_dict_it, idx, logger,
             collectors=None, finalize=False, include_train=True, include_input=True):
    """Run one outer fold for a head node. Returns True on success, False on error."""
    node_name = node_attrs['name']
    node_path = node_attrs['path'] / node_name

    matched = [c for c in (collectors or {}).values()
               if c.connector.match(node_name, node_attrs)]

    no = 0

    first_file = node_path / f'obj{idx}_0.pkl'
    if first_file.exists():
        for data_dict in data_dict_it:
            filename = node_path / f'obj{idx}_{no}.pkl'
            if not filename.exists():
                break
            with open(filename, 'rb') as f:
                obj, train_, spec = pkl.load(f)
            train_X, train_v_X, valid_X = _get_process_inputs(data_dict)
            if include_train:
                train_result = train_ if train_ is not None else obj.process(train_X)
                train_v_result = obj.process(train_v_X) if train_v_X is not None else None
                output_train = (train_result, train_v_result)
            else:
                output_train = None
            output_valid = obj.process(valid_X)
            if matched:
                context = {
                    'node_attrs': node_attrs, 'processor': obj, 'spec': spec,
                    'input': data_dict if include_input else None,
                    'output_train': output_train,
                    'output_valid': output_valid,
                }
                for collector in matched:
                    _safe_collector_call(collector, node_name, '_collect', logger, idx, no, context)
            no += 1
    else:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                os.makedirs(node_path, exist_ok=True)
                for obj, result, info, data_dict, output_train, output_valid in _build_iter_output(node_attrs, data_dict_it, logger, include_train=include_train):
                    if matched:
                        context = {
                            'node_attrs': node_attrs, 'processor': obj, 'spec': info,
                            'input': data_dict if include_input else None,
                            'output_train': output_train,
                            'output_valid': output_valid,
                        }
                        for collector in matched:
                            _safe_collector_call(collector, node_name, '_collect', logger, idx, no, context)
                    save_obj = None if finalize else obj
                    filename = node_path / f'obj{idx}_{no}.pkl'
                    with open(filename, 'wb') as f:
                        pkl.dump((save_obj, result, info), f)
                    no += 1
            for w in caught:
                logger.warning(f"[{node_name}] fold {idx}: {w.category.__name__}: {w.message}")
        except Exception as e:
            set_head_error(node_attrs['path'], node_name, {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc(),
                'fold': idx,
            })
            logger.info(f"[{node_name}] Exp error at fold {idx}: {type(e).__name__}: {e}")
            return False

    for collector in matched:
        _safe_collector_call(collector, node_name, '_end_idx', logger, idx)

    return True


def get_head_objs(path, name, idx):
    node_path = path / name
    no = 0
    while True:
        filename = node_path / f'obj{idx}_{no}.pkl'
        if not os.path.isfile(filename):
            break
        with open(filename, 'rb') as f:
            obj, train_, spec = pkl.load(f)
        yield obj, train_, spec
        no += 1


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class StageObj():
    def __init__(self, path):
        self.path = path
        self.status = None
        self.error = None

    def set_error(self, error_info):
        self.error = error_info
        self.status = 'error'
        if not os.path.isdir(self.path):
            os.makedirs(self.path, exist_ok=True)
        with open(self.path / 'error.txt', 'w') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.isdir(self.path):
            self.status = 'finalized'
            self.objs_ = None
            return

        error_path = self.path / 'error.txt'
        if error_path.exists():
            with open(error_path, 'r') as f:
                self.error = json.load(f)
            self.status = 'error'
            self.objs_ = None
            return

        self.objs_ = []
        idx = 0
        while True:
            objs = []
            no = 0
            while True:
                filename = self.path / ('obj' + str(idx) + '_' + str(no) + '.pkl')
                if not os.path.isfile(filename):
                    break
                with open(filename, 'rb') as f:
                    objs.append(pkl.load(f))
                no += 1
            if len(objs) == 0:
                break
            self.objs_.append(objs)
            idx += 1

        if len(self.objs_) == 0:
            self.status = 'finalized'
            self.objs_ = None
        else:
            self.status = 'built'

    def exp_idx(self, idx, node_attrs, data_dict_it, logger, include_input=True, include_output=True):
        if self.status == "built":
            for data_dict, (obj, train_, spec) in zip(data_dict_it, self.objs_[idx]):
                train_X, train_v_X, valid_X = _get_process_inputs(data_dict)
                sub_result = {'spec': spec, 'object': obj}
                if include_output:
                    if train_ is None:
                        train_result = obj.process(train_X)
                    else:
                        train_result = train_
                    if train_v_X is not None:
                        train_v_result = obj.process(train_v_X)
                    else:
                        train_v_result = None
                    sub_result['output_train'] = (train_result, train_v_result)
                    sub_result['output_valid'] = obj.process(valid_X)
                if include_input:
                    sub_result['input'] = data_dict
                yield sub_result
        elif self.status == "finalized":
            raise RuntimeError(f"Node is finalized and cannot be re-experimented")
        else:
            raise RuntimeError(f"StageObj cannot be experimented unless built")

    def start_build(self):
        self.objs_ = list()
        if not os.path.isdir(self.path):
            os.makedirs(self.path, exist_ok=True)

    def build_idx(self, idx, node_attrs, data_dict_it, logger):
        if idx != len(self.objs_):
            raise RuntimeError(f"Build sequence is not valid")
        objs = list()
        for no, (obj, _) in enumerate(_build_iter(node_attrs, data_dict_it, logger)):
            filename = self.path / ('obj' + str(idx) + '_' + str(no) + '.pkl')
            with open(filename, 'wb') as f:
                pkl.dump(obj, f)
            objs.append(obj)
        self.objs_.append(objs)

    def end_build(self):
        self.status = 'built'
        error_path = self.path / 'error.txt'
        if error_path.exists():
            error_path.unlink()

    def get_objs(self, idx):
        for obj, train_, spec in self.objs_[idx]:
            yield obj, train_, spec

    def finalize(self):
        self.status = 'finalized'
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        self.objs_ = None
