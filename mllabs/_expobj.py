import uuid
import json
import os
import time
import pickle as pkl
import traceback
import warnings
from multiprocessing import Process
from multiprocessing.connection import wait


def _build(node_attrs, train_data, valid_data, fit_process, logger, gpu_id_list=None):
    from ._node_processor import TransformProcessor, PredictProcessor
    method = node_attrs['method']
    if method in ['transform', 'fit_transform']:
        obj = TransformProcessor(node_attrs['name'], node_attrs['processor'], node_attrs['adapter'], node_attrs['params'])
    else:
        obj = PredictProcessor(node_attrs['name'], node_attrs['processor'], method, node_attrs['adapter'], node_attrs['params'])

    start_time = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if fit_process:
                result = obj.fit_process(train_data, valid_data, gpu_id_list=gpu_id_list, logger=logger)
            else:
                result = None
                obj.fit(train_data, valid_data, gpu_id_list=gpu_id_list, logger=logger)
        except Exception as e:
            warn_msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
            info = {
                'build_id': str(uuid.uuid4()),
                'fit_time': time.time() - start_time,
                'train_shape': None,
                'edges': node_attrs.get('edges'),
                'error': {
                    'type': type(e).__name__,
                    'message': str(e),
                    'traceback': traceback.format_exc(),
                },
            }
            if warn_msgs:
                info['warnings'] = warn_msgs
            return None, 'error', info

    elapsed_time = time.time() - start_time
    _ref_key = 'X' if 'X' in train_data else 'y'
    ref_data = train_data[_ref_key]
    info = {
        'build_id': str(uuid.uuid4()),
        'fit_time': elapsed_time,
        'train_shape': ref_data.get_shape() if ref_data is not None else None,
        'edges': node_attrs.get('edges'),
    }
    warn_msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
    if warn_msgs:
        info['warnings'] = warn_msgs
    return obj, result, info


def _safe_collect_call(collector, context, logger):
    try:
        return collector.collect(context)
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"[Collector:{collector.name}] collect failed: {type(e).__name__}: {e}\n{tb}"
        logger.warning(msg)
        collector.warnings.append({
            'method': 'collect', 'type': type(e).__name__, 'message': str(e), 'traceback': tb,
        })
        return None


def _run_collectors(collectors, node_attrs, obj, result, info, train_data, valid_data,
                    idx, inner_idx, logger,
                    include_input=True, include_output=True, include_train=True):
    node_name = node_attrs['name']
    matched = [c for c in collectors if c.connector.match(node_name, node_attrs)]
    if not matched:
        return {}
    context = {
        'node_attrs': node_attrs,
        'processor': obj,
        'spec': info,
        'input': train_data if include_input else None,
        'idx': idx,
        'inner_idx': inner_idx,
    }
    if include_output:
        context['output_valid'] = obj.process(valid_data) if valid_data else None
        context['output_train'] = (result if result is not None else obj.process(train_data)) if include_train else None
    return {c.name: _safe_collect_call(c, context, logger) for c in matched}


class _PipeLogger:
    def __init__(self, conn):
        self._conn = conn

    def adhoc_progress(self, current, total, metrics=None):
        self._conn.send(('progress', current, total, metrics))

    def warning(self, msg):
        self._conn.send(('warning', msg))

    def info(self, msg):
        self._conn.send(('info', msg))


class BuildWorker(Process):
    """Process-based worker. Receives jobs via Pipe, reports results/progress back.

    Job tuple: ``(node_path, file, node_attrs, idx, no, train_data, valid_data)``
    Sentinel ``None`` stops the worker.

    Messages sent to main via conn:
        ('progress', current, total, metrics)
        ('warning', msg)
        ('info', msg)
        ('done', info)
        ('error', error_info)
    """

    def __init__(self, conn, collectors,
                 include_output=False, include_train=True, include_input=True, finalize=False,
                 gpu_id=None):
        super().__init__(daemon=True)
        self.conn = conn
        self.collectors = collectors
        self.include_output = include_output
        self.include_train = include_train
        self.include_input = include_input
        self.finalize = finalize
        self.gpu_id = gpu_id

    def run(self):
        logger = _PipeLogger(self.conn)
        gpu_id_list = [self.gpu_id] if self.gpu_id is not None else []
        while True:
            job = self.conn.recv()
            if job is None:
                break
            node_path, file, node_attrs, idx, no, train_data, valid_data = job
            node_name = node_attrs['name']
            method = node_attrs['method']
            fit_process = method in ['fit_transform', 'fit_predict']
            os.makedirs(node_path, exist_ok=True)
            obj, result, info = _build(node_attrs, train_data, valid_data, fit_process, logger, gpu_id_list)
            for w in info.get('warnings', []):
                logger.warning(f"[{node_name}] fold {idx}: {w}")
            if obj is None:
                error_info = {**info['error'], 'fold': idx}
                with open(node_path / 'error.txt', 'w') as f:
                    json.dump(error_info, f, ensure_ascii=False, indent=2)
                self.conn.send(('error', error_info))
                continue

            save_obj = None if (self.include_output and self.finalize) else obj
            with open(file, 'wb') as f:
                pkl.dump((save_obj, result, info), f)

            coll_dict = _run_collectors(
                self.collectors, node_attrs, obj, result, info, train_data, valid_data,
                idx, no, logger,
                include_input=self.include_input, include_output=self.include_output,
                include_train=self.include_train,
            )
            if coll_dict:
                with open(node_path / f'_collect_{idx}_{no}.pkl', 'wb') as f:
                    pkl.dump(coll_dict, f)

            self.conn.send(('done', info))

# ---------------------------------------------------------------------------
# Flow build
# ---------------------------------------------------------------------------

def _is_stage_ready(flow, pipeline, node_name):
    for edge_list in pipeline.get_node_attrs(node_name)['edges'].values():
        for src_name, _ in edge_list:
            if src_name is None:
                continue
            if pipeline.grps[pipeline.nodes[src_name].grp].role == 'stage':
                if src_name not in flow.node_objs:
                    return False
    return True


def _build_flow_single(flows, pipeline, nodes, gpu_id_list=None, collectors=None, logger=None):
    from ._flow import TrainDataFlow

    gpu_id_list = gpu_id_list or []
    collectors = collectors or []

    errors = {}

    while True:
        ready = [
            (outer_idx, inner_idx, flow, n)
            for outer_idx, inner_idx, flow in flows for n in nodes
            if n not in flow.node_objs and (outer_idx, inner_idx, n) not in errors
            and _is_stage_ready(flow, pipeline, n)
        ]
        if not ready:
            break

        for outer_idx, inner_idx, flow, node_name in ready:
            node_attrs = pipeline.get_node_attrs(node_name)
            train_data = flow.get_train(node_attrs['edges'])
            valid_data = flow.get_valid(node_attrs['edges'])
            fit_process = node_attrs['method'] in ['fit_transform', 'fit_predict']

            obj, result, info = _build(node_attrs, train_data, valid_data, fit_process, logger, gpu_id_list)
            for w in info.get('warnings', []):
                if logger:
                    logger.warning(f"[{node_name}] fold {outer_idx}_{inner_idx}: {w}")
            if obj is None:
                errors[(outer_idx, inner_idx, node_name)] = info['error']
                if logger:
                    logger.info(f"[{node_name}] Build error at fold {outer_idx}_{inner_idx}: {info['error']['type']}: {info['error']['message']}")
                continue

            TrainDataFlow.write_objs(flow.get_objs_file(node_name), (obj, result, info))
            flow.set_objs(node_name, obj, result, info)

            if collectors:
                _run_collectors(collectors, node_attrs, obj, result, info, train_data, valid_data,
                                outer_idx, inner_idx, logger)

    return errors


def _build_flow_multi(flows, pipeline, nodes, n_jobs, gpu_id_list=None, collectors=None, logger=None,
                      gpu_fallback_cpu=True, cpu_fallback_gpu=True):
    from .adapter._base import GPU_NO

    gpu_id_list = gpu_id_list or []
    collectors = collectors or []
    n_gpu = len(gpu_id_list)

    flow_map = {(outer_idx, inner_idx): flow for outer_idx, inner_idx, flow in flows}

    def _needs_gpu(node_attrs):
        if not gpu_id_list:
            return False
        adapter = node_attrs.get('adapter')
        return adapter is not None and adapter.get_gpu_usage(node_attrs.get('params')) != GPU_NO

    workers = []  # [(process, parent_conn)]
    for i in range(n_jobs):
        parent_conn, child_conn = Pipe()
        w = BuildWorker(child_conn, collectors,
                        include_output=True, include_train=True, include_input=True,
                        gpu_id=gpu_id_list[i] if i < n_gpu else None)
        w.start()
        workers.append((w, parent_conn))

    free_gpu = list(range(n_gpu))
    free_cpu = list(range(n_gpu, n_jobs))
    busy = {}   # parent_conn -> (outer_idx, inner_idx, node_name)
    errors = {}
    all_conns = [conn for _, conn in workers]

    def _collect_ready():
        in_flight = set(busy.values())
        gpu_ready, cpu_ready = [], []
        for outer_idx, inner_idx, flow in flows:
            for n in nodes:
                key = (outer_idx, inner_idx, n)
                if n in flow.node_objs or key in errors or key in in_flight:
                    continue
                if not _is_stage_ready(flow, pipeline, n):
                    continue
                node_attrs = pipeline.get_node_attrs(n)
                (gpu_ready if _needs_gpu(node_attrs) else cpu_ready).append(key)
        return gpu_ready, cpu_ready

    def _dispatch(outer_idx, inner_idx, node_name, worker_idx):
        flow = flow_map[(outer_idx, inner_idx)]
        node_attrs = pipeline.get_node_attrs(node_name)
        train_data = flow.get_train(node_attrs['edges'])
        valid_data = flow.get_valid(node_attrs['edges'])
        _, conn = workers[worker_idx]
        file = flow.get_objs_file(node_name)
        conn.send((file.parent, file, node_attrs, outer_idx, inner_idx, train_data, valid_data))
        busy[conn] = (outer_idx, inner_idx, node_name)
        (free_gpu if worker_idx < n_gpu else free_cpu).remove(worker_idx)

    def _try_dispatch():
        gpu_ready, cpu_ready = _collect_ready()
        for outer_idx, inner_idx, node_name in gpu_ready:
            if free_gpu:
                _dispatch(outer_idx, inner_idx, node_name, free_gpu[0])
            elif free_cpu and gpu_fallback_cpu:
                _dispatch(outer_idx, inner_idx, node_name, free_cpu[0])
        for outer_idx, inner_idx, node_name in cpu_ready:
            if free_cpu:
                _dispatch(outer_idx, inner_idx, node_name, free_cpu[0])
            elif free_gpu and cpu_fallback_gpu:
                _dispatch(outer_idx, inner_idx, node_name, free_gpu[0])

    _try_dispatch()

    while busy:
        for conn in wait(all_conns):
            msg_type, *data = conn.recv()
            worker_idx = next(i for i, (_, c) in enumerate(workers) if c is conn)
            outer_idx, inner_idx, node_name = busy[conn]

            if msg_type == 'done':
                flow_map[(outer_idx, inner_idx)].load_objs(node_name)
                del busy[conn]
                (free_gpu if worker_idx < n_gpu else free_cpu).append(worker_idx)
                _try_dispatch()

            elif msg_type == 'error':
                error_info = data[0]
                errors[(outer_idx, inner_idx, node_name)] = error_info
                del busy[conn]
                (free_gpu if worker_idx < n_gpu else free_cpu).append(worker_idx)
                if logger:
                    logger.info(f"[{node_name}] Build error at fold {outer_idx}_{inner_idx}: {error_info['type']}: {error_info['message']}")
                _try_dispatch()

            elif msg_type == 'progress':
                if logger:
                    logger.adhoc_progress(*data)
            elif msg_type == 'warning':
                if logger:
                    logger.warning(data[0])
            elif msg_type == 'info':
                if logger:
                    logger.info(data[0])

    for _, conn in workers:
        conn.send(None)
    for w, _ in workers:
        w.join()

    return errors


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


def exp_node(path, node_attrs, idx, data_dict_it, collectors, logger,
             finalize=False, include_train=True, include_input=True):
    """Run one outer fold for a head node. Returns True on success, False on error."""
    node_name = node_attrs['name']
    node_path = path / node_name

    matched = [c for c in collectors if c.connector.match(node_name, node_attrs)]

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            os.makedirs(node_path, exist_ok=True)
            if (node_path / f'obj{idx}_0.pkl').exists():
                _dispatch_iter_output(node_path, node_attrs, idx, data_dict_it, matched, logger,
                                      include_train=include_train, include_input=include_input)
            else:
                for _ in _build_iter(node_path, node_attrs, idx, data_dict_it, matched, logger,
                                     include_output=True, include_train=include_train,
                                     include_input=include_input, finalize=finalize):
                    pass
        for w in caught:
            logger.warning(f"[{node_name}] fold {idx}: {w.category.__name__}: {w.message}")
    except Exception as e:
        set_head_error(path, node_name, {
            'type': type(e).__name__,
            'message': str(e),
            'traceback': traceback.format_exc(),
            'fold': idx,
        })
        logger.info(f"[{node_name}] Exp error at fold {idx}: {type(e).__name__}: {e}")
        return False

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


