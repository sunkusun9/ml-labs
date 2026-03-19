import os


def _get_cuda_visible_devices():
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env is None:
        return None
    if env.strip() in ('', 'NoDevFiles'):
        return []
    return [int(x.strip()) for x in env.split(',')]


def _detect_gpus_pynvml(visible):
    import pynvml
    pynvml.nvmlInit()
    try:
        total = pynvml.nvmlDeviceGetCount()
        physical_ids = list(range(total)) if visible is None else [i for i in visible if i < total]
        result = []
        for cuda_id, phys_id in enumerate(physical_ids):
            handle = pynvml.nvmlDeviceGetHandleByIndex(phys_id)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result.append({'cuda_id': cuda_id, 'physical_id': phys_id, 'free': mem.free, 'total': mem.total})
        return result
    finally:
        pynvml.nvmlShutdown()


def _detect_gpus_nvidia_smi(visible):
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'],
            timeout=10
        ).decode()
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return []

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 3:
            phys_id, free_mb, total_mb = int(parts[0]), int(parts[1]), int(parts[2])
            gpus.append({'physical_id': phys_id, 'free': free_mb * 1024 * 1024, 'total': total_mb * 1024 * 1024})

    if visible is not None:
        visible_set = set(visible)
        gpus = [g for g in gpus if g['physical_id'] in visible_set]

    for cuda_id, g in enumerate(gpus):
        g['cuda_id'] = cuda_id

    return gpus


def get_gpus():
    """Returns available GPU info list, respecting CUDA_VISIBLE_DEVICES.

    Each entry: {'cuda_id': int, 'physical_id': int, 'free': int (bytes), 'total': int (bytes)}

    Tries pynvml first, falls back to nvidia-smi. Returns [] if no GPUs available.
    """
    visible = _get_cuda_visible_devices()
    if visible is not None and len(visible) == 0:
        return []

    try:
        return _detect_gpus_pynvml(visible)
    except ImportError:
        pass

    return _detect_gpus_nvidia_smi(visible)


def get_idle_gpu(min_free_bytes=None):
    """Returns cuda_id of the GPU with the most free memory.

    Args:
        min_free_bytes: Minimum free memory required in bytes. None = no minimum.

    Returns:
        int or None: cuda_id of the selected GPU, or None if no GPU is available.
    """
    gpus = get_gpus()
    if not gpus:
        return None
    if min_free_bytes is not None:
        gpus = [g for g in gpus if g['free'] >= min_free_bytes]
    if not gpus:
        return None
    return max(gpus, key=lambda g: g['free'])['cuda_id']
