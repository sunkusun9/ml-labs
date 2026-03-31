class ExecuteTracker:
    def __init__(self, total, n_workers=1):
        self.total = total
        self.records = {}
        self.workers = {i: None for i in range(n_workers)}

    def start(self, worker_idx, node_name, outer_idx, inner_idx):
        self.workers[worker_idx] = {
            'job': (node_name, outer_idx, inner_idx),
            'progress': None,
        }
        self._on_update('start', worker_idx=worker_idx, node_name=node_name,
                        outer_idx=outer_idx, inner_idx=inner_idx)

    def progress(self, worker_idx, current, total, metrics=None):
        self.workers[worker_idx]['progress'] = (current, total, metrics)
        self._on_update('progress', worker_idx=worker_idx,
                        current=current, total=total, metrics=metrics)

    def done(self, worker_idx, node_name, outer_idx, inner_idx, info):
        self.workers[worker_idx] = None
        self.records[(node_name, outer_idx, inner_idx)] = {
            'status': 'done',
            'fit_time': info.get('fit_time'),
        }
        self._on_update('done', worker_idx=worker_idx, node_name=node_name,
                        outer_idx=outer_idx, inner_idx=inner_idx, info=info)

    def error(self, worker_idx, node_name, outer_idx, inner_idx, error_info):
        self.workers[worker_idx] = None
        self.records[(node_name, outer_idx, inner_idx)] = {
            'status': 'error',
            'error': error_info,
        }
        self._on_update('error', worker_idx=worker_idx, node_name=node_name,
                        outer_idx=outer_idx, inner_idx=inner_idx, error_info=error_info)

    def message(self, worker_idx, msg):
        self._on_update('message', worker_idx=worker_idx, msg=msg)

    def block(self, node_name, outer_idx, inner_idx):
        self.records[(node_name, outer_idx, inner_idx)] = {'status': 'blocked'}
        self._on_update('block', node_name=node_name,
                        outer_idx=outer_idx, inner_idx=inner_idx)

    def _on_update(self, event, **kwargs):
        pass

    def close(self):
        pass

    @property
    def n_done(self):
        return sum(1 for r in self.records.values() if r['status'] == 'done')

    @property
    def n_error(self):
        return sum(1 for r in self.records.values() if r['status'] == 'error')

    @property
    def n_blocked(self):
        return sum(1 for r in self.records.values() if r['status'] == 'blocked')

    def get_errors(self):
        return {k: r['error'] for k, r in self.records.items() if r['status'] == 'error'}

    def node_summary(self, node_name):
        counts = {'done': 0, 'error': 0, 'blocked': 0}
        for (n, *_), r in self.records.items():
            if n == node_name:
                counts[r['status']] += 1
        return counts

    def summary(self):
        return [
            {'node': k[0], 'outer_idx': k[1], 'inner_idx': k[2], **r}
            for k, r in self.records.items()
        ]


class LoggerExecuteTracker(ExecuteTracker):
    def __init__(self, total, n_workers, logger):
        super().__init__(total, n_workers)
        self.logger = logger
        self._overall = logger.create_session(0)
        self._overall.start_progress('tasks', total=total)
        self._worker_sessions = {
            i: logger.create_session(i + 1) for i in range(n_workers)
        }

    def _on_update(self, event, **kwargs):
        if event == 'message':
            self.logger.info(f"[worker {kwargs['worker_idx']}] {kwargs['msg']}")
            return

        if event == 'start':
            wi = kwargs['worker_idx']
            label = f"[{wi}] {kwargs['node_name']} {kwargs['outer_idx']}_{kwargs['inner_idx']}"
            session = self._worker_sessions[wi]
            session.clear_progress()
            session.start_progress(label)

        elif event == 'progress':
            wi = kwargs['worker_idx']
            self._worker_sessions[wi].adhoc_progress(
                kwargs['current'], kwargs['total'], kwargs.get('metrics')
            )

        elif event in ('done', 'error'):
            self._worker_sessions[kwargs['worker_idx']].clear_progress()
            self._overall.update_progress(self.n_done + self.n_error + self.n_blocked)

        elif event == 'block':
            self._overall.update_progress(self.n_done + self.n_error + self.n_blocked)

    def close(self):
        self._overall.end_progress()
        for session in self._worker_sessions.values():
            session.clear_progress()
        self.logger.remove_session(0)
        for i in self._worker_sessions:
            self.logger.remove_session(i + 1)
