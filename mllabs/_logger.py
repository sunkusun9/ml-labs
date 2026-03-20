
import sys
import threading
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    def __init__(self, level=['info', 'warning', 'progress'], n_store_warning=1000):
        self._progress = list()
        self.show_info = 'info' in level
        self.show_warning = 'warning' in level
        self.show_progress = 'progress' in level
        self.warning_list = list()
        self.n_store_warning = n_store_warning

    @abstractmethod
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def start_progress(self, title, total=None):
        pass

    def update_progress(self, current):
        pass

    def end_progress(self, current=None):
        pass

    def adhoc_progress(self, current, total, msg=None):
        pass

    def rename_progress(self, title):
        pass

    def clear_progress(self):
        pass

    def create_session(self, position, session_cls=None, **session_kwargs):
        return None

    def remove_session(self, position):
        pass


class BaseProgressSession(BaseLogger):
    _ansi_managed = True

    def __init__(self, position, logger):
        self._position = position
        self._logger = logger
        self._progress = []
        self._adhoc = None
        self.show_info = logger.show_info
        self.show_warning = logger.show_warning
        self.show_progress = logger.show_progress
        self.warning_list = logger.warning_list
        self.n_store_warning = logger.n_store_warning

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def _format_line(self):
        parts = []
        for title, current, total in self._progress:
            if total is not None and total > 0:
                pct = int(current * 100 / total)
                parts.append(f"{title} {current}/{total} ({pct}%)")
            else:
                parts.append(f"{title} {current}")
        if self._adhoc is not None:
            current, total, msg = self._adhoc
            pct = int(current * 100 / total) if total > 0 else 0
            adhoc_str = f"{current}/{total} ({pct}%)"
            if msg:
                adhoc_str += f" {msg}"
            parts.append(adhoc_str)
        return ' > '.join(parts)

    def start_progress(self, title, total=None):
        with self._logger._lock:
            self._progress.append([title, 0, total])
            if self._logger.show_progress:
                self._logger._render_all_sessions()

    def update_progress(self, current):
        with self._logger._lock:
            if self._progress:
                self._progress[-1][1] = current
                if self._logger.show_progress:
                    self._logger._render_all_sessions()

    def end_progress(self, current=None):
        with self._logger._lock:
            if self._progress:
                if current is not None:
                    self._progress[-1][1] = current
                self._progress.pop()
                if self._logger.show_progress:
                    self._logger._render_all_sessions()

    def adhoc_progress(self, current, total, msg=None):
        with self._logger._lock:
            if not self._logger.show_progress or not self._progress:
                return
            self._adhoc = (current, total, msg)
            self._logger._render_all_sessions()
            self._adhoc = None

    def rename_progress(self, title):
        with self._logger._lock:
            if self._progress:
                self._progress[-1][0] = title
                if self._logger.show_progress:
                    self._logger._render_all_sessions()

    def clear_progress(self):
        with self._logger._lock:
            if self._progress:
                self._progress.clear()
                if self._logger.show_progress:
                    self._logger._render_all_sessions()

    def create_session(self, position, session_cls=None, **session_kwargs):
        return self._logger.create_session(position, session_cls, **session_kwargs)

    def remove_session(self, position):
        self._logger.remove_session(position)


class TqdmProgressSession(BaseProgressSession):
    _ansi_managed = False

    def __init__(self, position, logger):
        super().__init__(position, logger)
        from tqdm.auto import tqdm as _tqdm_cls
        self._tqdm_cls = _tqdm_cls
        self._bars = []

    def _format_line(self):
        return ''

    def start_progress(self, title, total=None):
        with self._logger._lock:
            bar = self._tqdm_cls(
                total=total, desc=title,
                position=self._position + len(self._bars),
                leave=False
            )
            self._bars.append(bar)

    def update_progress(self, current):
        with self._logger._lock:
            if self._bars:
                self._bars[-1].n = current
                self._bars[-1].refresh()

    def end_progress(self, current=None):
        with self._logger._lock:
            if self._bars:
                if current is not None:
                    self._bars[-1].n = current
                self._bars[-1].close()
                self._bars.pop()

    def adhoc_progress(self, current, total, msg=None):
        with self._logger._lock:
            if self._bars:
                bar = self._bars[-1]
                bar.n = current
                bar.total = total
                if msg:
                    bar.set_postfix_str(msg)
                bar.refresh()

    def rename_progress(self, title):
        with self._logger._lock:
            if self._bars:
                self._bars[-1].set_description(title)

    def clear_progress(self):
        with self._logger._lock:
            for bar in reversed(self._bars):
                bar.close()
            self._bars.clear()


class RichProgressSession(BaseProgressSession):
    _ansi_managed = False

    def __init__(self, position, logger, rich_progress):
        super().__init__(position, logger)
        self._rich_progress = rich_progress
        self._task_ids = []

    def _format_line(self):
        return ''

    def start_progress(self, title, total=None):
        with self._logger._lock:
            task_id = self._rich_progress.add_task(title, total=total)
            self._task_ids.append(task_id)

    def update_progress(self, current):
        with self._logger._lock:
            if self._task_ids:
                self._rich_progress.update(self._task_ids[-1], completed=current)

    def end_progress(self, current=None):
        with self._logger._lock:
            if self._task_ids:
                if current is not None:
                    self._rich_progress.update(self._task_ids[-1], completed=current)
                self._rich_progress.remove_task(self._task_ids.pop())

    def adhoc_progress(self, current, total, msg=None):
        with self._logger._lock:
            if self._task_ids:
                self._rich_progress.update(
                    self._task_ids[-1], completed=current, total=total,
                    description=msg or None
                )

    def rename_progress(self, title):
        with self._logger._lock:
            if self._task_ids:
                self._rich_progress.update(self._task_ids[-1], description=title)

    def clear_progress(self):
        with self._logger._lock:
            for task_id in reversed(self._task_ids):
                self._rich_progress.remove_task(task_id)
            self._task_ids.clear()


class DefaultLogger(BaseLogger):
    def __init__(self, level=['info', 'warning', 'progress'], n_store_warning=1000):
        super().__init__(level, n_store_warning)
        self._prev_progress_len = 0
        self._lock = threading.Lock()
        self._sessions = {}

    def _ansi_session_count(self):
        return sum(1 for s in self._sessions.values() if s._ansi_managed)

    def _clear_session_rows(self):
        n = self._ansi_session_count()
        if n == 0:
            return
        sys.stdout.write(f"\r\033[{n}A")
        for _ in range(n):
            sys.stdout.write("\033[2K\n")
        sys.stdout.write(f"\r\033[{n}A")
        sys.stdout.flush()

    def _render_all_sessions(self):
        for pos in sorted(self._sessions.keys()):
            session = self._sessions[pos]
            if session._ansi_managed:
                line = session._format_line()
                sys.stdout.write(f"\r\033[2K{line}\n")
        sys.stdout.flush()

    def _write(self, msg):
        for session in self._sessions.values():
            if isinstance(session, RichProgressSession):
                session._rich_progress.console.print(msg)
                return
            if isinstance(session, TqdmProgressSession):
                import tqdm as _tqdm
                _tqdm.tqdm.write(msg)
                return
        print(msg)

    def info(self, msg):
        if self.show_info:
            with self._lock:
                n = self._ansi_session_count()
                if n > 0:
                    self._clear_session_rows()
                self._write(msg)
                if n > 0:
                    self._render_all_sessions()

    def warning(self, msg):
        with self._lock:
            if self.show_warning:
                n = self._ansi_session_count()
                if n > 0:
                    self._clear_session_rows()
                self._write(msg)
                if n > 0:
                    self._render_all_sessions()
            self.warning_list.append(msg)
            self.warning_list = self.warning_list[-self.n_store_warning:]

    def _render_progress(self):
        parts = []
        for title, current, total in self._progress:
            if total is not None and total > 0:
                pct = int(current * 100 / total)
                parts.append(f"{title} {current}/{total} ({pct}%)")
            else:
                parts.append(f"{title} {current}")
        line = ' > '.join(parts)
        print(f"\r{line.ljust(self._prev_progress_len)}", end='', flush=True)
        self._prev_progress_len = len(line)

    def start_progress(self, title, total=None):
        self._progress.append([title, 0, total])
        if self.show_progress:
            self._render_progress()

    def update_progress(self, current):
        self._progress[-1][1] = current
        if self.show_progress:
            self._render_progress()

    def end_progress(self, current=None):
        if current is not None:
            self._progress[-1][1] = current
            if self.show_progress:
                self._render_progress()
        self._progress.pop()
        if len(self._progress) == 0:
            print()
            self._prev_progress_len = 0

    def adhoc_progress(self, current, total, msg=None):
        if not self.show_progress or len(self._progress) == 0:
            return
        parts = []
        for title, cur, tot in self._progress:
            if tot is not None and tot > 0:
                pct = int(cur * 100 / tot)
                parts.append(f"{title} {cur}/{tot} ({pct}%)")
            else:
                parts.append(f"{title} {cur}")
        pct = int(current * 100 / total) if total > 0 else 0
        adhoc_str = f"{current}/{total} ({pct}%)"
        if msg:
            adhoc_str += f" {msg}"
        parts.append(adhoc_str)
        line = ' > '.join(parts)
        print(f"\r{line.ljust(self._prev_progress_len)}", end='', flush=True)
        self._prev_progress_len = len(line)

    def rename_progress(self, title):
        if self._progress:
            self._progress[-1][0] = title
            if self.show_progress:
                self._render_progress()

    def clear_progress(self):
        if len(self._progress) > 0:
            self._progress.clear()
            print()
            self._prev_progress_len = 0

    def create_session(self, position, session_cls=None, **session_kwargs):
        if session_cls is None:
            session_cls = BaseProgressSession
        with self._lock:
            session = session_cls(position, self, **session_kwargs)
            self._sessions[position] = session
            if self.show_progress and session._ansi_managed:
                sys.stdout.write('\n')
                sys.stdout.flush()
        return session

    def remove_session(self, position):
        with self._lock:
            if position not in self._sessions:
                return
            was_ansi = self._sessions[position]._ansi_managed
            if was_ansi and self.show_progress:
                self._clear_session_rows()
            del self._sessions[position]
            if was_ansi and self._ansi_session_count() > 0 and self.show_progress:
                self._render_all_sessions()
