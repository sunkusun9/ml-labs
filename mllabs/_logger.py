
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    def __init__(self, level = ['info', 'warning', 'progress'], n_store_warning = 1000):
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

class DefaultLogger(BaseLogger):
    def __init__(self, level=['info', 'warning', 'progress'], n_store_warning=1000):
        super().__init__(level, n_store_warning)
        self._prev_progress_len = 0

    def info(self, msg):
        if self.show_info:
            print(msg)

    def warning(self, msg):
        if self.show_warning:
            print(msg)
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
