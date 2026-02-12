
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

    def clear_progress(self):
        pass

class DefaultLogger(BaseLogger):
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
        print(f"\r{' > '.join(parts)}", end='', flush=True)

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
        print(f"\r{' > '.join(parts)}", end='', flush=True)

    def clear_progress(self):
        if len(self._progress) > 0:
            self._progress.clear()
            print()
