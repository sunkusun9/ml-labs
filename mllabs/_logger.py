
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

    def start_progress(self, session_id, title, total=None):
        pass

    def update_progress(self, session_id, current):
        pass

    def end_progress(self, session_id, current=None):
        pass

    def adhoc_progress(self, session_id, current, total, msg=None):
        pass

    def rename_progress(self, session_id, title):
        pass

    def clear_progress(self, session_id):
        pass

    def create_session(self, position, session_cls=None, **session_kwargs):
        return None

    def remove_session(self, position):
        pass

class BaseProgressSession:
    """Progress display for one position. Standalone — no Logger inheritance."""
    _ansi_managed = True
    def __init__(self, position):
        self._position = position
    @property
    def is_empty(self):
        return True
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
    def clear(self):
        pass

class DefaultProgressSession(BaseProgressSession):
    _ansi_managed = True

    def __init__(self, position):
        super().__init__(position)
        self._progress = []
        self._adhoc = None


class TqdmProgressSession(BaseProgressSession):
    _ansi_managed = False

    def __init__(self, position):
        super().__init__(position)
        from tqdm.auto import tqdm as _tqdm_cls
        self._tqdm_cls = _tqdm_cls
        self._bar= self._tqdm_cls(position=self._position, leave=False, total=100)

    def _format_line(self):
        return ''

    def start_progress(self, title, total=100):
        self._bar.n = 0
        self._bar.set_description(title)
        self._bar.reset(total)
        self._bar.refresh()

    def update_progress(self, current):
        self._bar.n = current
        self._bar.refresh()

    def end_progress(self, current=None):
        if current is not None:
            self._bar.n = current
        else:
            self._bar.n = self._bar.total
        self._bar.refresh()

    def adhoc_progress(self, current, total, msg=None):
        if self._bar.total != total:
            self._bar.reset(total)
        self._bar.n = current
        if msg is not None:
            self._bar.set_description(msg)
        self._bar.refresh()
    
    def rename_progress(self, title):
        self._bar.set_description(title)

    def clear(self):
        self._bar.clear()
        self._bar.close()

class DefaultLogger(BaseLogger):
    def __init__(self, level=['info', 'warning', 'progress'], n_store_warning=1000, session_cls = None):
        super().__init__(level, n_store_warning)
        if session_cls is None:
            self.session_cls = BaseProgressSession
        else:
            self.session_cls = session_cls
        self._sessions = {}


    def _write(self, msg):
        print(msg)

    def info(self, msg):
        if self.show_info:
            self._write(msg)

    def warning(self, msg):
        if self.show_warning:
            self._write(msg)
        self.warning_list.append(msg)
        self.warning_list = self.warning_list[-self.n_store_warning:]


    def start_progress(self, session_id, title, total=100):
        if session_id in self._sessions:
            self._sessions[session_id].start_progress(title, total)

    def update_progress(self, session_id, current):
        if session_id in self._sessions:
            self._sessions[session_id].update_progress(current)

    def end_progress(self, session_id, current=None):
        if session_id in self._sessions:
            self._sessions[session_id].end_progress(current)

    def adhoc_progress(self, session_id, current, total, msg=None):
        if session_id in self._sessions:
            self._sessions[session_id].adhoc_progress(current, total, msg)

    def rename_progress(self, session_id, title):
        if session_id in self._sessions:
            self._sessions[session_id].rename_progress(title)

    def clear_progress(self, session_id):
        if session_id in self._sessions:
            self._sessions[session_id].update_progress(0)

    def create_session(self, session_id, **session_kwargs):
        if session_id in self._sessions:
            return self._sessions[session_id]
        session = self.session_cls(session_id, **session_kwargs)
        self._sessions[session_id] = session
        return session

    def remove_session(self, session_id):
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
