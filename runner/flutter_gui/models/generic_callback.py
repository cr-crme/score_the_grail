from typing import List, Callable


class GenericCallback:
    def __init__(self):
        self._callbacks: List[Callable] = []

    def listen(self, callback: Callable):
        self._callbacks.append(callback)

    def cancel(self, callback: Callable):
        self._callbacks.remove(callback)

    def notify(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)
