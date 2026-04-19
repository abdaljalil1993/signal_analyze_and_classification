from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Hashable


class LRUFeatureCache:
    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> Any:
        with self._lock:
            if key not in self._data:
                return None
            value = self._data.pop(key)
            self._data[key] = value
            return value

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            self._data[key] = value
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)
