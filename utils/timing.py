# -*- coding: utf-8 -*-
"""高精度计时器（with 语法糖）。"""

from __future__ import annotations

import time
from typing import Optional


class Timer:
    """简易 ``with`` 计时器：``with Timer() as t: ...; t.elapsed``。"""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        assert self._start is not None
        self.elapsed = time.perf_counter() - self._start
