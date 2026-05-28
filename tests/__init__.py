# -*- coding: utf-8 -*-
"""单元测试包入口 — 自动注入项目 root 到 sys.path。"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
