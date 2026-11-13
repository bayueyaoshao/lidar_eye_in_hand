"""把参考工程 ladar_in_head 注入 sys.path(只读复用,严禁写入)。

参考工程提供算法包 calib/(io/pcd、registration、handeye、geometry 等)。
本工程自身模块统一放 hecal/ 下,避免与参考工程的 calib 包名冲突。

任何 hecal 模块 import 参考工程的 calib.* 之前,必须先 import 本模块。
"""
from __future__ import annotations

import sys
from pathlib import Path

REF_PROJECT_ROOT = Path(
    "/home/lx/Works/src/bolight_robot/RadarDAQ/code/ladar_in_head"
)

if not REF_PROJECT_ROOT.is_dir():
    raise RuntimeError(
        f"参考工程目录不存在: {REF_PROJECT_ROOT}\n"
        "本管线只读复用其算法模块(calib 包),请确认路径后重试。"
    )

_REF_STR = str(REF_PROJECT_ROOT)
if _REF_STR not in sys.path:
    sys.path.insert(0, _REF_STR)
