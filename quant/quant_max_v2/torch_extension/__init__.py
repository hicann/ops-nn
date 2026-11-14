from .quant_max_v2 import quant_max

# 在算子 schema 注册之后，加载随扩展发布的 GE 映射（torchair 不可用时模块内部自行降级）。
from . import graph_convert_quant_max_v2 as graph_convert_quant_max_v2

# Distribution entry-point compatibility (setup.py derives the attribute name
# from the operator directory name).
quant_max_v2 = quant_max
