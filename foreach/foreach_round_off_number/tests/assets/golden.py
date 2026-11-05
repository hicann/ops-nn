#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""foreach_round_off_number 精度 Golden(比对标准 binary_equal, 两方逐位)。

对输入张量列表 (tensor list) 中每个张量的每个元素按 roundMode 指定的
舍入方式取整 (模式 7 取小数部), 输出列表与输入逐张量同 shape/dtype。
对标竞品经 third_party 声明: Kernel/aclnn 路为按用例 roundMode 动态分派
的自定义类 (1/2/3/5/7 分派对应 foreach API, 0/4/6 无对标竞品 API), E2E 路
五入口以字符串 API 形式各自声明, 由测试框架在隔离的第三方进程中解析执行。

roundMode 分派 (与算子内核 0~7 全模式一致):
  0/其他 = 恒等; 1 = rint (half-to-even); 2 = floor; 3 = ceil;
  4 = round (half-away-from-zero); 5 = trunc; 6 = 就近取整、恰为 .5 取奇;
  7 = frac (x - trunc(x), 负值小数部保号, -0.0 与整数子域按 IEEE x−x=+0.0,
  ±inf -> NaN)。
模式 1/2/3/5/7 的计算核按参考实现映射直接调用
torch._foreach_round/_foreach_floor/_foreach_ceil/_foreach_trunc/_foreach_frac
(float16/float32 列表级直调; bfloat16 走逐张量 f32 域路径, NaN 位型按
IEEE 位级语义保持输入 payload 并置静默位);
模式 4/6 无单一参考 API, 以复合表达式构造。

边界值示例 (float32, x = [-0.4, 0.5, 2.5, -2.5, -0.0, 3.5, -3.5, 100.7, inf, nan]):
  mode 0: [-0.4, 0.5, 2.5, -2.5, -0.0, 3.5, -3.5, 100.7, inf, nan] (恒等)
  mode 1: [-0.0, 0.0, 2.0, -2.0, -0.0, 4.0, -4.0, 101.0, inf, nan] (half-to-even; 负零保号)
  mode 2: [-1.0, 0.0, 2.0, -3.0, -0.0, 3.0, -4.0, 100.0, inf, nan] (floor 向 -inf)
  mode 3: [-0.0, 1.0, 3.0, -2.0, -0.0, 4.0, -3.0, 101.0, inf, nan] (ceil 向 +inf; 负零保号)
  mode 4: [-0.0, 1.0, 3.0, -3.0, -0.0, 4.0, -4.0, 101.0, inf, nan] (half-away-from-zero; 负零保号)
  mode 5: [-0.0, 0.0, 2.0, -2.0, -0.0, 3.0, -3.0, 100.0, inf, nan] (trunc 向零; 负零保号)
  mode 6: [0.0, 1.0, 3.0, -3.0, -0.0, 3.0, -3.0, 101.0, inf, nan] (就近取整、恰为 .5 取奇)
  mode 7: [-0.4, 0.5, 0.5, -0.5, 0.0, 0.5, -0.5, 0.69999695, nan, nan] (frac: 负值小数部保号, ±inf -> NaN, -0.0 为 x−x=+0.0)

通路:
  - Kernel  : list[np.ndarray]  -> list[np.ndarray]
  - ACLNN   : list[torch.Tensor] -> list[torch.Tensor] (可选 out 容器就地写)
  - E2E     : list[torch.Tensor] -> list[torch.Tensor]

roundMode 载体: Kernel/aclnn V1 为整型单元素张量 (int8), aclnn V2 为
标量 (int8|int64), 亦可按用例属性以标量到达 — 统一经 _round_mode_value
归一; 列表元素个数必须为 1。

整型张量 (int16): 模式 1~6 恒等 (逐位可比), 模式 7 特化输出 0 (参考
API 的整型 frac 路径不可用, 按特化声明承接)。

空 TensorList: 算子侧对空列表直接成功返回空列表, 故短路返回 [];
参考 foreach API 要求列表非空。

Import discipline: torch 只在函数体内 lazy import — 本文件会在可能已
初始化 aclrt 的进程内被加载, 模块级框架 import 有 SIGSEGV 风险;
numpy 可全局。

浮点槽到达契约 (仅 kernel/geir/aclnn 通路, 由测试框架在比对需要时统一
抬升一档: fp16/bf16 -> fp32, fp32 -> fp64): 到达什么精度就在该精度
计算 (纯函数语义); bf16 在 numpy 池恒以 f32 携带值到达, 计算后位级
转换由框架侧完成。E2E 通路不经抬升, 以原始 dtype 直达。

标量形态: 输入张量可为任意 rank (含 rank-0 标量); ascontiguousarray 会
把 0-d 升为 (1,), 转换链路以 reshape 落回原 shape。
"""

__spec__ = {
    "foreach_round_off_number": "ForeachRoundOffNumberKernelSpec",
    "ForeachRoundOffNumber": "ForeachRoundOffNumberKernelSpec",  # GE 算子类型名变体 (REG_OP(ForeachRoundOffNumber))
    "aclnnForeachRoundOffNumber": "AclnnForeachRoundOffNumberSpec",
    "aclnnForeachRoundOffNumberV2": "AclnnForeachRoundOffNumberSpec",  # V2 接口: roundMode 以 aclScalar (int8|int64) 承载, 同一计算核
    "torch._foreach_round": "TorchForeachRoundSpec",  # E2E 入口: roundMode=1 (rint, half-to-even)
    "torch._foreach_floor": "TorchForeachFloorSpec",  # E2E 入口: roundMode=2 (floor)
    "torch._foreach_ceil": "TorchForeachCeilSpec",  # E2E 入口: roundMode=3 (ceil)
    "torch._foreach_trunc": "TorchForeachTruncSpec",  # E2E 入口: roundMode=5 (trunc)
    "torch._foreach_frac": "TorchForeachFracSpec",  # E2E 入口: roundMode=7 (frac)
}

import numpy as np

# 各模式输出均为离散整数格点或 x−trunc(x) 的精确小数部, 在输出 dtype 下
# 精确可表示 (fp16/bf16 经 fp32 中间域计算后 RNE 回写无损) — 全 dtype
# 逐位比对, 无数值容差放行。
_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
}


_DEFAULT_ROUND_MODE = 1
_CAST_RINT = 1
_CAST_FLOOR = 2
_CAST_CEIL = 3
_CAST_ROUND = 4
_CAST_TRUNC = 5
_CAST_ODD = 6
_CAST_FRAC = 7

# roundMode -> 参考 foreach API; 模式 0/4/6 无单一参考 API, 以复合表达式构造。
_FOREACH_API_BY_MODE = {
    _CAST_RINT: "_foreach_round",
    _CAST_FLOOR: "_foreach_floor",
    _CAST_CEIL: "_foreach_ceil",
    _CAST_TRUNC: "_foreach_trunc",
    _CAST_FRAC: "_foreach_frac",
}


def _round_mode_value(round_mode):
    """Extract the scalar mode from the Tensor/scalar carriers of all routes."""
    import torch

    if round_mode is None:
        return _DEFAULT_ROUND_MODE
    if isinstance(round_mode, (list, tuple)):
        if len(round_mode) != 1:
            raise ValueError("roundMode must contain exactly one element")
        return _round_mode_value(round_mode[0])
    if isinstance(round_mode, torch.Tensor):
        if round_mode.numel() != 1:
            raise ValueError("roundMode must contain exactly one element")
        return int(round_mode.item())
    if isinstance(round_mode, np.ndarray):
        if round_mode.size != 1:
            raise ValueError("roundMode must contain exactly one element")
        return int(round_mode.reshape(-1)[0].item())
    return int(round_mode)


def _round_to_odd(value):
    import torch

    floor_value = torch.floor(value)
    fraction = value - floor_value
    odd_value = torch.where(
        torch.remainder(floor_value, 2.0) != 0.0,
        floor_value,
        floor_value + 1.0,
    )
    return torch.where(
        fraction > 0.5,
        floor_value + 1.0,
        torch.where(fraction < 0.5, floor_value, odd_value),
    )


def _apply_round_mode(value, round_mode):
    import torch

    # Integer tensors: only FRAC changes the value (the fractional part of
    # an integer is always zero); mirrors the kernel's integer specialization.
    if not value.is_floating_point():
        return torch.zeros_like(value) if round_mode == _CAST_FRAC else value.clone()

    if value.dtype in (torch.float16, torch.bfloat16):
        # IEEE 单次舍入语义: 复合表达式的中间量按无限精度计算、仅最终结果
        # 舍入回原生 dtype; 原生 dtype 逐算子舍入会引入多重舍入偏差 (如
        # |x|+0.5 的 ulp 区间进位)。宽域计算结果均有精确表示, 回写逐位无损。
        # 该语义经独立位级仿真交叉验证 (numpy 整数位域, 与本实现逐位一致)。
        result = _apply_round_mode(value.to(torch.float32), round_mode).to(value.dtype)
        # 半精度提升路径的参考实现会把 NaN 规格化为统一位型 (尾数归一);
        # 按 IEEE 位级语义, NaN 输入的输出位型保持输入 payload 并置静默位 —
        # 本实现按该语义恢复位型 (独立仿真同口径验证)。
        nan_mask = torch.isnan(value)
        if bool(nan_mask.any()):
            quiet = 0x0200 if value.dtype == torch.float16 else 0x0040
            kept = (value.view(torch.int16) | quiet).view(value.dtype)
            result = torch.where(nan_mask, kept, result)
        return result

    if round_mode == _CAST_RINT:
        return torch.round(value)
    if round_mode == _CAST_FLOOR:
        return torch.floor(value)
    if round_mode == _CAST_CEIL:
        return torch.ceil(value)
    if round_mode == _CAST_ROUND:
        # 内核为 roundf (half-away-from-zero): |x|+0.5 复合式的中转加法在
        # f32 下自身会舍入 (|x| >= 2^23 的奇数格点 +0.5 恰为 tie 进位、
        # 0.5 下侧邻值 +0.5 落入偶数侧进位), 偏离 roundf 的精确取整 —
        # 改为 trunc + 精确小数部判定, 全程无中转舍入。
        trunc_value = torch.trunc(value)
        fraction = value - trunc_value
        away = torch.where(
            fraction >= 0.5, 1.0, torch.where(fraction <= -0.5, -1.0, 0.0)
        )
        rounded = trunc_value + away
        return torch.where(rounded == 0, torch.copysign(rounded, value), rounded)
    if round_mode == _CAST_TRUNC:
        return torch.trunc(value)
    if round_mode == _CAST_ODD:
        return _round_to_odd(value)
    if round_mode == _CAST_FRAC:
        return torch.frac(value)
    # CAST_NONE and unsupported values fall back to identity, matching the
    # kernel's default branch.
    return value.clone()


def _frac_inf(value, out):
    """frac(+-inf) = inf - inf 的无效运算结果为规格化 NaN (参考平台位型
    0xFFC00000)。f32/fp16 参考实现自然给出该位型; bf16 的低精度回写会把
    NaN 改写为全一尾数形态, 按规格化位型的截断像 (0xFFC0) 恢复。"""
    import torch

    if value.dtype != torch.bfloat16:
        return out
    inf_mask = torch.isinf(value)
    if not bool(inf_mask.any()):
        return out
    # 0xFFC0 = 0xFFC00000 截断到 bf16 的规格化 NaN 位型 (int16 有符号值 -64)
    canonical = (torch.zeros_like(out.view(torch.int16)) - 64).view(out.dtype)
    return torch.where(inf_mask, canonical, out)


def _foreach_round_off_number(x, roundMode=_DEFAULT_ROUND_MODE):
    import torch

    if not x:
        # 空 TensorList: 算子侧直接成功返回空列表 (参考 foreach API 要求非空)。
        return []
    round_mode = _round_mode_value(roundMode)
    api_name = _FOREACH_API_BY_MODE.get(round_mode)
    # 直调分支的位型依据与不可达域说明:
    # - NaN/±inf 位型 (含 mode7 的 ±inf -> NaN): 直调参考 API 对 fp16/f32 的
    #   NaN 逐元素保真搬运 (不规格化 payload), 与内核 __half2float/
    #   __float2half_rn 链位级一致为实测结论 — 板级全量执行 (609 例, 含
    #   NaN/±inf 行) binary_equal 全 PASS, GPU 侧 foreach 参考实现同口径;
    #   比较器对双真 NaN 取 "双真即等" 特判 (两侧均为 NaN 即视为相等),
    #   位型分歧不构成误报面, torch 升级改变保真行为时表现为可见 FAIL
    #   (方向安全)。
    # - ST 实跑矩阵的 inf×mode7: f32(+inf)/bf16(−inf) 有行, fp16 无行 —
    #   fp16 口径由 golden 自检的 "模式7 ±inf -> NaN" 位级 invariant anchor
    #   承载。
    # - 混合 dtype (fp16+fp32) 列表直调会被 torch promote 到 f32, fp16 成员
    #   输出 dtype 漂移 — 该输入算子侧不可达 (aclnn 入口逐成员 dtype 一致
    #   校验与 tiling 列表 dtype 一致性守卫双重拦截), 不为不可达域补防御。
    if api_name is not None and all(
        t.dtype in (torch.float16, torch.float32) for t in x
    ):
        result = list(getattr(torch, api_name)(x))
    else:
        result = [_apply_round_mode(value, round_mode) for value in x]
    if round_mode == _CAST_FRAC:
        result = [_frac_inf(value, out) for value, out in zip(x, result)]
    return result


class _ForeachRoundOffNumberTorchRef:
    """torch 侧第三方参考 (按用例 roundMode 动态分派对应 foreach API)。

    third_party 类形态: 测试框架 server 端实例化并调用, __init__/__call__
    按名各取所声明的参数 (roundMode 载体经 _round_mode_value 归一后定分派,
    x 为张量列表), 其余池项经 **kwargs 到达不消费。roundMode ∈ {1, 2, 3, 5, 7}
    分派 _foreach_round/_foreach_floor/_foreach_ceil/_foreach_trunc/_foreach_frac;
    0/4/6 无对标竞品 API, 如实不可用 (NotImplementedError)。
    """

    def __init__(self, roundMode=None, **kwargs):
        mode = _round_mode_value(roundMode)
        api = _FOREACH_API_BY_MODE.get(mode)
        if api is None:
            raise NotImplementedError(f"roundMode={mode} has no torch counterpart API")
        self._api_name = api

    def __call__(self, x, **kwargs):
        import torch

        return list(getattr(torch, self._api_name)(x))


class ForeachRoundOffNumberKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。

    roundMode 与算子内核的 0~7 分派及非法值回退语义一致。

    到达/回程转换: 原生 dtype 走 from_numpy/.numpy(); 扩展 dtype 数组
    (bfloat16 等不受 torch.from_numpy 支持) 以 uint16 视图位级搬运,
    值流与位型均无损, 回程还原为到达时的数组 dtype。
    """

    def golden(x, roundMode=_DEFAULT_ROUND_MODE, **kwargs):
        import torch

        x_list = []
        arrival_dtypes = []
        for t in x:
            arr = np.ascontiguousarray(t)
            try:
                tensor = torch.from_numpy(arr)
            except TypeError:
                tensor = torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
            x_list.append(tensor.reshape(t.shape))
            arrival_dtypes.append(arr.dtype)
        result = _foreach_round_off_number(x_list, roundMode)
        out = []
        for value, arrival in zip(result, arrival_dtypes):
            if value.dtype == torch.bfloat16:
                out.append(value.view(torch.uint16).numpy().view(arrival))
            else:
                out.append(value.numpy())
        return out

    third_party = {"torch": _ForeachRoundOffNumberTorchRef}
    tolerance = _TOLERANCE


class AclnnForeachRoundOffNumberSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: aclnnForeachRoundOffNumberGetWorkspaceSize(x, roundMode, out, workspaceSize, executor);
    V2 以 aclScalar (int8|int64) 承载 roundMode, 经同一分派。
    """

    def golden(x, roundMode=_DEFAULT_ROUND_MODE, out=None, **kwargs):
        # out 容器不写入: 参考结果直接返回, 保持 "kernel 漏写输出" 类缺陷
        # 的可检出性 (设备输出缓冲不得被参考值预热)。
        return list(_foreach_round_off_number(x, roundMode))

    third_party = {"torch": _ForeachRoundOffNumberTorchRef}
    tolerance = _TOLERANCE


class TorchForeachRoundSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    API_NAME (_foreach_round) 语义即 roundMode=1 (rint, half-to-even); 用例的
    decimals 属性经 **kwargs 接收消费, 不向参考 API 转发 (该 API 无此参数)。
    """

    API_NAME = "_foreach_round"

    def golden(x, roundMode=_DEFAULT_ROUND_MODE, **kwargs):
        import torch

        return list(getattr(torch, TorchForeachRoundSpec.API_NAME)(x))

    third_party = {"torch": "torch." + API_NAME}
    tolerance = _TOLERANCE


# 历史注册类名, 保留为 round 入口的别名 (既有引用方零影响)
TorchForeachRoundOffNumberSpec = TorchForeachRoundSpec


class TorchForeachFloorSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    API_NAME (_foreach_floor) 语义即 roundMode=2 (floor, 向 -inf 取整); 用例属性
    经 **kwargs 接收消费, 不向参考 API 转发 (该 API 无此参数)。
    """

    API_NAME = "_foreach_floor"

    def golden(x, roundMode=_DEFAULT_ROUND_MODE, **kwargs):
        import torch

        return list(getattr(torch, TorchForeachFloorSpec.API_NAME)(x))

    third_party = {"torch": "torch." + API_NAME}
    tolerance = _TOLERANCE


class TorchForeachCeilSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    API_NAME (_foreach_ceil) 语义即 roundMode=3 (ceil, 向 +inf 取整); 用例属性
    经 **kwargs 接收消费, 不向参考 API 转发 (该 API 无此参数)。
    """

    API_NAME = "_foreach_ceil"

    def golden(x, roundMode=_DEFAULT_ROUND_MODE, **kwargs):
        import torch

        return list(getattr(torch, TorchForeachCeilSpec.API_NAME)(x))

    third_party = {"torch": "torch." + API_NAME}
    tolerance = _TOLERANCE


class TorchForeachTruncSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    API_NAME (_foreach_trunc) 语义即 roundMode=5 (trunc, 向零取整); 用例属性
    经 **kwargs 接收消费, 不向参考 API 转发 (该 API 无此参数)。
    """

    API_NAME = "_foreach_trunc"

    def golden(x, roundMode=_DEFAULT_ROUND_MODE, **kwargs):
        import torch

        return list(getattr(torch, TorchForeachTruncSpec.API_NAME)(x))

    third_party = {"torch": "torch." + API_NAME}
    tolerance = _TOLERANCE


class TorchForeachFracSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    API_NAME (_foreach_frac) 语义即 roundMode=7 (frac, x − trunc(x)): 负值小数部
    保号, ±inf -> NaN (inf−inf 的 IEEE 无效运算位型; 与内核 VSUB 无效运算位型
    的一致性为实测结论 — 板级全量执行 binary_equal 全 PASS, 且比较器对双真
    NaN "双真即等", 位型分歧不构成误报面), -0.0 与整数子域按 IEEE x−x=+0.0;
    用例属性经 **kwargs 接收消费, 不向参考 API 转发。
    """

    API_NAME = "_foreach_frac"

    def golden(x, roundMode=_DEFAULT_ROUND_MODE, **kwargs):
        import torch

        return list(getattr(torch, TorchForeachFracSpec.API_NAME)(x))

    third_party = {"torch": "torch." + API_NAME}
    tolerance = _TOLERANCE
