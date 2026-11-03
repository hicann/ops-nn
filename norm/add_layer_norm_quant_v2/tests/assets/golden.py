#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""AddLayerNormQuantV2 多通路 golden（TestSpec 范式）。

通路支持面（见 01_requirement.md §3.3）：
  kernel ✅ / geir ❌（未注册 V2 infershape，图模式不可编译）/ aclnn ✅ /
  e2e ❌（torch_npu 二进制未引用 aclnnAddLayerNormQuantV2，对照 aclnnAddRmsNormQuant 有命中）。
"""

__spec__ = {
    # kernel（GEIR 复用同一 op 蛇形名，但本算子 GE 通路不适用）
    "add_layer_norm_quant_v2": "AddLayerNormQuantV2KernelSpec",
    # aclnn
    "aclnnAddLayerNormQuantV2": "AddLayerNormQuantV2AclnnSpec",
}

import numpy as np
import torch

# kernel 通路：三方 cross_check（有 GPU 时生效）
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "quant"},  # 量化输出判据：绝对误差 <= 1 LSB
}

# aclnn 通路：TTK 当前不取用 third_party，三方一律免检 → 本机判。
# 浮点用 stat_rel_err（按 dtype 阈值），int8 显式 quant（|out-golden| <= 1）。
# 不用 cross_check：它会连远端 party，party down 时浮点输出全 GOLDEN_FAILURE 假失败。
_TOL_ACLNN = {
    # 本机无 GPU，aclnn 通路三方免检 → 本机判。浮点用 mix_tolerance（|a-b| <= atol + rtol*|b|，
    # 99% 元素满足即过）：纯 stat_rel_err 的 mare 在近零元素上会被 ULP 噪声放大
    # （实测 9 例 fp16/fp32 单元素 |diff|~1e-8 而 mare 0.02 > 10*th → 假失败），
    # 该口径与 verification §5 的「联合绝对+相对」真 bug 判据一致。int8 用 quant（|diff| <= 1）。
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
    "int8": {"standard": "quant"},
}


def _attr(kwargs, name, default, *aliases):
    for n in (name,) + aliases:
        if n in kwargs:
            v0 = kwargs[n]
            if isinstance(v0, (list, tuple)) and v0:
                v0 = v0[0]
            if isinstance(v0, str):
                s = v0.strip().strip('"').strip("'").lower()
                if s in ("true", "false", "yes", "no", "1", "0"):
                    return s in ("true", "yes", "1")
                try:
                    return type(default)(s)
                except Exception:
                    return v0.strip().strip('"').strip("'")
            return v0
    return default


def _as_tensor(a):
    """numpy/torch -> torch.Tensor；缺失可选输入(None/object 占位) -> None；fp16/bf16 抬到 fp32。

    抬精度仅向上兜底：内核 VF 全程 fp32、出口才窄回；golden 若留在 fp16/bf16 逐步 round
    会比内核糙，边缘元素差几 ULP → 判据在容差边缘随机翻红。
    """
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        return a.to(torch.float32) if a.dtype in (torch.bfloat16, torch.float16) else a
    arr = np.asarray(a)
    if arr.dtype == object:
        flat = [x for x in arr.reshape(-1).tolist() if x is not None]
        if not flat:
            return None
        arr = np.asarray(flat[0])
    if str(arr.dtype) in ("bfloat16", "float16"):
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _empty_outputs(x, gamma, quant_mode, has_scales2):
    """空 tensor 语义（A5 口径，与 V1 对齐）：y1/y2/x/layernorm_res 为空；
    dynamic 且归一化轴为 0（cols==0 && rows!=0）时，走 empty-input kernel，out_scales 填 -inf。

    aclnn 判据：dynamic 且 outScales1 为空 → no-op；static 且 x1 为空 → no-op；
    其余空 tensor 继续下发，由 empty kernel（key=9）写 dynamic 的 out_scales。
    注意 A2/A3 无 empty kernel，一律 no-op，out_scales 内容未定义。
    """
    dev = x.device
    x_shape = tuple(x.shape)
    base_dtype = x.dtype
    g_rank = gamma.dim() if gamma is not None else 1
    prefix = tuple(x_shape[: len(x_shape) - g_rank]) if len(x_shape) > g_rank else ()
    y1 = torch.zeros(x_shape, dtype=torch.int8, device=dev)
    y2 = (
        torch.zeros(x_shape, dtype=torch.int8, device=dev)
        if has_scales2
        else torch.zeros((1,), dtype=torch.int8, device=dev)
    )
    xo = torch.zeros(x_shape, dtype=base_dtype, device=dev)
    ln = torch.zeros(x_shape, dtype=base_dtype, device=dev)
    if quant_mode == "dynamic":
        os_shape = prefix if prefix else (1,)
        os1 = torch.full(os_shape, float("-inf"), dtype=torch.float32, device=dev)
        os2 = (
            os1.clone()
            if has_scales2
            else torch.zeros((1,), dtype=torch.float32, device=dev)
        )
    else:
        os1 = torch.zeros((1,), dtype=torch.float32, device=dev)
        os2 = torch.zeros((1,), dtype=torch.float32, device=dev)
    return [y1, y2, xo, ln, os1, os2]


def _compute(*tensors, **kwargs):
    """x1,x2,gamma,beta,bias?,scales1?,scales2?,zero_points1?,zero_points2?
    -> y1,y2,x,layernorm_res,out_scales1,out_scales2（顺序照 def.cpp）。"""
    t = list(tensors) + [None] * 9
    x1, x2, gamma, beta, bias, scales1, scales2, zp1, zp2 = t[:9]

    quant_mode = (
        str(_attr(kwargs, "quant_mode", "dynamic", "quantMode"))
        .strip()
        .strip('"')
        .strip("'")
    )
    epsilon = float(_attr(kwargs, "epsilon", 1e-5))
    div_mode = bool(_attr(kwargs, "div_mode", True, "divMode"))

    x = x1 + x2 if bias is None else x1 + x2 + bias
    if x.numel() == 0 or gamma.numel() == 0:
        return _empty_outputs(x, gamma, quant_mode, scales2 is not None)

    row = int(gamma.reshape(-1).numel())
    xf = x.reshape(-1, row)
    x_mean = xf.mean(dim=-1, keepdim=True)
    x_var = xf.var(dim=-1, unbiased=False, keepdim=True)
    g = gamma.reshape(1, -1).to(xf.dtype)
    b = beta.reshape(1, -1).to(xf.dtype)
    y = (xf - x_mean) / torch.sqrt(x_var + epsilon) * g + b
    y = y.reshape(x.shape)

    layernorm_res = y

    def _round_i8(v):
        return torch.clamp(torch.round(v), -128, 127).to(torch.int8)

    dev = x1.device
    os_shape = tuple(x.shape[: -gamma.dim()]) if gamma.dim() < x.dim() else ()

    if quant_mode == "static":
        # 静态：per-channel（scales==gamma）或 per-tensor（scales 单元素）
        s1 = scales1.reshape(-1)[0] if scales1.numel() == 1 else scales1
        y1 = (
            _round_i8(y / s1 + (0.0 if zp1 is None else zp1))
            if div_mode
            else _round_i8(y * s1 + (0.0 if zp1 is None else zp1))
        )
        if scales2 is not None:
            s2 = scales2.reshape(-1)[0] if scales2.numel() == 1 else scales2
            y2 = (
                _round_i8(y / s2 + (0.0 if zp2 is None else zp2))
                if div_mode
                else _round_i8(y * s2 + (0.0 if zp2 is None else zp2))
            )
        else:
            y2 = torch.zeros((1,), dtype=torch.int8, device=dev)
        # static 下 out_scales 无实际意义（内核不写），返回占位 1
        out_scale1 = torch.ones((1,), dtype=torch.float32, device=dev)
        out_scale2 = torch.ones((1,), dtype=torch.float32, device=dev)
    else:
        yf = y.reshape(-1, row)

        def _dyn(sc):
            if sc is None:
                tmp = yf
            elif sc.numel() == 1:
                tmp = yf * sc.reshape(-1)[0].to(yf.dtype)
            else:
                # scales shape == gamma（可为多维，如 [4, 64]）；yf 已展平成 [rows, prod(gamma)]，
                # 必须同步展平，否则多维 gamma 会按尾部维广播失败（实测 2x2x2x2x2x2x4x64）。
                tmp = yf * sc.reshape(-1).to(yf.dtype)
            m = torch.abs(tmp).amax(dim=-1)
            os_ = (m / 127.0).to(torch.float32)
            # 全零行（row_max=0）在核心里是 0/0；这里置 0 避免 NaN 污染，正常数据不会命中
            denom = os_.reshape(-1, 1).clone()
            denom[denom == 0] = 1.0
            out = _round_i8(tmp / denom).reshape(x.shape)
            return out, (os_.reshape(os_shape) if os_shape else os_)

        y1, out_scale1 = _dyn(scales1)
        if scales2 is not None:
            y2, out_scale2 = _dyn(scales2)
        else:
            y2 = torch.zeros((1,), dtype=torch.int8, device=dev)
            out_scale2 = torch.zeros((1,), dtype=torch.float32, device=dev)

    return [y1, y2, x, layernorm_res, out_scale1, out_scale2]


_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _od(kwargs):
    od = kwargs.get("output_dtypes") or []
    return [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]


def _cast_out(outs, od):
    """KernelSpec 出口：按【下发】的 output_dtypes 窄回，返 numpy。"""
    res = []
    for i, o in enumerate(outs):
        arr = o.detach().cpu().numpy()
        if i < len(od) and od[i]:
            arr = arr.astype(od[i])
        res.append(arr)
    return res


def _cast_out_tensor(outs, od):
    """AclnnSpec 出口：按【下发】的 output_dtypes 窄回，返 torch.Tensor（模板要求返 Tensor）。

    Promote 下 od 已是抬高档（fp16→fp32、fp32→fp64）→ 本步退化为 no-op，等价规范的「零 cast」；
    两方（无 Promote）下 od 是算子声明 dtype → 与 NPU 出口窄回行为一致。
    """
    res = []
    for i, o in enumerate(outs):
        if not isinstance(o, torch.Tensor):
            o = torch.as_tensor(o)
        if i < len(od) and od[i] and od[i] in _TORCH_DTYPE:
            o = o.to(_TORCH_DTYPE[od[i]])
        res.append(o)
    return res


class _AddLayerNormQuantV2Compose:
    """三方竞品标杆：与 arch35 kernel **同算法**的 torch 拼接（远端 GPU server 执行）。

    铁律：本类**不复用** `_compute`。`_compute` 是数学最优写法（mean/var 库实现 + `1/sqrt`），
    只作 CPU 真值；本类逐条对齐 kernel 的运算序列，只作 cross_check 的竞品标杆。

    与 kernel 的对齐点（证据见 GOLDEN_REF.md §3）：
      · Add 结合序：`(x1 + x2) + bias`。
      · LayerNorm 统计量：mean-centered 两遍归约，**逐元素先除以 cols 再 ReduceSum**
        （VFCalcMeanVarFast/VFCalcMeanVar: Div→ReduceSum）；全程 fp32。
      · rstd：`torch.rsqrt(var + eps)`，对齐 kernel 的 Newton-Raphson rsqrt（NR 结果即 rsqrt 近似）。
      · y：`(x - mean) * rstd * gamma + beta`（Sub→Mul(rstd)→Mul(gamma)→Add(beta)）。
      · static：`round(y/sc + zp)` / `round(y*sc + zp)`，RINT 取整 + clamp[-128,127]。
      · dynamic：`tmp=y*sc → rowmax(|tmp|) → os = m * (1/127) → round(tmp/os)`
        （kernel `Muls(outScale, outScale, 1/127f)`，不是 m/127）。
      · **cast 矩阵（按 golden_standard.md §一/§四：GPU 腿按 NPU 加宽行为同步 cast、出口对应窄回）**：
          入口：fp16/bf16 → fp32（kernel VF 全程 fp32）；fp32 不升档（kernel 累加也是 fp32）。
          出口：x/layernorm_res → x1.dtype（NPU 窄回）；out_scales → fp32；
                **y1/y2 → int8**（torch 整型中间量会升 int32，必须显式 .to(torch.int8)，否则漏成 int32）。
        int8 出参用 clamp[-128,127]+RINT 对齐内核饱和 Cast（±1 LSB 判据）。
        本类不做整型除法，故不涉及 torch INT_MIN/-1 的 SIGFPE 陷阱。

    已知差异（无法用 torch 完全复刻，选择依据=不高于 kernel 精度）：
      · 大列（welford 通路）内核用并行 Welford + 二分合并；本类统一用 mean-centered 两遍归约，
        归约合并次序不同，误差同为 fp32 量级、不优于 Welford，不会把竞品凭空变准。
      · kernel 的 NR 含 2 次修正，`torch.rsqrt` 与之差 ≤1 ULP。
      · kernel int8 前多一次 fp32→fp16 舍入；本类直接 round→int8（±1 LSB 判据覆盖）。
    """

    def __init__(
        self,
        quant_mode="dynamic",
        epsilon=1e-5,
        additional_output=False,
        div_mode=True,
        **kwargs,
    ):
        def _one(v):
            if isinstance(v, (list, tuple)) and v:
                return v[0]
            return v

        qm = _one(quant_mode)
        self.quant_mode = (
            str(qm).strip().strip('"').strip("'") if qm is not None else "dynamic"
        )
        self.epsilon = float(_one(epsilon))
        ao = _one(additional_output)
        self.additional_output = (
            ao if isinstance(ao, bool) else str(ao).lower() in ("true", "1", "yes")
        )
        dm = _one(div_mode)
        self.div_mode = (
            dm if isinstance(dm, bool) else str(dm).lower() in ("true", "1", "yes")
        )

    def __call__(
        self,
        x1,
        x2,
        gamma,
        beta,
        bias=None,
        scales1=None,
        scales2=None,
        zero_points1=None,
        zero_points2=None,
        input_dtypes=None,
        output_dtypes=None,
        **kwargs,
    ):
        # 声明 dtype 窄回目标：三方腿收到的是 **Promote 后**的 fp32 输入（TTK 现状），
        # 若只按 x1.dtype 窄回会漏掉算子声明的低精度输出（bf16/fp16）。若框架下发了
        # input_dtypes/output_dtypes（逻辑 dtype），优先用它；否则退回 x1.dtype。
        dt_tab = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        declared = None
        for cand in (output_dtypes, input_dtypes):
            if cand:
                c0 = cand[0] if isinstance(cand, (list, tuple)) else cand
                if isinstance(c0, str) and c0 in dt_tab:
                    declared = dt_tab[c0]
                    break
        # ⚠ 出口 dtype 必须在入口加宽**之前**取：x1.dtype 是 NPU 的输出 dtype（bf16/fp16/fp32）。
        # 若先 map(f32)，x1 已被改成 fp32，出口就漏窄回（GPU 停在 fp32，与 NPU 低精度输出不同精度 →
        # cross_check 的 rmse 比值被放大成假红）。
        out_dtype = declared or x1.dtype

        def f32(t):
            if t is None:
                return None
            return (
                t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t
            )

        x1, x2, gamma, beta = map(f32, (x1, x2, gamma, beta))
        bias, s1, s2, z1, z2 = map(
            f32, (bias, scales1, scales2, zero_points1, zero_points2)
        )

        x = (x1 + x2) if bias is None else (x1 + x2 + bias)

        n = int(gamma.numel())
        prefix = tuple(x.shape[: x.shape.__len__() - gamma.dim()])
        if x.numel() == 0:
            y1 = torch.zeros(x.shape, dtype=torch.int8, device=x.device)
            y2 = (
                torch.zeros(x.shape, dtype=torch.int8, device=x.device)
                if s2 is not None
                else torch.zeros((1,), dtype=torch.int8, device=x.device)
            )
            xo = torch.zeros(x.shape, dtype=out_dtype, device=x.device)
            ln = torch.zeros(x.shape, dtype=out_dtype, device=x.device)
            if self.quant_mode == "static":
                os1 = torch.ones((1,), dtype=torch.float32, device=x.device)
                os2 = torch.ones((1,), dtype=torch.float32, device=x.device)
            else:
                os_shape = prefix if prefix else (1,)
                os1 = torch.full(
                    os_shape, float("-inf"), dtype=torch.float32, device=x.device
                )
                os2 = (
                    os1.clone()
                    if s2 is not None
                    else torch.zeros((1,), dtype=torch.float32, device=x.device)
                )
            return [y1, y2, xo, ln, os1, os2]

        # ---- LayerNorm：mean-centered 两遍归约（fp32，逐元素先除 cols 再求和）----
        xf = x.reshape(-1, n)
        nf = float(n)
        mean = (xf / nf).sum(dim=-1, keepdim=True)
        var = (((xf - mean) ** 2) / nf).sum(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + self.epsilon)
        y = ((xf - mean) * rstd * gamma.reshape(1, -1) + beta.reshape(1, -1)).reshape(
            x.shape
        )

        def _i8(v):
            return torch.clamp(torch.round(v), -128, 127).to(torch.int8)

        if self.quant_mode == "static":
            sc1 = (
                s1.reshape(-1)[0] if s1.numel() == 1 else s1.reshape(tuple(gamma.shape))
            )
            # perTensor 下 zero_points 与 scales 同为 [1]：单元素先取标量，
            # 否则 reshape 到 gamma 形状（如 [1, 128]）会因元素数不足报错。
            off1 = (
                0.0
                if z1 is None
                else (
                    z1.reshape(-1)[0]
                    if z1.numel() == 1
                    else z1.reshape(tuple(gamma.shape))
                )
            )
            y1 = _i8(y / sc1 + off1) if self.div_mode else _i8(y * sc1 + off1)
            if s2 is not None:
                sc2 = (
                    s2.reshape(-1)[0]
                    if s2.numel() == 1
                    else s2.reshape(tuple(gamma.shape))
                )
                off2 = (
                    0.0
                    if z2 is None
                    else (
                        z2.reshape(-1)[0]
                        if z2.numel() == 1
                        else z2.reshape(tuple(gamma.shape))
                    )
                )
                y2 = _i8(y / sc2 + off2) if self.div_mode else _i8(y * sc2 + off2)
            else:
                y2 = torch.zeros((1,), dtype=torch.int8, device=x.device)
            os1 = torch.ones((1,), dtype=torch.float32, device=x.device)
            os2 = torch.ones((1,), dtype=torch.float32, device=x.device)
        else:
            yf = y.reshape(-1, n)

            def _dyn(sc):
                tmp = (
                    yf
                    if sc is None
                    else yf * (sc.reshape(-1)[0] if sc.numel() == 1 else sc.reshape(-1))
                )
                m = torch.abs(tmp).amax(dim=-1, keepdim=False)
                os_ = (m * (1.0 / 127.0)).to(torch.float32)  # kernel Muls(1/127f)
                out = _i8(tmp / os_.reshape(-1, 1)).reshape(x.shape)
                return out, (os_.reshape(prefix) if prefix else os_)

            y1, os1 = _dyn(s1)
            if s2 is not None:
                y2, os2 = _dyn(s2)
            else:
                y2 = torch.zeros((1,), dtype=torch.int8, device=x.device)
                os2 = torch.zeros((1,), dtype=torch.float32, device=x.device)

        xo = x.to(out_dtype)
        ln = y.to(out_dtype)
        return [y1, y2, xo, ln, os1, os2]


class AddLayerNormQuantV2KernelSpec:
    """kernel；收 numpy.ndarray，返 numpy.ndarray。"""

    def golden(*inputs, **kwargs):
        od = _od(kwargs)
        t = [_as_tensor(a) for a in inputs]
        outs = _compute(*t, **kwargs)
        return _cast_out(outs, od)

    # 三方竞品标杆（kernel/GEIR 通路取用；aclnn 见下方预留注释）
    third_party = {"torch": _AddLayerNormQuantV2Compose}
    tolerance = _TOL_KERNEL


# golden → pre_compare 之间的单例标记（每 case 顺序执行：golden 先于 compare）。
# additional_output=False 时 x 由内核决定不写出（def 默认 false），x 属 undefined 输出。
_LAST_ADDITIONAL_OUTPUT = True


class AddLayerNormQuantV2AclnnSpec:
    """aclnn；golden 收已 H2D 的 torch.Tensor，【直接返 Tensor】（规范 §6 / 模板要求）。"""

    def golden(*inputs, **kwargs):
        global _LAST_ADDITIONAL_OUTPUT
        # ⚠ TTK aclnn 按 C 头**参数顺序**把标量属性 positional 传入（不是 kwargs）：
        #   [0..8] 9 个输入 tensor / [9] quantMode / [10] epsilon / [11] additionalOutput /
        #   [12] divMode / [13..18] 6 个输出 tensor / [19] workspaceSize / [20] executor。
        # 只按 kwargs 取属性会让 quantMode 落回默认 "dynamic" → static 用例 golden 全错
        # （实测 y1/y2 与 NPU 差 ~94%，x/layernorm_res 正常）。这里两路都接。
        n_in = 9
        t = list(inputs[:n_in])
        merged = dict(kwargs)
        if len(inputs) >= n_in + 4 and isinstance(inputs[n_in], str):
            merged.update(
                {
                    "quantMode": inputs[n_in],
                    "epsilon": inputs[n_in + 1],
                    "additionalOutput": inputs[n_in + 2],
                    "divMode": inputs[n_in + 3],
                }
            )
        # 输出 dtype：TTK 传 tensor_dtypes（9 入 + 6 出，C 顺序），用它还原出口窄回
        td = merged.get("tensor_dtypes")
        if td and len(td) >= n_in + 6 and not merged.get("output_dtypes"):
            merged["output_dtypes"] = list(td)[n_in : n_in + 6]
        _LAST_ADDITIONAL_OUTPUT = bool(
            merged.get("additionalOutput", merged.get("additional_output", True))
        )
        od = _od(merged)
        # fp16/bf16 输入抬到 fp32 再算（与 KernelSpec 的 _as_tensor 一致）：仅向上兜底，不向下 cast；
        # 镜像 NPU 内核的加宽行为（VF 全程 fp32、出口才窄回）。Promote 开启时输入已被 TTK 抬过 → no-op。
        t = [
            a.to(torch.float32)
            if isinstance(a, torch.Tensor)
            and a.dtype in (torch.bfloat16, torch.float16)
            else a
            for a in t
        ]
        outs = _compute(*t, **merged)
        return _cast_out_tensor(outs, od)

    tolerance = _TOL_ACLNN

    # 契约上 undefined 的输出（scales2 缺省时的 y2/out_scales2）内核不写，整块比对会把
    # HBM 脏数据算进去。kernel 通路用 absolute_precision=1e9 跳过；aclnn 通路由 pre_compare
    # 把 NPU 缓冲对齐到 golden（等价"这两路不判"）。y2 的形状编码 scales2 是否存在，
    # 因此无需属性即可判定（存在 = x1 形状；缺省 = [1] 占位）。
    # 注：static 的 out_scales1/2 实测 NPU 侧恒为 1.0（与 golden 的 ones(1) 一致），无需屏蔽。
    def pre_compare(*arrays):
        n = len(arrays) // 2
        if n < 6:
            return
        outs, golds = arrays[:n], arrays[n:]
        try:
            y2 = golds[1]
            dual = (y2.numel() if hasattr(y2, "numel") else np.asarray(y2).size) > 1
        except Exception:
            dual = True
        if not dual:
            try:
                outs[5][:] = golds[5][:]
            except Exception:
                pass
        # additional_output=False：x 不写出（kernel `if (outputX_)` 守写；aclnn 契约称 xOut 为脏数据），
        # 对齐到 golden 即"不判这一路"。layernorm_res 不受该属性影响，仍正常比对。
        if not _LAST_ADDITIONAL_OUTPUT:
            try:
                outs[2][:] = golds[2][:]
            except Exception:
                pass

    # 【预留】TTK aclnn 通路当前不取用 third_party（仅 kernel/GEIR 取用），
    # 写在此处不生效也无副作用；待该通路支持三方后自动接上，无需改动本文件。
    # 注：三方取数走 remote/client.py，当前以 op_name 查 spec，届时注册名需对齐。
    # third_party = {"torch": _compose_torch}


# 【不适用】geir 通路：本算子未注册 AddLayerNormQuantV2 infershape（arch910b/canndev/built-in 三处均无，
# 见 01 §3.3），GE 图模式不可编译（Cannot find Node op1[...] custom infer_datatype func）。
# 【不适用】e2e/torch 通路：torch_npu 未引用 aclnnAddLayerNormQuantV2（strings 查 0 命中，
# 对照 aclnnAddRmsNormQuant 有命中，方法有效）。
