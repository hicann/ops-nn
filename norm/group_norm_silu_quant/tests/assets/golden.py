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
"""
TTK golden plugin for group_norm_silu_quant (kernel mode, arch35/Ascend950).

**Implemented with torch tensor ops** (torch.var_mean / rsqrt / addcmul / F.silu / round+clamp),
not hand-written numpy formulas.

刻意不用层级 API `F.group_norm`：它带 BN 系列的训练态保护
（"Expected more than 1 value per channel when training"），每通道只剩 1 个元素时直接拒收
（2 维输入 (N,C) 且 group=C 即触发，实测 case00608 崩）。那是网络层的约束，与本算子的
数学定义无关——算子本身 elemNum=1 合法。张量算子拼接同样满足红线 R3「竞品算子拼接实现」。 红线 R3：golden 只能"竞品接口实现"或"竞品算子拼接实现"，禁 numpy 纯公式——
纯公式与被测内核容易犯同一个错，拿和内核一样烂的参照去比，会把精度短板伪装成达标。

Reference for IO/attr semantics:
    ops-nn/norm/group_norm_silu_quant/tests/st/aclnnGroupNormSiluQuant/executor_aclnnGroupNormSiluQuant.py
and the requirement/design docs (changwei_deliverables/GroupNormSiluQuant/).

Compute (per group of elemNum = (C/num_groups)*HW elements):
    mean = mean(x)                          # over each group
    var  = var(x, ddof=0)                   # population variance
    rstd = 1/sqrt(var + eps)
    x_norm = (x - mean) * rstd
    y_f    = x_norm * gamma[c] + beta[c]     # per-channel affine
    silu   = activate_silu ? y_f*sigmoid(y_f) : y_f
    y_i8   = clamp(round(silu / quantScale), -128, 127)   # per-tensor(len 1) or per-channel(len C)
    meanOut/rstdOut: per group, shape (N, num_groups), cast back to x dtype.

Canonical IO order (group_norm_silu_quant_def.cpp):
    inputs : x, gamma, beta, quantScale      (x/gamma/beta same dtype bf16|fp16; quantScale fp32)
    outputs: yOut(int8), meanOut, rstdOut
    attrs  : num_groups(REQUIRED int), eps(OPTIONAL float=1e-5), activate_silu(OPTIONAL bool=True)

TTK passes input arrays positionally in CSV input_shapes order; attributes + output_dtypes via **kwargs.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def _f32(a):
    return np.asarray(a).astype(np.float32)


def _work_dtype(*arrays):
    """计算档位:只向上兜底 —— f16/bf16 抬 f32;任一输入是 f64(Promote 抬档)则全 f64。

    golden 自己不做 Promote:三方档(cross_check)由 TTK 抬(入口 dtype 已整体高一档),
    这里再 astype(float64) 是重复动作、还会掩盖 Promote 空转;两方泛化档 TTK 不提升,
    规范(ttk_golden_logic.md 三/四)要求跟 NPU 的加宽行为——内核把 f16/bf16 抬到 f32
    做中间量,f32 不再加宽。
    """
    if any(np.asarray(a).dtype == np.float64 for a in arrays if a is not None):
        return np.float64
    return np.float32


def _as_work(a, work):
    return np.asarray(a).astype(work)


def _cast_back(arr_f32, target):
    if target == "bfloat16":
        return (
            arr_f32.astype(_bf16) if _bf16 is not None else arr_f32.astype(np.float32)
        )
    return arr_f32.astype(target)


# aclnn 头文件的属性名与 def.cpp 不同(group/activateSilu vs num_groups/activate_silu),
# 两套都要认: kernel/geir 通路按 def 名下发, aclnn 通路按头文件名。
_ATTR_ALIAS = {
    "num_groups": ("num_groups", "group"),
    "activate_silu": ("activate_silu", "activateSilu"),
    "eps": ("eps",),
}


def _attr(kwargs, name, default):
    for key in _ATTR_ALIAS.get(name, (name,)):
        v = kwargs.get(key)
        if v is None:
            attrs = kwargs.get("attributes")
            if isinstance(attrs, dict):
                v = attrs.get(key)
        if v is not None:
            return v
    return default


def __golden_group_norm_silu_quant(x, gamma, beta, quant_scale, **kwargs):
    num_groups = int(_attr(kwargs, "num_groups", 1))
    # 计算档位由 _work_dtype 定:三方档 TTK 已 Promote(入口 f64), golden 不再自行抬档;
    # 泛化档跟内核(f16/bf16 抬 f32, f32 不加宽)。量化前 out 的精度随之决定 round 在
    # .5 边界的归属, 因此 eps 也必须同档, 不能固定 f64。
    _work = _work_dtype(x, gamma, beta, quant_scale)
    eps = torch.from_numpy(np.asarray([_attr(kwargs, "eps", 1e-5)], dtype=_work))[0]
    silu = _attr(kwargs, "activate_silu", True)
    if isinstance(silu, str):
        silu = silu.strip().lower() in ("1", "true", "yes")
    silu = bool(silu)

    output_dtypes = kwargs.get("output_dtypes")

    def _od(i, default):
        if output_dtypes and i < len(output_dtypes):
            od = output_dtypes[i]
            return od[0] if isinstance(od, (tuple, list)) else str(od)
        return default

    x_dt = str(np.asarray(x).dtype)
    mean_dt = _od(1, x_dt)
    rstd_dt = _od(2, x_dt)

    xf = _as_work(x, _work)
    N, C = xf.shape[0], xf.shape[1]
    HW = int(np.prod(xf.shape[2:])) if xf.ndim > 2 else 1
    G = num_groups

    # 可选输入缺省:gamma/beta 是 OPTIONAL 参数, 不传时按算子语义取 gamma=1、beta=0
    # (kernel arch35 ..._split_reduce.h:265-266 `hasGamma ? gF[ci] : 1.0f` / `hasBeta ? bF[ci] : 0.0f`)。
    # 此前直接 _f32(gamma) 会在 gamma 为 None 时抛异常, 导致该档用例判成 GOLDEN_FAILURE ——
    # 等于"可选输入不给"这一档从来没被真正验证过(issue #21 的 L1_014 正是这一档)。
    if gamma is None:
        gamma = np.ones((C,), dtype=np.float32)
    if beta is None:
        beta = np.zeros((C,), dtype=np.float32)

    # 空 Tensor(任意维度为 0):N==0 时下面的 reshape/var_mean 会抛异常(判成 GOLDEN_FAILURE), 需短路。
    # 空 Tensor 契约:**meanOut 填 0, rstdOut 填 NAN**。
    # 权威依据是本算子自己的两处资料(README.md:73 / docs/aclnnGroupNormSiluQuant.md:112)与手写 aclnn
    # 的实现(op_host/op_api/aclnn_group_norm_silu_quant.cpp:251 `IsEmpty()` 分支显式
    # FillScalar(meanOut, 0) + FillScalar(rstdOut, NAN) 后直接返回, 不下发内核)。
    # ⚠️ arch35 内核的 empty 分支原本对 mean 也填 NAN, 与 aclnn 通路语义不一致(GE 图通路才会走到),
    # 已在 op_kernel/arch35/..._empty_tensor.h 修正为填 0。golden 对齐契约, 不对齐一时的实现。
    if xf.size == 0:
        y_empty = np.zeros(np.asarray(x).shape, dtype=np.int8)
        mean_e = np.zeros((N, G), dtype=np.float32)
        rstd_e = np.full((N, G), np.nan, dtype=np.float32)
        return [y_empty, _cast_back(mean_e, mean_dt), _cast_back(rstd_e, rstd_dt)]

    # ── 用 torch 库算子拼接，不手写公式 ──
    # torch.var_mean(unbiased=False) 给出与算子定义一致的总体方差与均值。
    xt = torch.from_numpy(xf).reshape(N, C, HW)
    gt = torch.from_numpy(_as_work(gamma, _work).reshape(-1))
    bt = torch.from_numpy(_as_work(beta, _work).reshape(-1))

    xg = xt.reshape(N, G, -1)
    var_t, mean_t = torch.var_mean(xg, dim=-1, unbiased=False, keepdim=True)
    rstd_t = torch.rsqrt(var_t + eps)

    # ⚠️ 不能直接用 F.group_norm：每通道只剩 1 个元素时（如 2 维输入 (N,C) 且 group=C）
    # 它会抛 "Expected more than 1 value per channel when training"——那是 BN 系列的
    # 训练态保护，对本算子不适用（算子本身支持 elemNum=1）。实测 case00608 ((1,24),...) 即崩。
    # 改用 torch 的归一化+仿射算子拼接，语义与 F.group_norm 一致但不带这条限制：
    #   normalized = (x - mean) * rstd   （mean/rstd 已由 torch.var_mean 给出，同一套统计量）
    #   out = normalized * gamma[c] + beta[c]
    gn_t = ((xg - mean_t) * rstd_t).reshape(N, C, HW)
    out_t = torch.addcmul(bt.reshape(1, C, 1), gn_t, gt.reshape(1, C, 1))
    if silu:
        out_t = F.silu(out_t)

    qs = _as_work(quant_scale, _work).reshape(-1)
    if qs.size == C:
        scale_t = torch.from_numpy(qs).reshape(1, C, 1)  # per-channel
    else:
        scale_t = torch.tensor(float(qs[0]), dtype=xt.dtype)  # per-tensor
    q_t = torch.clamp(torch.round(out_t / scale_t), -128.0, 127.0)
    y_i8 = q_t.to(torch.int8).numpy().reshape(np.asarray(x).shape)

    mean = mean_t.numpy()
    rstd = rstd_t.numpy()

    mean_out = _cast_back(mean.reshape(N, G), mean_dt)
    rstd_out = _cast_back(rstd.reshape(N, G), rstd_dt)
    return [y_i8, mean_out, rstd_out]


# 模块级别名:类体内直接引用 `__golden_...` 会触发 Python 的 name mangling
# (被改写成 _GroupNormSiluQuantSpec__golden_...), 导致 Spec.golden 调用时 NameError。
_golden_impl = __golden_group_norm_silu_quant


class _Compose:
    """竞品标杆(A100 上执行): 走 torch 的层级 API `F.group_norm` + `F.silu`。

    与 golden 的张量算子拼接是**相互独立的实现路径**: golden 因 elemNum=1 的训练态保护
    刻意绕开 F.group_norm, 第三方标杆这里正面用它 —— 竞品的真实用法就是这一句。
    命中该保护(每通道只剩 1 个元素, 如 2 维输入 (N,C) 且 group=C)时回退到统计量拼接。

    ⚠️ **mean/rstd 这两路的第三方标杆不构成独立证据**: 任何 torch 实现都只能是
    `torch.var_mean(unbiased=False)`, 与 golden 逐位同源, cross_check 的分母会塌到
    safe_div 的 err 地板。只有 y(int8) 那一路走的是真正不同的代码路径。

    `**kwargs` 必须留: 用例 CSV 带 input_formats 时服务端会把逐输入 format 并进调用
    kwargs, 不吞掉会直接 TypeError 让整条第三方标杆 FAIL。
    """

    def __call__(self, x, gamma=None, beta=None, quantScale=None, **kwargs):
        # 形参必须叫 quantScale(驼峰): 服务端 `_bind`/`bind_params` 按**名字**绑定,
        # 名字来自 ops-info.json 的 def 名(core_modules/npu/op/profiling.py:184
        # `input_names = [ipt["name"] for ipt in op_info["inputs"]]`), def.cpp 里
        # 这个输入就叫 quantScale。写成 quant_scale 会绑不上、保持 None, 再 .to()
        # 直接 AttributeError —— 实测 40 例第三方标杆连续失败触发熔断, 而 gamma/beta
        # 因名字与 def.cpp 一致不受影响。golden() 走的是位置传参, 不受此约束。
        quant_scale = quantScale
        num_groups = int(_attr(kwargs, "num_groups", 1))
        eps = float(_attr(kwargs, "eps", 1e-5))
        silu = _attr(kwargs, "activate_silu", True)
        if isinstance(silu, str):
            silu = silu.strip().lower() in ("1", "true", "yes")

        dev = x.device
        xf = x.to(torch.float32)
        N, C = xf.shape[0], xf.shape[1]
        if xf.numel() == 0:
            # 必须与 golden 的空 Tensor 契约一致(见 __golden_group_norm_silu_quant 的
            # xf.size == 0 分支): yOut 空、meanOut 填 0、rstdOut 填 NAN。
            # 不短路的话 torch.var_mean 在空归约轴上返回 NaN, meanOut 会得到 NaN 而非 0,
            # 与 golden 背离(实测 (2,16,2,0,4) G=2: 短路前 [nan]*4, 短路后 [0.0]*4)。
            # 注意这条背离**不会被 cross_check 抓到**: 该档 rstdOut 按契约两边都是 NaN、
            # meanOut 两边全零, 相对量无定义, mare/mere/rmse 恒为 None —— 实测短路前后
            # 22 例空档的指标逐字节相同。所以空档的真实把关在 kernel 泛化(NPU vs golden,
            # --compare close, NaN 感知)那条腿上; 这里短路是为了让第三方标杆不自相矛盾,
            # 不要指望它能让三方判据重新生效。
            return [
                torch.zeros(tuple(x.shape), dtype=torch.int8, device=dev),
                torch.zeros((N, num_groups), dtype=x.dtype, device=dev),
                torch.full((N, num_groups), float("nan"), dtype=x.dtype, device=dev),
            ]
        HW = int(torch.tensor(xf.shape[2:]).prod().item()) if xf.dim() > 2 else 1
        G = num_groups

        gt = (
            torch.ones(C, dtype=torch.float32, device=dev)
            if gamma is None
            else gamma.to(torch.float32).reshape(-1)
        )
        bt = (
            torch.zeros(C, dtype=torch.float32, device=dev)
            if beta is None
            else beta.to(torch.float32).reshape(-1)
        )

        x3 = xf.reshape(N, C, HW)
        xg = x3.reshape(N, G, -1)
        var_t, mean_t = torch.var_mean(xg, dim=-1, unbiased=False, keepdim=True)
        rstd_t = torch.rsqrt(var_t + eps)

        try:
            out_t = F.group_norm(x3, G, weight=gt, bias=bt, eps=eps)
        except (RuntimeError, ValueError):
            # 每通道 1 元素触发 BN 训练态保护 —— 回退到与 golden 同形的统计量拼接
            out_t = torch.addcmul(
                bt.reshape(1, C, 1),
                ((xg - mean_t) * rstd_t).reshape(N, C, HW),
                gt.reshape(1, C, 1),
            )
        if silu:
            out_t = F.silu(out_t)

        qs = quant_scale.to(torch.float32).reshape(-1)
        scale_t = qs.reshape(1, C, 1) if qs.numel() == C else qs[0]
        q_t = torch.clamp(torch.round(out_t / scale_t), -128.0, 127.0)

        y_i8 = q_t.to(torch.int8).reshape(x.shape)
        return [
            y_i8,
            mean_t.reshape(N, G).to(x.dtype),
            rstd_t.reshape(N, G).to(x.dtype),
        ]


def _inject_nonfinite_one(arr):
    """把 +inf / -inf / nan 定点写进 arr 的前 3 个元素(元素数 < 3 则不注入)。

    位置固定不随机: 随机会让复现依赖 seed, 小 shape 时还可能一个都注不进去。
    不能靠 CSV 的 input_data_ranges 写 inf/nan —— TTK 的 RandomData 会把值域里的
    inf/nan 钳到 dtype 极值(ttk/utilities/data.py `_digitize_inf_nan`)。
    """
    import numpy as _np

    if arr is None:
        return arr
    out = _np.ascontiguousarray(arr).copy()
    if out.size < 3:
        return out
    flat = out.reshape(-1)
    flat[0], flat[1], flat[2] = _np.inf, -_np.inf, _np.nan
    return out


class GroupNormSiluQuantSpec:
    """判据声明。**必须显式给 int8 输出声明 quant 标准**:
    y 是量化输出, 不声明时 TTK 会把 int8 硬路由到 binary_equal(逐位相等), 量化舍入产生的 ±1 LSB
    会被大面积误判为失败——实测新配额集 211 例里 26 例"失败"全部是 |diff|==1(最小那例 dump:
    32 个元素中 6 个差 1、最大 |diff|=1)。quant 标准的判定是 |out-golden|>1 才计错, 且 ptol 默认 0
    (一个都不许超), 既不放水也不误杀。

    浮点输出(mean/rstd)声明 CANN 开源精度标准 `stat_rel_err`(mere < th 且 mare < 10*th, th 按 dtype 取
    2^-8/2^-10/2^-13)。**不能沿用 TTK 默认的 isclose**:其 atol=1e-8 低于 fp32/bf16 在常见量级上的分辨率
    (bf16 尾数仅 8 位, 1 ULP 的相对误差就有 ~0.4%), 等价于要求逐位相等。实测 case00603_rg(2,5120,7) bf16:
    2560 个 mean 元素里 10 个不相等, **每一个都恰好差 1 ULP**(ULP 倍数 max=1.0000, 超 1 ULP 的 0 个)
    —— 内核 fp32 累加后舍入到 bf16 与 torch 求和次序不同, 边界值差一格, 任何实现都做不到更好。
    """

    @staticmethod
    def customize_inputs(*arrays, **kwargs):
        """签名必须是**可变参数**。

        kernel / geir 通路只传 4 个输入(x, gamma, beta, quantScale), 而 aclnn 通路会把
        用例集 `tensor_view_shapes` 里的**全部张量(含 3 个输出)连同 3 个属性一并传进来**
        —— 实测写成 4 个位置参数时 aclnn 直接
        `customize_inputs() takes 4 positional arguments but 10 were given` → INPUT_GEN_FAILURE。

        只注入数据面的第 0 个张量(x), 其余原样返回: 注入权重/量化系数会让整通道输出非有限,
        掩盖"非有限值沿计算链如何传播"这一档真正要看的东西。
        """
        if not arrays:
            return arrays
        name = kwargs.get("testcase_name", "") or ""
        if "_nonfinite" not in name:
            return arrays
        return (_inject_nonfinite_one(arrays[0]),) + tuple(arrays[1:])

    third_party = {"torch": _Compose}

    tolerance = {
        "int8": {"standard": "quant"},
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def golden(x, gamma, beta, quant_scale, **kwargs):
        return _golden_impl(x, gamma, beta, quant_scale, **kwargs)


def _aclnn_compose(
    self_tensor, gammaOptional=None, betaOptional=None, quantScale=None, **kwargs
):
    """aclnn 通路的第三方标杆 compose。

    **必须是模块级函数, 且形参名用 aclnn 头文件口径**:
    - 服务端 `_bind`/`bind_params` 按**名字**绑参, 名字由用例集张量名下发 ——
      kernel/geir 走 def.cpp 名(x/gamma/beta/quantScale), aclnn 走头文件名
      (self/gammaOptional/betaOptional/quantScale)。父类 `_Compose.__call__` 首参叫 x,
      在 aclnn 上服务端直接 400
      `parameter 'x' of _Compose.__call__ is not a known input or attribute name`,
      连续 30 例后触发熔断中止整批(实测喂 80 只收 51)。
    - 首参名为 `self_tensor`: aclnn 头文件的主输入叫 `self`, 但 xpu-server 的 _invoke_function
      会无条件丢弃名为 self 的入参(为 torch 内建准备), 故 TTK 客户端下发时已改名为 self_tensor。
      故写成函数(服务端 `_invoke` 对函数与类都支持)。同理 Spec.golden 也是 @staticmethod。
    """
    return _Compose()(
        self_tensor,
        gamma=gammaOptional,
        beta=betaOptional,
        quantScale=quantScale,
        **kwargs,
    )


class AclnnGroupNormSiluQuantSpec(GroupNormSiluQuantSpec):
    """aclnn 通路 Spec。

    **与 kernel/geir Spec 的唯一差别是接参方式**:aclnn 通路 golden **按 aclnn 头文件的参数
    位置**接位置参数, 而不是按 def.cpp 的名字。头文件顺序(docs/aclnnGroupNormSiluQuant.md:59-71):

        aclnnGroupNormSiluQuantGetWorkspaceSize(
            self, gammaOptional, betaOptional, quantScale,   <- 4 个输入
            group, eps, activateSilu,                        <- 3 个属性
            out, meanOut, rstdOut, ...)                      <- 3 个输出

    属性在 aclnn 侧叫 group/activateSilu, 在 def.cpp 侧叫 num_groups/activate_silu ——
    `_ATTR_ALIAS` 两套都认, 所以判据与第三方标杆可以整份复用父类, 不另起一套(避免两边漂移)。
    """

    third_party = {"torch": _aclnn_compose}

    @staticmethod
    def golden(self, gammaOptional, betaOptional, quantScale, *args, **kwargs):
        # 属性既可能以位置参数跟在 quantScale 之后, 也可能以 kwargs 下发, 两种都接。
        pos = list(args)
        if pos:
            kwargs.setdefault("num_groups", pos[0])
        if len(pos) > 1:
            kwargs.setdefault("eps", pos[1])
        if len(pos) > 2:
            kwargs.setdefault("activate_silu", pos[2])
        return _golden_impl(self, gammaOptional, betaOptional, quantScale, **kwargs)


__spec__ = {
    "group_norm_silu_quant": "GroupNormSiluQuantSpec",  # kernel / geir 共用 op_name
    "aclnnGroupNormSiluQuant": "AclnnGroupNormSiluQuantSpec",  # aclnn 按 api_name 绑定
}
__golden__ = {"kernel": {"group_norm_silu_quant": "__golden_group_norm_silu_quant"}}
