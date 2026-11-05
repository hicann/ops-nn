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

**Implemented with torch tensor ops** (mean / rsqrt / mul+add / exp+div / round+clamp),
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

import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


# 量化(整型)输出的参照档:整型无双标杆, 必须与 NPU 计算档同精度。内核对 f16/bf16/f32
# 输入一律在 f32 下算统计与量化, 故平级档恒为 f32。
_PEER_DTYPE = np.float32


def _work_dtype(x):
    """计算档位 = TTK 交付 x 时的 dtype, 对等取用, 不自行抬档(抬档决定权在 TTK 的
    golden_mode=Promote, 逐输入各抬一档)。f16/bf16 加宽到 f32 跟随内核行为; f32/f64 照用。

    🔴 只能看 x, 不能对全部入参取 max: quantScale 是 f32 输入、三方档被抬成 f64,
    原 `any(dtype==f64)` 会把统计量也拖到 f64, 越过 TTK 对 x 只抬到 f32 的决定。
    实测只在满量程兑现(常规量程 0/384 不同、满量程 64/64), 但该档会让 golden 与被测落到
    相邻两个 bf16 格点, 使 cross_check 特殊位的逐位相等不成立(L1_110/150/213/226 等 8 例)。
    """
    dt = np.asarray(x).dtype
    return np.float64 if dt == np.float64 else np.float32


def _to_np(t):
    """aclnn 通路 golden 入参是 torch.Tensor, 统一转 numpy(kernel/geir 本就是 numpy, 直通)。
    bf16 经 uint16 按位 view 到 ml_dtypes(np.asarray 对 torch bf16 抛 TypeError)。"""
    if t is None or isinstance(t, np.ndarray) or not isinstance(t, torch.Tensor):
        return t
    if t.dtype == torch.bfloat16:
        if _bf16 is None:
            raise RuntimeError("bfloat16 input requires ml_dtypes")
        return t.view(torch.uint16).numpy().view(_bf16)
    return t.numpy()


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


# ── 双浮点(TwoSum/Dekker)用到的常量: 全部从 torch.finfo 推导, 不写死 ──
# torch.finfo 只提供 eps / max / tiny, 没有 numpy 的 nmant / maxexp, 故由二者反推:
#   有效位数 p: eps = 2^-(p-1)        -> p = 1 - log2(eps)      (fp32: 24, fp64: 53)
#   最大二进阶码 e_max: frexp(max)    -> 指数 - 1               (fp32: 127, fp64: 1023)
def _mant_digits(dt):
    return int(round(1.0 - math.log2(torch.finfo(dt).eps)))


def _max_binary_exp(dt):
    return math.frexp(torch.finfo(dt).max)[1] - 1


def _dekker_split(dt):
    """Dekker 拆分常量 = 2^ceil(p/2) + 1。fp32: 2^12+1 = 4097 ; fp64: 2^27+1 = 134217729。"""
    p = _mant_digits(dt)
    return float((1 << ((p + 1) // 2)) + 1)


def _sum_exp_headroom(dt):
    """Σx 不溢出所需的阶码余量。
    溢出门限是 2^(e_max+1)(fp32: FLT_MAX = 2^128, 注意**不是** 2^127),
    故缩放后允许的阶码上界取 e_max + 0.5, 即留半位余量: 2^127.5 = 2.41e38 < 3.40e38。"""
    return _max_binary_exp(dt) + 0.5


def _sq_exp_headroom(dt):
    """Σ(2x)^2 不溢出: 要求 2*(e+1) + log2(n) <= e_max+1, 即
    e + 1 + 0.5*log2(n) <= (e_max+1)/2; 再留半位余量。fp32: 128/2 - 0.5 = 63.5。"""
    return (_max_binary_exp(dt) + 1) / 2.0 - 0.5


def _two_sum(a, b):
    """Knuth TwoSum: a + b = s + err, 在 a/b 的 dtype 下**精确**(无需更高精度)。"""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _df_add(ah, al, bh, bl):
    """双浮点(hi, lo)相加。"""
    s, e = _two_sum(ah, bh)
    e = e + (al + bl)
    return _two_sum(s, e)


def _df_tree_sum(hi, lo):
    """沿最后一维做**双浮点两两树求和**: 每层一次张量运算, 共 ceil(log2(n)) 层。
    拓扑是"两两配对 + 奇数进位"的平衡二叉树 —— 与任何实现无关的规范形态。"""
    while hi.shape[-1] > 1:
        n = hi.shape[-1]
        half = n // 2
        sh, sl = _df_add(
            hi[..., 0 : 2 * half : 2],
            lo[..., 0 : 2 * half : 2],
            hi[..., 1 : 2 * half : 2],
            lo[..., 1 : 2 * half : 2],
        )
        if n % 2:  # 节点数为奇: 落单的一条原样进下一层
            sh = torch.cat([sh, hi[..., -1:]], dim=-1)
            sl = torch.cat([sl, lo[..., -1:]], dim=-1)
        hi, lo = sh, sl
    return hi[..., 0], lo[..., 0]


def _split(a):
    """Dekker 拆分, 用于取精确积的低位。常数按 dtype 取 2^ceil(p/2)+1。"""
    c = a * _dekker_split(a.dtype)
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    """精确乘积 a*b = p + err。"""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl


def _pow2_scale(xg):
    """逐组选 pw = 2^k, 使 Σx 与 Σ(x-mean)^2 都不溢出; 乘除 2 的幂精确, 不引入舍入。
    指数在**对数域**相加 —— 写成 max|x| * n 会在满量程档自身溢出成 inf(踩过)。
    仅 fp32 需要: fp64 的动态范围下 bf16 满量程的平方(~1.1e77)远未触顶。"""
    if xg.dtype != torch.float32:
        return None
    mx = xg.abs().amax(dim=-1, keepdim=True)
    # 非有限输入(含 ±inf / nan)不做缩放: 否则 log2(inf)=inf -> pw=inf -> x/inf=nan,
    # 把本该按契约自然传播的 inf 变成 nan(实测 L0_034 的 x 为 fp16 inf, rstd 整列 nan)。
    mx = torch.where(torch.isfinite(mx), mx, torch.ones_like(mx))
    # log2 下界保护: 全零组的 max|x| = 0, log2(0) = -inf 会污染后续 ceil/clamp
    lmx = torch.log2(torch.clamp(mx, min=torch.finfo(xg.dtype).tiny))
    ln = float(torch.log2(torch.tensor(float(xg.shape[-1]))))
    e_sum = torch.ceil(lmx + ln - _sum_exp_headroom(xg.dtype))
    e_sq = torch.ceil(lmx + 1.0 + 0.5 * ln - _sq_exp_headroom(xg.dtype))
    return torch.exp2(torch.clamp(torch.maximum(e_sum, e_sq), min=0.0))


def _df_div_n(hi, lo, n):
    """双浮点除以标量计数 n: 先取商再用余项修正, 避免 (hi+lo) 先折回再除的双重舍入。"""
    q = hi / n
    ph, pl = _two_prod(q, n)
    rh, rl = _df_add(hi, lo, -ph, -pl)
    return q + (rh + rl) / n


def _df_rsqrt(var, eps):
    """rstd = 1/sqrt(Var+eps) 的地板(精确值正确舍入到 var.dtype 的唯一值)。

    原写法 `1.0/torch.sqrt(var+eps)` 叠三次舍入, 对 4013 组比精确有理数参照
    (Fraction+isqrt)实测最坏 2 ULP、fp32 仅 70.35% 命中地板; 本实现两档均 100%。
    rstd 乘进归一化链, 这 1~2 ULP 经放大足以改变 int8 落格, 与 _group_stat 同理。

    做法: TwoSum 取 var+eps 的精确和 -> 按 4 的幂归一(Dekker 拆分在 |s| 近上下界会溢出/
    丢位) -> 一次 Newton 用双浮点算残差 -> 末尾只舍入一次。非有限/零输入兜回原写法,
    模式与改前逐位一致。
    """
    one = torch.ones_like(var)
    naive = one / torch.sqrt(var + eps)  # 兜底 = 原写法, 保非有限语义不变

    sh0, sl0 = _two_sum(var, eps)
    _, e2 = torch.frexp(sh0)
    k = torch.ceil(e2.to(sh0.dtype) / 2)
    k = torch.where(torch.isfinite(sh0) & (sh0 > 0), k, torch.zeros_like(k))
    pw = torch.exp2(-2.0 * k)
    sh, sl = sh0 * pw, sl0 * pw

    y = one / torch.sqrt(sh)
    y2h, y2l = _two_prod(y, y)  # y^2 精确
    ph, pe = _two_prod(sh, y2h)  # s * y^2 的双浮点积
    pe = pe + (sh * y2l + sl * y2h) + sl * y2l
    th, tl = _two_sum(ph, pe)
    eh, el = _df_add(one, torch.zeros_like(one), -th, -tl)  # e = 1 - s*y^2
    refined, _ = _two_sum(y, y * (eh + el) * 0.5)  # Newton 一次 + 末尾单次舍入
    refined = refined * torch.exp2(-k)  # 还原缩放: 1/sqrt(s) = y'*2^-k

    ok = torch.isfinite(naive) & (naive > 0) & torch.isfinite(refined)
    return torch.where(ok, refined, naive)


def _group_stat(xg):
    """组内统计量 = **精确统计量正确舍入到 xg.dtype 的那个唯一值**(下称地板)。

    做法: 两遍法, 每一遍都用**双浮点(TwoSum)补偿**的两两树求和 ——
      第一遍 Σx -> mean; 第二遍 Σ(x-mean)^2 -> var。

    ⚠️ 拓扑必须与被测实现无关: 本函数只有"两两配对树求和"这一个规范形态, 不含 lane 宽度、
    向量长度、链长、分段切分等任何内核信息。照抄被测切分会让"规约拓扑不当"这类缺陷在比对里
    恒等消失。

    为什么从"手写 fp32 Welford 平衡树"换成补偿两遍法(实测, 210 组随机抽样对精确有理数地板):
        实现                     mean 命中地板    var 命中地板
        Welford 平衡树(原实现)        77.1%          47.1%
        本实现(补偿两遍法)           100.0%          98.6%
    原实现只有 47% 的 var 落在地板上, 即 golden 自身就带 1~2 ULP 的归约误差; 该误差经
    |gamma|/|quant_scale|(实测可达 2.1e9)放大后足以改变 int8 落格, 使"内核对不对"的判定
    取决于 golden 自己偏哪边(实测存在内核已落地板、却因 golden 偏 1 ULP 而判红的用例)。

    为什么补偿不违反"两方平级": 地板是**该 dtype 可表示的唯一正确值**, 纯 fp32 实现即可命中
    (已验证: TwoSum + Dekker 的纯 fp32 方案逐位命中同一值)。故这不是把 golden 抬到 fp64,
    而是让两腿有一个与实现无关的共同目标 —— 都打到地板就逐位一致, 不存在谁迁就谁。

    为什么不是两遍法的老问题: "代入近似均值 Ê 使偏差为 Var+(E-Ê)²"只在 Ê 粗糙时成立;
    这里第一遍已把 mean 做到地板, (E-Ê)² 是 (半个 ULP)² 量级。残差平方的溢出由二次幂
    预缩放解决(见 _pow2_scale), 且预缩放在正常档逐位不改变结果(已验 n=192/343/729)。

    非有限与满量程: 预缩放后 Σ 不溢出, mean 恒为有限且命中地板; var 还原时若超出 dtype
    上界则自然成 inf —— 与内核在该档的落位一致(实测两侧 var 同为 inf、rstd 同为 0)。
    输入本身含 nan/inf 时由 TwoSum 自然传播。
    """
    pw = _pow2_scale(xg)
    xs = xg if pw is None else xg / pw
    n = torch.tensor(float(xg.shape[-1]), dtype=xs.dtype)
    sh, sl = _df_tree_sum(xs, torch.zeros_like(xs))
    mean_s = _df_div_n(sh, sl, n)
    if xg.shape[-1] == 1:
        # 单元素组: 按 Welford 定义 M2 恒为 0, 故 var 恒为 0。
        # 不能走两遍法: x 为 ±inf 时 (x-mean) = inf-inf = nan, 会把 var/rstd 污染成 nan
        # (内核 VFInnerWelfordParallelUpdateWithInit 对同一问题已显式置 0, 两侧须一致)。
        var_s = torch.zeros_like(mean_s)
        if pw is None:
            return mean_s, var_s
        pw1 = pw.squeeze(-1)
        return mean_s * pw1, var_s
    d = xs - mean_s.unsqueeze(-1)
    ph, pl = _two_prod(d, d)
    mh, ml = _df_tree_sum(ph, pl)
    var_s = _df_div_n(mh, ml, n)
    if pw is None:
        return mean_s, var_s
    pw0 = pw.squeeze(-1)
    return mean_s * pw0, (var_s * pw0) * pw0


def __golden_group_norm_silu_quant(x, gamma, beta, quant_scale, **kwargs):
    num_groups = int(_attr(kwargs, "num_groups", 1))
    # 计算档位 = TTK 交付 x 的 dtype(见 _work_dtype), 不看 quantScale。
    # 量化前 out 的精度决定 round 在 .5 边界的归属, 故 eps 也须同档, 不能固定 f64。
    _work = _work_dtype(x)
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

    # 空 Tensor(任意维度为 0):N==0 时下面的 reshape/归约会抛异常(判成 GOLDEN_FAILURE), 需短路。
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
    xt = torch.from_numpy(xf).reshape(N, C, HW)
    gt = torch.from_numpy(_as_work(gamma, _work).reshape(-1))
    bt = torch.from_numpy(_as_work(beta, _work).reshape(-1))

    xg = xt.reshape(N, G, -1)
    # 统计量走自实现的补偿两遍法(见 _group_stat): 结果 = 精确统计量正确舍入到工作 dtype 的
    # 那个唯一值(地板), 与实现拓扑无关。工作 dtype 由 _work_dtype 决定, golden 不自行抬档。
    mean_t, var_t = _group_stat(xg)
    mean_t = mean_t.unsqueeze(-1)
    var_t = var_t.unsqueeze(-1)
    # rstdOut = 1/sqrt(Var+eps): 走 _df_rsqrt 取**地板**。按公式字面写 1.0/sqrt(var+eps)
    # 会叠三次舍入(实测最坏 2 ULP、近三成不在地板上), 与 var 本身走补偿两遍法是同一个
    # 要求 —— 真值的每一步都必须是该 dtype 的唯一正确值, 不能只管统计量不管 rstd。
    rstd_t = _df_rsqrt(var_t, eps)

    # ⚠️ 不能直接用 F.group_norm：每通道只剩 1 个元素时（如 2 维输入 (N,C) 且 group=C）
    # 它会抛 "Expected more than 1 value per channel when training"——那是 BN 系列的
    # 训练态保护，对本算子不适用（算子本身支持 elemNum=1）。实测 case00608 ((1,24),...) 即崩。
    # 改用 torch 的归一化+仿射算子拼接，语义与 F.group_norm 一致但不带这条限制：
    #   normalized = (x - mean) * rstd   （mean/rstd 由上面按公式算出，同一套统计量）
    #   out = normalized * gamma[c] + beta[c]
    # ── 量化输出(yOut, int8)走【平级档】, 与统计量输出分开 ──
    # 依据 ttk_golden_logic.md §3.5: DTYPE_PROMOTE_MAP 无任何整型条目 ->「整型没有三方:
    # 既不提升, 也组不成双标杆」; §一:「只有三方比对才有真值, 两方是平级实现互比」。
    # 即 int8 输出没有"真值"可言, 它的参照必须与 NPU 同精度。而 TTK 的 Promote 是
    # **用例级**的(任一输出声明 cross_check 即抬高 input_arrays), 于是三方档下量化链会被
    # 连带抬到 fp64 —— 判据(quant, abs_err<=1)却是不带竞品的平级两方比较, NPU 等于拿 fp32
    # 去追双精度参照, 系统性吃亏(§3.4 已警告该形态「红了也不是缺陷」)。
    # 故此处对量化链单独用 _PEER_DTYPE 重算一套统计量。这不是撤销 Promote(§三禁止):
    # out1/out2 的 fp64 真值完整保留, 只有本就无双标杆的 int8 一路落回平级。
    if _work == _PEER_DTYPE:
        xg_q, mean_q, gt_q, bt_q = xg, mean_t, gt, bt
        rstd_q = rstd_t
    else:
        xg_q = xg.to(torch.float32)
        eps_q = float(_attr(kwargs, "eps", 1e-5))
        # 平级档: int8 无双标杆, 参照必须与 NPU **同精度**(fp32)。只同精度、**不同实现** ——
        # 统计量走 golden 自己的补偿两遍法(_group_stat, 两两树求和拓扑), 不照抄内核的 lane
        # 宽度/链长切分: 照抄可让比对恒等通过, 却使"规约拓扑放大抵消误差"这类缺陷不可检出。
        # 该档的参照值是 fp32 地板(纯 fp32 可达), 故红 = 内核统计量未达 fp32 地板。
        mean_qs, var_qs = _group_stat(xg_q)
        mean_q = mean_qs.unsqueeze(-1)
        rstd_q = _df_rsqrt(
            var_qs.unsqueeze(-1), torch.tensor(eps_q, dtype=torch.float32)
        )
        gt_q = gt.to(torch.float32)
        bt_q = bt.to(torch.float32)

    # 归一化用 (x-mean)*rstd: rstd 本就是算子输出之一(定义为 1/sqrt(Var+eps)),
    # 用它乘是公式的自然读法, 且比再除一次 sd 少一次舍入。
    gn_t = ((xg_q - mean_q) * rstd_q).reshape(
        N, C, HW
    )  # 公式: (x-E[x]) * 1/sqrt(Var+eps)
    # 仿射按公式分两步写, 不用 addcmul: 后者走 FMA(乘加只舍入一次, 实测与 fp64 降档逐位
    # 相同), 等于凭空比逐步运算准一档。
    out_t = gn_t * gt_q.reshape(1, C, 1) + bt_q.reshape(1, C, 1)
    if silu:
        # siluOut = out/(1+e^-out), 按公式写成一次除法; 不用 F.silu(它是 out*sigmoid(out),
        # 先取倒数再相乘, 多一次舍入)。
        out_t = out_t / (torch.exp(-out_t) + 1.0)

    qs = _as_work(quant_scale, _PEER_DTYPE).reshape(-1)
    if qs.size == C:
        scale_t = torch.from_numpy(qs).reshape(1, C, 1)  # per-channel
    else:
        scale_t = torch.tensor(float(qs[0]), dtype=out_t.dtype)  # per-tensor
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

    ⚠️ **mean/rstd 这两路的第三方标杆独立性有限**: 都归结到同一条 Welford 递归上, 与
    golden 的差别只在工作精度(此处 fp32 / golden 被 Promote 到 fp64)与执行环境(GPU / CPU),
    算法层面不构成独立证据; 只有 y(int8) 那一路走的是真正不同的代码路径。
    ⚠️ 不要照字面相信"第三方只能用 torch.var_mean": 该函数在 fp32 满量程档会溢出产 nan,
    而 Promote 后的 golden 不会, 两个参照自相矛盾会把 cross_check 的 NaN/Inf 检查堵死
    (详见下方统计量处的说明)。

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
            # 不短路的话统计量在空归约轴上无定义(Welford 递归一步都走不了; 换 torch.var_mean
            # 则返回 NaN), meanOut 会得到 NaN 而非 0, 与 golden 背离
            # (实测 (2,16,2,0,4) G=2: 短路前 [nan]*4, 短路后 [0.0]*4)。
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
        # 第三方标杆**原封不动用 torch 库实现**: fp32 的 var_mean + rsqrt, 与被测同精度
        # —— 这是 cross_check 的 mare/mere/rmse(被测误差/第三方误差 的比值)成立的前提,
        # 不能为了让判据通过而动它。
        #
        # ⚠️ **已知限制: 以下两档的第三方比对不适用**(该档请以两方 close(NPU vs 被 Promote
        # 的 fp64 golden)为准, 实测该腿全绿):
        #   (1) 满量程档(|x| 近 3.4e38): var_mean 内部 Welford delta 递归的中间量 (x-m)
        #       越过 fp32 上界成 inf 后 M2 得 inf*0 = nan(实测 L1_613 1678 组里 177 组),
        #       而真值在该档是**有限**的(6.7e-38 量级), 被测亦然 —— 第三方的 nan 是它自己
        #       fp32 溢出的产物, 不是算子语义。
        #   (2) 单样本(每组 n=1)且输入含 nan/±inf: Welford 的 M2_1 := 0 是算法定义、不引用
        #       x, 故方差恒为 0、rstd = 1/sqrt(eps) 有限(golden 与被测共同遵循的契约);
        #       torch 走两遍法 E[(x-mean)^2] 得 (nan-nan)^2 = nan(实测 L1_145/236/273/672
        #       golden 给 2.002/280/12032/1016, 第三方给 nan)。
        # TTK 的 NaN/Inf 检查是前置硬否决且**只比 target 与 third_party**(实测被测与 golden
        # 非有限模式 1678/1678 全一致仍判 mismatch), 故这两档第三方必然否决被测。
        # 三方腿必须是**独立实现**: 用 torch 自己的 var_mean(ATen 的 Welford —— 与我方内核、
        # 与 golden 的平衡二叉树都是不同代码/不同拓扑), 不转写内核算法。转写会让三方与被测
        # 同拓扑、非有限位置逐位一致, 比值判据恒等成立, 三套实现互相制约的检出能力归零。
        # 代价是满量程档(|gamma| 近 FLT_MAX)下三方与被测的非有限位置可能对不上而被前置
        # 否决 —— 那是内核在该档的非有限语义需要对齐标准实现, 不能靠 golden 照抄来回避。
        var_s, mean_s = torch.var_mean(xg, dim=-1, unbiased=False)
        mean_t = mean_s.unsqueeze(-1)
        rstd_t = (1.0 / torch.sqrt(var_s + eps)).unsqueeze(-1)

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
        # 非有限落位契约: nan -> 0, -inf -> -128, +inf -> +127。
        # 与仓内其他 quant 算子口径一致(norm/add_rms_norm_quant 的 _quantize:
        # `np.nan_to_num(value, nan=0.0, neginf=low, posinf=high)`; norm/rms_norm_quant、
        # norm/rms_norm_quant_v3 同样用 nan_to_num(nan=0); activation/swiglu_group_quant
        # 用 isfinite 显式兜)。原先只有 clamp+to(int8), nan 的落位取决于 torch 的
        # `.to(torch.int8)` 默认行为(未定义), 与内核侧不一致 —— 实测 quantScale≡0 档
        # (L1_rangebound_highpre_433, 0/0=nan)内核给 -128 而此处给 0, 差 128 超 quant 判据。
        q_t = torch.round(out_t / scale_t)
        q_t = torch.nan_to_num(q_t, nan=0.0, neginf=-128.0, posinf=127.0)
        q_t = torch.clamp(q_t, -128.0, 127.0)

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

        返回值只含 4 个输入槽: aclnn 把输出张量与属性一并传进来(实测 10 个), 原样全返回
        会被 TTK 拒(expected 4 input tensor slots) -> INPUT_GEN_FAILURE。切片对
        kernel/geir 是恒等。
        """
        N_IN = 4
        if not arrays:
            return arrays
        head = tuple(arrays[:N_IN])
        name = kwargs.get("testcase_name", "") or ""
        if "_nonfinite" not in name:
            return head
        return (_inject_nonfinite_one(head[0]),) + head[1:]

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


class _AclnnCompose:
    """aclnn 通路的第三方标杆。形参名必须照抄头文件张量名(self/gamma/beta/quantScale),
    服务端按名绑参; kernel/geir 用 def.cpp 名, 不能混用, 故与 `_Compose` 分开写。

    必须是**类**且实例参改名 `_instance`: 头文件首参就叫 `self`, 写成普通函数会走
    `_invoke_function`, 那里一律丢弃名为 self 的入参 -> 主输入丢失、张量前移 -> 服务端
    500 IndexError。类路径走 `_bind(inst.__call__)`, 绑定方法签名里 `self` 是普通形参。
    参照 `activation/hard_swish_grad_v2` 的同名写法。
    """

    def __call__(_instance, self, gamma=None, beta=None, quantScale=None, **kwargs):
        return _Compose()(
            self,
            gamma=gamma,
            beta=beta,
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

    third_party = {"torch": _AclnnCompose}

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
        # aclnn 通路收 torch.Tensor 且本算子 tolerance(quant/stat_rel_err)不触发
        # Promote: bf16 原样送达, _golden_impl 的 np.asarray 对 torch bf16 直接
        # 抛 TypeError(实测 L0_002 GOLDEN_FAILURE), 入口先统一转 numpy
        return _golden_impl(
            _to_np(self),
            _to_np(gammaOptional),
            _to_np(betaOptional),
            _to_np(quantScale),
            **kwargs,
        )


__spec__ = {
    "group_norm_silu_quant": "GroupNormSiluQuantSpec",  # kernel / geir 共用 op_name
    "aclnnGroupNormSiluQuant": "AclnnGroupNormSiluQuantSpec",  # aclnn 按 api_name 绑定
}
__golden__ = {"kernel": {"group_norm_silu_quant": "__golden_group_norm_silu_quant"}}
