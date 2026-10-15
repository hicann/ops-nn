#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""bn3_d_training_reduce_grad golden(竞品算子拼接实现,torch 张量算子)+ kernel/geir TestSpec.

- 上库件(随算子工程):`__golden__`(TTK kernel 模式,签名照 def.cpp)。
- 验证件(同文件):`__spec__` —— kernel 与 geir 共用同一注册键与 Spec 类;aclnn/e2e 不存在
  (依据见文件末尾)。
- 计算核只写一遍:`_compute()` torch 进出;KernelSpec.golden 只做 numpy↔torch 容器转换。
- 三方 compose 是**另一条独立实现路径**(`_Bn3dReduceGradCompose`),不调 `_compute`、
  不逐行同构:归一化交 torch 原生 BN eval 核,修正项按合同式另作合并(红线 R3)。

golden 采用竞品算子拼接(torch 张量算子 sub/mul/div/sqrt),不手写 numpy 纯公式(红线 R3:
纯公式常与被测实现犯同一个理解错误,失去交叉验证意义)。

运算序列与 arch35 内核 5 条 VF 链逐步对齐(bn3d_training_reduce_grad_kernel.h):
    s   = sqrt(bv + eps)             ChainS : Adds+Sqrt(epsilon_guard)
    t_a = (x - bm) * ds * inv_num    ChainA1: Sub+Mul+Muls
    t1  = grads - t_a / s            ChainA2: Div+Sub(÷s 在括号内、先于 grads−)
    t2  = t1 - do * inv_num          ChainB : Muls+Sub
    y   = (t2 * sc) / s              ChainC : Mul+Div
f16/bf16 输入升 f32 参与全部中间运算,结果回落原 dtype(kernel CAST_RINT=RN-even,
torch .to(float16/bfloat16) 同口径);num = 除通道轴外所有维的乘积(NCDHW/NDHWC 一致),
num==0 时 inv_num 取 0(零统计量)而非除零崩。

精度档位契约(golden_standard.md §三/§七/§八):
- 本算子落「浮点 + 三方」格:tolerance 三条全 cross_check → TTK 自动开 Promote
  (f16/bf16→f32、f32→f64)并执行 GPU 腿,判据为比值 cross_check;
- golden 入口零 cast、只许向上兜底: _work_dtype / _stat_to_torch 仅把 f16/bf16 抬到 f32,
  f32/f64 **照收下发 dtype** —— Promote 抬到 fp64 后不得砍回(§八#1「撤销 Promote」);
- 出口不 cast,停留 TTK 下发的(已抬档)dtype;
- 两方泛化档请显式传 --compare close;**勿用 close 替代三方档** —— 它会覆盖声明判据、
  连带关掉 Promote 与 GPU 腿,双标杆塌成单标杆(§八#5)。
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

# 通道轴:按声明的 format 严格判定

_FORMAT_CH_AXIS = {"NCDHW": 1, "NDHWC": 4}

# ---------------------------------------------------------------------------
# 三方腿共用件:C 轴在 dim1 的规范布局 + 竞品原生批量归一化(eval 形态)。
# 三方腿据此与计算核 `_compute` 分属两条独立实现路径(红线 R3 / handover §4 铁律
# 「禁照抄被测内核的实现细节」):归一化由 torch 内建 (cudnn/native) 核完成。
# ---------------------------------------------------------------------------
_C1_PERM = {"NCDHW": (0, 1, 2, 3, 4), "NDHWC": (0, 4, 1, 2, 3)}


def _inv_perm(perm):
    """permute 的逆置换(把规范布局还原回下发布局)。"""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def _bn_eval_c1(x_c1, mean, var, weight, bias, eps):
    """竞品原生批量归一化 eval 形态:(x − mean)/sqrt(var + eps)·weight + bias。

    要求 x 的 C 轴已落在 dim1(NDHWC 先按 _C1_PERM 重排)。
    刻意不自己拼 (x−μ)·rsqrt(σ²+ε)·w + b 的算子链 —— 交由 torch 内建核计算,
    与 golden 的 `_compute` 舍入路径不同,三方比对才有区分度。
    """
    return F.batch_norm(
        x_c1,
        mean,
        var,
        weight,
        bias,
        training=False,
        momentum=0.0,
        eps=float(eps),
    )


# CANN proto 口径默认 0.0001(非 PyTorch 的 1e-5,由插件层显式映射)
_DEFAULT_EPS = 1.0e-4


# ---------------------------------------------------------------------------
# numpy(ml_dtypes bf16)↔ torch 位保真转换:torch.from_numpy 不认 ml_dtypes
# bfloat16 ndarray,经 uint16 view 逐位搬运,无任何舍入。
# ---------------------------------------------------------------------------
def _to_torch(a):
    a = np.asarray(a)
    if a.dtype.name == "bfloat16" and _bf16 is not None:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(a)


def _to_numpy(t):
    if t.dtype == torch.bfloat16 and _bf16 is not None:
        return t.view(torch.uint16).numpy().view(_bf16)
    return t.numpy()


__golden__ = {
    "kernel": {"bn3_d_training_reduce_grad": "bn3d_training_reduce_grad_golden"}
}

# kernel 与 geir 共用同一个注册键(算子蛇形名,与 def.cpp opInterface.value 一致)与类;
# aclnn/e2e 不存在(见文件末尾注释)。
__spec__ = {"bn3_d_training_reduce_grad": "Bn3dReduceGradKernelSpec"}


# ---------------------------------------------------------------------------
# 工具:channel 轴解析 —— 以 kwargs['input_formats'] 声明为准(TTK 下发的键名,
# 旧版 golden 误用 'tensor_formats' 收不到,恒走抛错分支)。
# ---------------------------------------------------------------------------
def _resolve_ch_axis(shape, c, kwargs):
    if len(shape) != 5:
        # 检查器用通用 2D/低维输入探测 golden 时, 本算子语义要求 5D 才会到这里;
        # 抛清晰错误以区分「探测输入非法」与「format 不支持」。
        raise ValueError(
            "BN3DTrainingReduceGrad grads 必须 5D(探测输入 %r 非法); "
            "golden 只处理合法 5D 用例" % (tuple(shape),)
        )
    fmts = kwargs.get("input_formats")
    fmt = None
    if fmts:
        f0 = fmts[0] if isinstance(fmts, (list, tuple)) else fmts
        fmt = str(f0).upper()
    if fmt in _FORMAT_CH_AXIS:
        return _FORMAT_CH_AXIS[fmt]
    raise ValueError(
        "unsupported tensor format for 5D grads: %r (must be NCDHW/NDHWC, "
        "shape %s, C=%s)" % (fmt, tuple(shape), c)
    )


def _work_dtype(*tensors):
    """计算档位:只向上兜底 —— f16/bf16 抬 f32;任一输入是 f64(Promote 抬档)则全 f64。"""
    if any(t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


def _stat_to_torch(value):
    """stat 输入入口:零 cast —— 仅 f16/bf16 向上兜到 f32,其余照收下发 dtype。

    不得写死 np.float32:Promote 场景下 5 个参数张量会被 TTK 抬到 fp64,
    写死 fp32 等于把它们砍回 fp32 = 撤销 Promote(golden_standard.md §八#1)。
    """
    tensor = _to_torch(value)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(torch.float32)
    return tensor


# ---------------------------------------------------------------------------
# 计算核(全算子只写这一份,torch 进出)——运算序列照 arch35 内核 5 条 VF 链。
# ---------------------------------------------------------------------------
def _compute(
    grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance, epsilon, ch
):
    """BN3DTrainingReduceGrad 语义;grads/x 可为 fp16/fp32/bf16,5 个参数张量 fp32。

    y = (grads - ds*(x-bm)/(num*s) - do/num) * sc/s, s = sqrt(bv + epsilon);
    num = 除通道轴(ch)外所有维的乘积;返回 [y](回落 grads.dtype)。
    """
    out_dt = grads.dtype  # 回落目标:Promote 下=抬档精度(近真值),原生=原 dtype
    work = _work_dtype(
        grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
    )
    g = grads.to(work)
    xx = x.to(work)
    ds = diff_scale.to(work)
    do = diff_offset.to(work)
    sc = scale.to(work)
    bm = batch_mean.to(work)
    bv = batch_variance.to(work)

    c = int(ds.shape[0])
    view = [1] * grads.dim()
    view[ch] = c
    num = 1
    for d, s_ in enumerate(grads.shape):
        if d != ch:
            num *= int(s_)
    # 空归约域守卫:num==0 时 1.0/num 直接抛 ZeroDivisionError(记 GOLDEN_FAILURE)。
    # 同族 bn3_d_training_update 的 golden 同款写法(num>0 ? 1/num : 0.0,其 :106,112);
    # num==0 ⇒ 零统计量(t_a=0、t2=t1),与同族「空 batch 零统计」契约一致。
    # 现网不可达:infershape 已直接拒 0 维(HasZeroDim(grads_shape)||HasZeroDim(x_shape)
    # → GRAPH_FAILED,op_host/bn3_d_training_reduce_grad_infershape.cpp:91-92),属健壮性加固。
    inv_num = (
        (1.0 / num) if num > 0 else 0.0
    )  # python 标量:·inv_num 替代 ÷num,与 kernel Muls 同口径

    s = torch.sqrt(bv + float(epsilon)).reshape(view)  # ChainS
    t_a = (xx - bm.reshape(view)) * ds.reshape(view) * inv_num  # ChainA1
    t1 = g - t_a / s  # ChainA2
    t2 = t1 - do.reshape(view) * inv_num  # ChainB
    y = (t2 * sc.reshape(view)) / s  # ChainC
    return [y.to(out_dt)]


def _pin_eager_fp32_numerics():
    """把 `torch.compile` 的融合形态钉在「eager fp32 逐步舍入」精度档(禁 FMA 收缩)。

    背景:Inductor 的 Triton 后端默认把 `a * b ± c` 收缩成**单次舍入的 FMA**,
    比 eager fp32 少一次中间乘积舍入 → 竞品被抬到 fp32 以上,三方 ratio 失去
    「同算法同精度档」(verification.md §7.2「融合 FMA vs 分步乘加」;
    golden_standard.md §一/§五)。本算子内核把每一步都写成**独立的** AscendC Reg 原语
    (`Reg::Mul` / `Reg::Add` / `Reg::Select`,各自一次 RN 舍入;全文件无
    `Fma` / `MulAdd`),竞品侧必须同档。

    机制(已在交付 GPU host 上静态核实):
      - Inductor 生成 `triton_meta["enable_fp_fusion"] = not config.emulate_precision_casts`
        (torch 2.12 `_inductor/codegen/triton.py:5560`;
         交付 host torch 2.14.0+cu126 `_inductor/codegen/triton.py:6960`);
      - 该值经 `TritonHeuristics._get_compile_options` 透传给 `triton.compile`
        (`_inductor/runtime/triton_heuristics.py:810-811`);
      - 置 True 即令 Triton 的 mul/add 不带 `contract` 标志 → 不产生 FMA。
    只关 fp 融合:**融合图 / 单 kernel 形态不变**,性能腿仍是最优形态(§7.1);
    精度腿则与 eager fp32 逐位同档(§7.2)。

    该配置进入 Inductor FX graph 缓存键(`codecache.py:1014`
    `FxGraphCacheKey.inductor_config = config.save_config_portable(...)`),
    故不会复用此前「带 FMA」的旧编译产物。
    ⚠️ 该配置在**编译时刻**读取,必须在进程内**首次编译之前**置位 —— 故本函数在
    `__call__` 里先调用、再 `torch.compile`(实测:A100 同进程内若先按旧形态编译过一次,
    后续再置位不会重编,仍带 FMA)。

    返回旋钮是否已生效;torch 版本无此旋钮 → False,调用方回落 eager(仍 fp32 同档)。
    """
    try:
        from torch._inductor import config as _inductor_config
    except Exception:
        return False
    try:
        if not getattr(_inductor_config, "emulate_precision_casts", False):
            _inductor_config.emulate_precision_casts = True
    except Exception:
        return False
    return bool(getattr(_inductor_config, "emulate_precision_casts", False))


# ---------------------------------------------------------------------------
# 三方 compose(远端 XPU/GPU server 上执行):属性喂 __init__、输入喂 __call__;
# 参数名与 def.cpp / proto REG_OP 注册名逐字一致(grads, x, diff_scale,
# diff_offset, scale, batch_mean, batch_variance, epsilon)。torch.compile 包一层
# 拿最优形态(竞品必须开融合,分解实现会虚高 G/N;**融合图内禁 FMA 收缩**,见
# `_pin_eager_fp32_numerics`),失败回落 eager。
# 输出必须 cast 回 NPU 输出 dtype(y 回 grads dtype),保证三方同精度对等。
#
# 独立性(红线 R3):本类**不调用** `_compute`,也不逐行复刻内核 5 条 VF 链 ——
# 归一化由 torch 原生 BN(eval)核完成,与 golden 腿是两条独立实现路径。
# (旧版此处 `return _compute(...)`,与 golden 共用同一函数,三方比对无区分度。)
#
# 通道轴消歧:`input_formats` 经 X-Input-Schema 内嵌下发(见 __init__),本类**纯 format
# 驱动**判通道轴并有 `_C1_PERM` 定重排,不做 shape 启发式;拿不到 format 直接报错。
# 注: 必须定义在 Spec 之前,供 third_party 以类对象引用(xpu-server 只收
# module.attr 点路径,字符串类名会被拒 400 —— 实测教训)。
# ---------------------------------------------------------------------------
class _Bn3dReduceGradCompose:
    """三方腿:竞品(torch)独立实现 —— **不调 `_compute`、不逐行同构**。

    合同语义(K = grads,s = sqrt(σ²+ε)):
        y = K·sc/s − diff_scale·(x−μ)·sc/(num·s²) − diff_offset·sc/(num·s)

    三方腿按**与内核 5 条 VF 链不同的分解**实现同一数学式:
        x̂ = (x − μ)/sqrt(σ² + ε)          ← 由 torch 原生 BN(eval)核给出,非手写链
        y = ( K − (diff_scale·x̂ + diff_offset)/num ) · (sc/s)
    两处恒等:把 x̂ 展开即 (x−μ)/s,与内核 `t_a/s = (x−μ)·ds·inv_num/s` 同值;
    `(sc/s)` 逐通道先约再乘,与内核 `(t2·sc)/s` 的乘后除同值。差别在**实现路径**:
    归一化原语、合并次序、逐通道系数的组织方式都与 `_compute` 不同,故三方腿能
    捕获公式级/语义级错误(红线 R3)。

    精度档位(照 golden_standard.md §一/§五「浮点 + 三方」格):
      - 框架固定 **torch**;
      - 入口**按 NPU 的加宽行为同步 cast**:arch35 `CastUpVF` 只把 f16/bf16 升 f32
        (bn3d_training_reduce_grad_kernel.h:517-531),fp32 不加宽;5 个参数张量在
        内核侧恒以 `GlobalTensor<float>` 读入(同文件 :738),故一律对齐 f32;
      - 出口**对应地窄回** f32 → 下发 grads dtype:内核 `CastDownVF` 用 `CAST_RINT`
        (round-half-even,同文件 :34-39 / :538-550),torch `.to()` 同口径;
      - 类型覆盖:def 的 `DT_*` 仅 f16/f32/bf16
        (op_host/bn3_d_training_reduce_grad_def.cpp:26-65),**无整型 dtype**,
        故无整型分支。
      - **算法档位(同算法同精度档,§7.2)**:`ds * normed + do` 与
        `g1 - corr` 都必须是**分步乘 + 分步加/减**(各一次 RN 舍入),与内核逐条
        `Reg::Mul` / `Reg::Add` / `Reg::Select` 原语一致
        (bn3d_training_reduce_grad_kernel.h:351/397 等;全文件无 `Fma`/`MulAdd`),
        **不得**被收缩成单次舍入的 FMA。故 `__call__` 在 `torch.compile` 前调用
        `_pin_eager_fp32_numerics()`。

    形态与两条腿的关系(同一 compose 同时驱动精度腿与性能腿,TTK 不透传腿别,
    见 ops-test-kit `executor.py:1104-1126`:mode 只选 `_run_perf` / `_invoke`,
    `compose_kwargs` 仅 attrs + input_formats):
      - 性能腿(§7.1「竞品必须开融合」):保留 `torch.compile(mode="reduce-overhead")`,
        仍是融合图 / 单 kernel 的最优形态,不拿分解实现虚高 G/N;
      - 精度腿(§7.2):融合图内**禁 FMA 收缩**,逐步舍入与 eager fp32 / 内核同档。
    """

    def __init__(self, **kwargs):
        self.epsilon = float(kwargs.get("epsilon", _DEFAULT_EPS))
        # TTK 新功能: input_formats 经 X-Input-Schema 内嵌传给 compose __init__
        # (ops-test-kit executor.py compose_kwargs["input_formats"])。
        fmts = kwargs.get("input_formats") or None
        # 序列形态与裸字符串形态都收(与同文件 `_resolve_ch_axis` 同款宽容口径),
        # 避免下发形态变化时三方腿静默失联。
        if fmts:
            self.input_formats = str(
                fmts[0] if isinstance(fmts, (list, tuple)) else fmts
            ).upper()
        else:
            self.input_formats = None

    def _perm(self, t):
        # 纯 format 驱动判通道轴(NCDHW→dim1 / NDHWC→dim4),【无 shape 启发式】;
        # 拿不到 format 直接报错, 不做形状猜测(全维度相等歧义由 format 唯一确定)。
        perm = _C1_PERM.get(self.input_formats)
        if perm is None or len(perm) != t.dim():
            raise ValueError(
                "compose: no input format for channel axis (need input_formats): "
                "input_formats=%r grads.shape=%s" % (self.input_formats, tuple(t.shape))
            )
        return perm

    def _rg_eager(
        self, grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
    ):
        out_dt = grads.dtype
        perm = self._perm(grads)
        identity = perm == tuple(range(grads.dim()))
        dim = grads.dim()
        ch = perm[1]  # C 在下发布局里的位置(perm 把 C 送到 dim1,回推即 perm[1])
        C = int(diff_scale.shape[0])

        # 入口:按 NPU 的加宽行为同步 cast(f16/bf16 → f32;fp32 不加宽)。
        if out_dt in (torch.float16, torch.bfloat16):
            g = grads.to(torch.float32)
            xx = x.to(torch.float32)
        else:
            g = grads
            xx = x
        # 5 个参数张量内核侧恒 f32(GlobalTensor<float>),同口径对齐。
        ds = diff_scale.to(torch.float32)
        do = diff_offset.to(torch.float32)
        sc = scale.to(torch.float32)
        bm = batch_mean.to(torch.float32)
        bv = batch_variance.to(torch.float32)

        # 空归约域守卫:num==0 ⇒ 零统计量(与 golden 的 num>0 ? 1/num : 0.0 同款)。
        num = 1
        for d in range(dim):
            if d != ch:
                num *= int(grads.shape[d])
        inv_num = (1.0 / num) if num > 0 else 0.0

        # 全程在「C 轴在 dim1」的规范布局上算,逐通道量按 [1,C,1,...] 广播。
        g1 = g if identity else g.permute(*perm)
        if grads.numel() == 0:
            # 空输入保底(infershape 已拒 0 维):无逐元素结果可算。
            y1 = torch.empty_like(g1)
        else:
            x1 = xx if identity else xx.permute(*perm)
            # 归一化输入 x̂:(x − μ)/sqrt(σ² + ε),竞品原生 BN(eval)核,无仿射。
            normed = _bn_eval_c1(x1, bm, bv, None, None, self.epsilon)
            # 规范布局里 C 落在 dim1,逐通道量按 [1, C, 1, ...] 广播。
            shape = [1] * dim
            shape[1] = C
            # 反标准差:1/s 的除法口径(与内核各链的 Div 同族)。
            s = torch.sqrt(bv + float(self.epsilon))
            # (diff_scale·x̂ + diff_offset)/num 先合并,再乘 sc/s 主项。
            corr = (ds.view(shape) * normed + do.view(shape)) * inv_num
            y1 = (g1 - corr) * (sc.view(shape) / s.view(shape))

        y = y1 if identity else y1.permute(*_inv_perm(perm))
        # 出口:窄回下发 grads dtype,与内核 CAST_RINT(round-half-even)同口径。
        return [y.to(out_dt)]

    def __call__(
        self, grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
    ):
        args = (grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance)
        # 竞品形态 = 融合(§7.1 性能腿要求,不拿分解实现虚高 G/N)且**禁 FMA 收缩**
        # (§7.2 精度腿要求,与内核逐条 Reg 原语同档)。旋钮不可用 → 直接走 eager
        # (仍是 fp32 同精度档,只是不融合),绝不放行「带 FMA 的融合形态」。
        if _pin_eager_fp32_numerics():
            try:
                fn = torch.compile(self._rg_eager, mode="reduce-overhead")
                return fn(*args)
            except Exception:
                pass
        return self._rg_eager(*args)


# ---------------------------------------------------------------------------
# kernel/geir Spec:numpy 进出,golden 只做容器转换后调 _compute。
# ---------------------------------------------------------------------------
class Bn3dReduceGradKernelSpec:
    @staticmethod
    def _attr(kwargs, name, default):
        v = kwargs.get(name)
        if v is None:
            attrs = kwargs.get("attributes")
            if isinstance(attrs, dict):
                v = attrs.get(name)
        return default if v is None else v

    @staticmethod
    def golden(
        grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance, **kwargs
    ):
        epsilon = float(Bn3dReduceGradKernelSpec._attr(kwargs, "epsilon", _DEFAULT_EPS))
        ch = _resolve_ch_axis(
            np.asarray(grads).shape, np.asarray(diff_scale).shape[0], kwargs
        )
        outs = _compute(
            _to_torch(grads),
            _to_torch(x),
            _stat_to_torch(diff_scale),
            _stat_to_torch(diff_offset),
            _stat_to_torch(scale),
            _stat_to_torch(batch_mean),
            _stat_to_torch(batch_variance),
            epsilon,
            ch,
        )
        return [_to_numpy(o) for o in outs]

    # 三方标杆(远端 XPU server 执行):竞品原生原语拼接的独立实现(见类 docstring),
    # torch.compile 拿最优形态(**但融合图内禁 FMA 收缩**,见 `_pin_eager_fp32_numerics`)。
    # 类对象引用(xpu-server 要求 module.attr 点路径;字符串类名会 400 拒收)。
    third_party = {"torch": _Bn3dReduceGradCompose}

    # tolerance 是【未显式传 --compare 时】的默认路由:
    #   - 三方精度: 显式 --compare cross_check(默认即 cross_check, 与 tolerance 一致)
    #   - 泛化/性能/invoke-path(非三方): 必须显式 --compare close(联合 rtol+atol,
    #     避免 stat_rel_err 对近零元素假红), 否则会误路由到 cross_check
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


# ---------------------------------------------------------------------------
# 上库件老入口(签名照 def.cpp):numpy 进出,指向同一计算核,供 TTK kernel 模式直调。
# ---------------------------------------------------------------------------
def bn3d_training_reduce_grad_golden(
    grads,
    x,
    diff_scale,
    diff_offset,
    scale,
    batch_mean,
    batch_variance,
    epsilon=_DEFAULT_EPS,
    **kwargs,
):
    return Bn3dReduceGradKernelSpec.golden(
        grads,
        x,
        diff_scale,
        diff_offset,
        scale,
        batch_mean,
        batch_variance,
        epsilon=epsilon,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 通路支持面:
#   kernel = ✅(op_kernel/arch35 实现,RANK=4/8 双 TilingKey 分支)
#   geir   = ✅(op_graph/bn3_d_training_reduce_grad_proto.h REG_OP)
#   aclnn  = ❌(本算子不交付 aclnn 通路——按需求方口径舍弃;旧 golden 的
#              aclnnBN3DTrainingReduceGrad 注册条目与 TestSpec 已删,不留空壳)
#   e2e    = ❌(无 aclnn 交付 → torch_npu 无绑定可派发)
# aclnn/e2e 不存在 → 不注册对应 __spec__ 条目(留空壳 TTK 会真的加载到空实现)。
# ---------------------------------------------------------------------------
