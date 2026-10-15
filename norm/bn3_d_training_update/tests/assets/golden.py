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
"""bn3d_training_update golden(竞品算子拼接实现,torch 张量算子)+ 多通路 TestSpec.

- 上库件(随算子工程):`__golden__` / `__input__`(TTK kernel 模式,签名照 def.cpp)。
- 验证件(同文件):`__spec__` —— kernel 与 geir 共用同一注册键与 Spec 类;aclnn/e2e 不存在
  (依据见文件末尾)。
- 计算核只写一遍:`_compute()` torch 进出;KernelSpec.golden 只做 numpy↔torch 容器转换。
- 三方腿是**另一条独立实现路径**(`_Bn3dThirdPartyCompose`),不调 `_compute`、不逐行
  同构:元素级仿射交 torch 原生批量归一化 eval 核,矩按合同定义式独立推出。

golden 采用竞品算子拼接(torch 张量算子:mul/sub/div/sqrt 等),不手写 numpy 纯公式。
计算顺序与算子契约一致:非折叠式 y = (x − batch_mean)·mult + offset(Sub → Mul → Add,
三次舍入),中间精度 fp32,y 回 cast 到 x 的 dtype(fp16/bf16/fp32)。
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

_FORMAT_CH_AXIS = {"NCHW": 1, "NCDHW": 1, "NHWC": 3, "NDHWC": 4}

# ---------------------------------------------------------------------------
# 三方腿共用件:C 轴在 dim1 的规范布局 + 竞品原生批量归一化(eval 形态)。
# 这是三方腿与计算核 `_compute` 之间「不共享实现」的落点:归一化由 torch 内建
# (cudnn/native) 核完成,不是本文件手写的算子链。
# ---------------------------------------------------------------------------
_C1_PERM = {
    "NCHW": (0, 1, 2, 3),
    "NCDHW": (0, 1, 2, 3, 4),
    "NHWC": (0, 3, 1, 2),
    "NDHWC": (0, 4, 1, 2, 3),
}


def _inv_perm(perm):
    """permute 的逆置换(把规范布局还原回下发布局)。"""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def _bn_eval_c1(x_c1, mean, var, weight, bias, eps):
    """竞品原生批量归一化 eval 形态:(x − mean)/sqrt(var + eps)·weight + bias。

    要求 x 的 C 轴已落在 dim1(NCDHW/NHWC 等先按 _C1_PERM 重排)。
    这里刻意不自己拼 (x−μ)·rsqrt(σ²+ε)·w + b 的算子链 —— 交由 torch 内建核
    计算,舍入路径与 golden 的 `_compute` 不同,三方比对才有区分度。
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


__golden__ = {"kernel": {"bn3_d_training_update": "bn3d_training_update_golden"}}
__input__ = {"kernel": {"bn3_d_training_update": "bn3_d_training_update_input_gen"}}

# kernel 与 geir 共用同一个注册键(算子蛇形名)与类;aclnn/e2e 不存在(见文件末尾注释)。
__spec__ = {"bn3_d_training_update": "Bn3dKernelSpec"}


# ---------------------------------------------------------------------------
# 工具:channel 轴解析 —— 纯 format 驱动
# format 在 TTK 场景恒有(input_formats 经 X-Input-Schema 内嵌传给 golden/compose),
# NCHW/NHWC 且空间维等于 C 的歧义 shape 由 format 唯一确定;拿不到 format 直接报错。
# ---------------------------------------------------------------------------
def _resolve_ch_axis(x_shape, C, kwargs):
    fmt = kwargs.get("input_formats", None)
    if fmt is not None and len(fmt) > 0 and fmt[0] in _FORMAT_CH_AXIS:
        return _FORMAT_CH_AXIS[fmt[0]]
    raise ValueError(
        "no input format for channel axis (need input_formats in kwargs): "
        "x.shape=%s C=%s" % (tuple(x_shape), C)
    )


# ---------------------------------------------------------------------------
# 输入生成器:sum/square_sum 由 x 重算,保证与前驱 BN3DTrainingReduce 的语义一致。
# (输入生成属数据准备,非 golden 计算,保持 numpy。)
# ---------------------------------------------------------------------------
def bn3_d_training_update_input_gen(*input_arrays, **kwargs):
    if len(input_arrays) < 7:
        return list(input_arrays)
    x = np.asarray(input_arrays[0])
    C = input_arrays[1].shape[0]
    ch = _resolve_ch_axis(x.shape, C, kwargs)
    x_f32 = x.astype(np.float32)
    axes = tuple(i for i in range(x.ndim) if i != ch)
    sum_arr = x_f32.sum(axis=axes).astype(np.float32)
    sq_arr = (x_f32 * x_f32).sum(axis=axes).astype(np.float32)
    result = list(input_arrays)
    result[1] = sum_arr
    result[2] = sq_arr
    return result


# ---------------------------------------------------------------------------
# 计算核(全算子只写这一份,torch 进出)——算法语义见下。
# ---------------------------------------------------------------------------
def _compute(
    x, sum_t, square_sum, scale, offset, mean, variance, factor, epsilon, ch_axis
):
    """BN3DTrainingUpdate 语义;x 可为 fp16/fp32/bf16,统计量 fp32。"""
    x_dt = x.dtype
    x_w = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
    C = sum_t.shape[0]
    numel = x.numel()
    # 空 tensor 守卫:C=0 时 num 无定义(python 整除零直接抛 ZeroDivisionError)。
    num = (numel // C) if C > 0 else 0

    # 与算子实现同序:乘倒数(num_rec),非除法(除法与倒数乘在 fp32 下差 1ULP)。
    # 空 batch 契约(tiling num_rec=0 同款):num=0 时零统计——bm/bv=0、
    # mean_out=mean·(1-f)、var_out=var·(1-f)。竞品佐证:torch CUDA native 对空
    # 归约域 save_mean=0(非 nan),CPU/拼接原版则直接拒绝/崩溃,无 nan 语义。
    num_rec = (1.0 / num) if num > 0 else 0.0
    batch_mean = sum_t * num_rec
    save_variance = square_sum * num_rec - batch_mean * batch_mean

    inv_std = 1.0 / torch.sqrt(save_variance + epsilon)
    multiplier = scale * inv_std

    stat_shape = [1] * x.dim()
    stat_shape[ch_axis] = C
    y = (x_w - batch_mean.reshape(stat_shape)) * multiplier.reshape(
        stat_shape
    ) + offset.reshape(stat_shape)
    y = y.to(x_dt)

    # Bessel 修正:num==1 时分母为 0 → 无偏 batch variance 置 0(running variance 不更新)。
    scaler = 0.0 if num == 1 else float(num) / float(num - 1)
    unbiased_batch_var = save_variance * scaler

    mean_out = mean * (1.0 - factor) + batch_mean * factor
    variance_out = variance * (1.0 - factor) + unbiased_batch_var * factor

    return [y, mean_out, variance_out, batch_mean, save_variance]


# ---------------------------------------------------------------------------
# kernel/geir Spec:numpy 进出,golden 只做容器转换后调 _compute。
# ---------------------------------------------------------------------------


def _pin_eager_fp32_numerics():
    """把 `torch.compile` 的融合形态钉在「eager fp32 逐步舍入」精度档(禁 FMA 收缩)。

    背景:Inductor 的 Triton 后端默认把 `a * b ± c` 收缩成**单次舍入的 FMA**,
    比 eager fp32 少一次中间乘积舍入 → 竞品被抬到 fp32 以上,三方 ratio 失去
    「同算法同精度档」(「融合 FMA vs 分步乘加」)。本算子内核刻意用三步非融合
    `Reg::Sub + Reg::Mul + Reg::Add`(新实现 `op_kernel/arch35/bn3_d_training_update_common.h`
    的 `VFChannelFirst` / `VFChannelLast` 热循环里是三条独立的 `AscendC::Reg::Sub` +
    `AscendC::Reg::Mul` + `AscendC::Reg::Add`,系数折叠同文件 `FoldRegs` 亦为分步
    `Muls`/`Muls`/`Mul`/`Sub`/`Adds`/`Sqrt`/`Div`),竞品侧必须同档。

    机制(已在交付 GPU host 上静态核实):
      - Inductor 生成 `triton_meta["enable_fp_fusion"] = not config.emulate_precision_casts`
        (torch 2.12 `_inductor/codegen/triton.py:5560`;
         交付 host torch 2.14.0+cu126 `_inductor/codegen/triton.py:6960`);
      - 该值经 `TritonHeuristics._get_compile_options` 透传给 `triton.compile`
        (`_inductor/runtime/triton_heuristics.py:810-811`);
      - 置 True 即令 Triton 的 mul/add 不带 `contract` 标志 → 不产生 FMA。
    只关 fp 融合:**融合图 / 单 kernel 形态不变**,性能腿仍是最优形态;
    精度腿则与 eager fp32 逐位同档。

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


class _Bn3dThirdPartyCompose:
    """三方腿:竞品(torch)独立实现。

    与计算核 `_compute` **不共享任何代码、不逐行同构**,按算法语义另走一条实现路径:
      1. 逐通道矩按**合同定义式**独立推出 —— 真除法 `Σx/num` 与二阶中心矩展开式
         `(Σx² − Σx·μ)/num`,而非 `_compute` 的「乘倒数 + μ·μ 差」口径;
      2. 元素级仿射交给 **torch 原生批量归一化 eval 核**(`_bn_eval_c1`),不再
         手写 `(x − μ)·mult + offset` 的算子链;
      3. running 统计按**合同式**(`lerp`:a + f·(b − a))写,非展开式。
    三者数学语义与契约一致:`y = (x − μ)/sqrt(σ² + ε)·scale + offset`。

    精度档位(「浮点 + 三方」格):
      - 框架固定 **torch**;
      - 入口**按 NPU 的加宽行为同步 cast**:arch35 内核
        `Cast(xf, xb, CAST_NONE)` 只对 f16/bf16 升 f32,fp32 不加宽
        (新实现 `op_kernel/arch35/bn3_d_training_update_common.h` 的
        `LoadAlignedXToFp32` / `LoadUnalignedXToFp32`,随路 `LoadDist::DIST_UNPACK_B16` +
        `AscendC::Reg::Cast`);本类同款;
      - 出口**对应地窄回** f32 → 下发输入 dtype:内核 `Cast(yT, yf, CAST_RINT)`
        (同文件的 `kCastFp32ToB16` 定的是 `RoundMode::CAST_RINT`,经 `StoreAlignedYFromFp32` /
        `StoreUnalignedYFromFp32` 写出,round-half-even),torch `.to()` 同口径;不回停加宽类型,
        否则与走 Promote 的 golden 逼近逐位相等,双标杆塌成单标杆;
      - 类型覆盖:本算子 def 的 x 仅 f16/f32/bf16(xDTypes,
        op_host/bn3_d_training_update_def.cpp:21-25),**无整型 dtype**,
        故无整型分支(整型另有「不提升、无三方」规则,此处不适用)。
      - **算法档位(同算法同精度档)**:`sum_t * batch_mean` 必须是**分步乘 +
        分步减**(各一次 RN 舍入);
        **不得**被收缩成单次舍入的 FMA。
        故 `__call__` 在 `torch.compile` 前调用 `_pin_eager_fp32_numerics()`。

    形态与两条腿的关系(同一 compose 同时驱动精度腿与性能腿,TTK 不透传腿别,
    见 ops-test-kit `executor.py:1104-1126`:mode 只选 `_run_perf` / `_invoke`,
    `compose_kwargs` 仅 attrs + input_formats):
      - 性能腿(竞品必须开融合):保留 `torch.compile(mode="reduce-overhead")`,
        仍是融合图 / 单 kernel 的最优形态,不拿分解实现虚高 G/N;
      - 精度腿:融合图内**禁 FMA 收缩**,逐步舍入与 eager fp32 / 内核同档。
    """

    def __init__(self, **kwargs):
        self.factor = float(kwargs.get("factor", 0.1))
        self.epsilon = float(kwargs.get("epsilon", 1.0e-5))
        # 输入 format(内嵌在 X-Input-Schema 传入):有则 format-aware 定通道轴
        self.input_formats = kwargs.get("input_formats") or None

    def _perm(self, x):
        """format 驱动定「C 轴在 dim1」的重排元组;拿不到 format 直接报错,不猜轴。

        `input_formats` 的序列形态与裸字符串形态都收(与同文件 `_resolve_ch_axis`
        的宽容口径一致),避免下发形态变化时三方腿静默失联。
        """
        fmts = self.input_formats
        if not fmts:
            raise ValueError(
                "compose: no input format for channel axis (need input_formats): "
                "x.shape=%s" % (tuple(x.shape),)
            )
        fmt = str(fmts[0] if isinstance(fmts, (list, tuple)) else fmts).upper()
        perm = _C1_PERM.get(fmt)
        if perm is None or len(perm) != x.dim():
            raise ValueError(
                "compose: unsupported format/rank: format=%r x.shape=%s"
                % (fmt, tuple(x.shape))
            )
        return perm

    def _update_eager(self, x, sum_t, square_sum, scale, offset, mean, variance):
        out_dt = x.dtype
        perm = self._perm(x)
        identity = perm == tuple(range(x.dim()))
        C = int(sum_t.shape[0])
        numel = x.numel()
        # 空 batch 契约(与契约一致):num=0 ⇒ 零统计量,不抛除零。
        num = (numel // C) if C > 0 else 0

        # 入口:按 NPU 的加宽行为同步 cast(内核 Cast(CAST_NONE) 只作用于 f16/bf16)。
        if out_dt in (torch.float16, torch.bfloat16):
            xn = x.to(torch.float32)
        else:
            xn = x

        # 逐通道矩:合同定义式(真除法 + 二阶中心矩展开),非内核的乘倒数口径。
        if num > 0:
            batch_mean = sum_t / num
            save_variance = (square_sum - sum_t * batch_mean) / num
        else:
            batch_mean = torch.zeros_like(sum_t)
            save_variance = torch.zeros_like(square_sum)

        if numel > 0:
            x_c1 = xn if identity else xn.permute(*perm)
            y = _bn_eval_c1(
                x_c1, batch_mean, save_variance, scale, offset, self.epsilon
            )
            if not identity:
                y = y.permute(*_inv_perm(perm))
        else:
            # C==0 的空输入:本算子 infershape 只做 shape 拷贝(不拒零维),tiling 亦支持 C==0 ⇒ 产出空 y。
            y = torch.empty_like(xn)
        # 出口:窄回下发输入 dtype,与内核 CAST_RINT(round-half-even)同口径。
        y = y.to(out_dt)

        # Bessel 修正:num==1 时分母为 0 → 无偏 batch variance 置 0(running variance 不更新)。
        scaler = 0.0 if num == 1 else float(num) / float(num - 1)
        unbiased_batch_var = save_variance * scaler
        # running 统计:合同式(a + f·(b − a) 的 lerp 形态),非展开式。
        mean_out = torch.lerp(mean, batch_mean, self.factor)
        variance_out = torch.lerp(variance, unbiased_batch_var, self.factor)
        return [y, mean_out, variance_out, batch_mean, save_variance]

    def __call__(self, x, sum, square_sum, scale, offset, mean, variance):
        args = (x, sum, square_sum, scale, offset, mean, variance)
        # 竞品形态 = 融合(性能腿要求,不拿分解实现虚高 G/N)且**禁 FMA 收缩**
        # (精度腿要求,与内核三步非融合同档)。旋钮不可用 → 直接走 eager
        # (仍是 fp32 同精度档,只是不融合),绝不放行「带 FMA 的融合形态」。
        if _pin_eager_fp32_numerics():
            try:
                fn = torch.compile(self._update_eager, mode="reduce-overhead")
                return fn(*args)
            except Exception:
                pass
        return self._update_eager(*args)


class Bn3dKernelSpec:
    @staticmethod
    def _attr(kwargs, name, default):
        v = kwargs.get(name)
        if v is None:
            attrs = kwargs.get("attributes")
            if isinstance(attrs, dict):
                v = attrs.get(name)
        return default if v is None else v

    @staticmethod
    def golden(x, sum, square_sum, scale, offset, mean, variance, **kwargs):
        factor = float(Bn3dKernelSpec._attr(kwargs, "factor", 0.1))
        epsilon = float(Bn3dKernelSpec._attr(kwargs, "epsilon", 1.0e-5))
        ch = _resolve_ch_axis(np.asarray(x).shape, np.asarray(sum).shape[0], kwargs)
        outs = _compute(
            _to_torch(x),
            _to_torch(np.asarray(sum)),
            _to_torch(np.asarray(square_sum)),
            _to_torch(np.asarray(scale)),
            _to_torch(np.asarray(offset)),
            _to_torch(np.asarray(mean)),
            _to_torch(np.asarray(variance)),
            factor,
            epsilon,
            ch,
        )
        return [_to_numpy(o) for o in outs]

    # 三方标杆(远端 GPU 执行):竞品原生原语拼接的独立实现(见类 docstring),
    # torch.compile 融合为最优形态(**但融合图内禁 FMA 收缩**,见
    # `_pin_eager_fp32_numerics`),编译失败回落 eager。
    third_party = {"torch": _Bn3dThirdPartyCompose}

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


# ---------------------------------------------------------------------------
# 上库件老入口(签名照 def.cpp):numpy 进出,指向同一计算核,供 TTK kernel 模式直调。
# ---------------------------------------------------------------------------
def bn3d_training_update_golden(
    x,
    sum,
    square_sum,
    scale,
    offset,
    mean,
    variance,
    factor=0.1,
    epsilon=1.0e-5,
    **kwargs,
):
    return Bn3dKernelSpec.golden(
        x,
        sum,
        square_sum,
        scale,
        offset,
        mean,
        variance,
        factor=factor,
        epsilon=epsilon,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 三方 compose(远端 GPU 上执行):类形式,属性喂 __init__、输入喂 __call__;
# 参数名与 proto REG_OP 注册名逐字一致(x, sum, square_sum, scale, offset,
# mean, variance, factor, epsilon)。torch.compile 包一层拿最优形态(融合图内禁 FMA
# 收缩,见 `_pin_eager_fp32_numerics`),失败回落 eager。
# 输出必须 cast 到 NPU 输出 dtype(y 回 x dtype;统计量 fp32),保证 cross_check 同精度对等。
# 与计算核 `_compute` 的关系:两条**独立实现路径**(不共享函数、不逐行同构)——
# golden 腿走 `_compute` 的手写链,三方腿走 torch 原生 BN 核 + 独立推得的矩;
# 这也是三方比对具备公式级/语义级区分度的前提。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 通路支持面:
#   kernel = ✅(op_kernel/arch35 实现)
#   geir   = ✅(op_graph/bn3_d_training_update_proto.h REG_OP)
#   aclnn  = ❌(op_host/CMakeLists.txt ACLNNTYPE aclnn_exclude;无 op_api 实现)
#   e2e    = ❌(canndev 无本算子 aclnn → torch_npu 无绑定可派发)
# aclnn/e2e 不存在 → 不注册对应 __spec__ 条目(不留空壳,TTK 会真的加载空实现)。
# ---------------------------------------------------------------------------
