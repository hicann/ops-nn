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

"""TTK custom golden for ScatterDiv (kernel mode).

计算公式 (docs/aclnnScatterDiv.md):
    varRef[indices[i], ...] = varRef[indices[i], ...] / updates[i, ...]
多个 updates 命中同一切片时连除（顺序无关，与 TF scatter_div 一致）。
越界索引 (idx < 0 或 idx >= var.shape[0]) 跳过 (scatter_reduce_common_simt.h:112)。
整型 dtype (int32/int8/uint8) 走 C++ 整数除法（向零截断）。
in-place: 输出 var 与输入 var 同缓冲（output_inplace_indexes=(0,)）。

实现说明: 计算由 numpy/python 公式改为 torch 竞品算子拼接（index_select + torch.div +
index_copy_，整型走 rounding_mode="trunc" 即 C++ 向零截断）——纯 numpy 公式实现与被测
kernel 易犯同类错误，会掩盖 kernel 精度短板。numpy 仅保留 I/O 与 dtype 转换。
除法不满足交换律（a/b1/b2 != a/(b1*b2)，逐位不同），故仍按索引顺序逐条下发，
不能折叠成一次 index_reduce；语义（连除顺序、越界跳过、工作 dtype、返回结构）未变。
"""

import numpy as np
import torch

import inspect as _kf_inspect

try:
    from ml_dtypes import bfloat16 as _KF_BF16
except ImportError:
    _KF_BF16 = None

_INT_DTYPES = {"int32", "int8", "uint8"}


_CHAIN_SEQ_CELLS = 1 << 28  # 单块 seq 槽位上限(~1GiB@fp32); 峰值与用例规模解耦


def _chain_row_blocks(row_starts, counts, slice_size, cap_cells=None):
    """按累计条目数切连续行块, 使每块 seq 槽位 (条目数+行数) * slice_size <= cap_cells。

    不排序、不分桶: 条目在 order 里按行连续, 故"连续行块"即"连续条目区间", 用一次
    searchsorted 定右端即可。热点行(单行吃百万条目)自成一块, 冷行成批, 两种分布都有界。
    """
    nrows = counts.shape[0]
    if cap_cells is None:  # 取调用时的全局值, 便于用例把块调小来压分块路径
        cap_cells = _CHAIN_SEQ_CELLS
    per = max(2, int(cap_cells // max(slice_size, 1)))
    r0 = 0
    while r0 < nrows:
        # 槽位 = 条目 + 行, 各给一半预算, 两者同时有界
        by_rows = r0 + per // 2
        by_ents = (
            int(np.searchsorted(row_starts, row_starts[r0] + per // 2, side="right"))
            - 1
        )
        r1 = min(nrows, by_rows, max(by_ents, r0 + 1))
        yield r0, r1
        r0 = r1


def _chain_seq_layout(counts, row_starts, r0, r1):
    """返回该行块的 (条目数, 行数, 段起点 seg, updates 槽位 upd_pos)。

    seq 逐行拼接 [var行 | 按稳定序的 updates], 无填充; 行 m 的 var 槽位 = seg[m-r0],
    其第 t 个 update 槽位 = (条目局部下标) + (m-r0) + 1。
    """
    a, b = int(row_starts[r0]), int(row_starts[r1])
    cb = counts[r0:r1]
    nb, rb = b - a, r1 - r0
    seg = np.empty(rb, dtype=np.int64)
    seg[0] = 0
    np.cumsum(cb[:-1] + 1, out=seg[1:])
    upd_pos = np.arange(nb, dtype=np.int64)
    upd_pos += np.repeat(np.arange(rb, dtype=np.int64), cb)
    upd_pos += 1
    return a, b, nb, rb, seg, upd_pos


def _chain_div_seq(work_np, upd_np, order, counts, row_starts):
    """fp 除链的序无关真值: 链 = var / Πu, 在 log2 域求值。

    为什么不用顺序除法链: ttk 的比对以本函数为真值, 而 kernel 的排序折叠(合并排序)对
    同段内更新**不稳定**, 侧序 ≠ 位置序; f32/f64 的顺序除法链在溢出/下溢边界是序敏感的
    (实测 ScatterDiv_L1_033: 位置序 f64 链 inf, 另序 0), 拿"某种顺序的中间量伪影"当真值
    会制造不可收敛的分歧。数学真值 var/Πu 本身序无关, log2 域求值天然无中间溢出:
      exp_sum = Σ log2|u| (add.reduceat, 交换律); sign = Πsign(u) (±1 精确乘);
      值 = ldexp(var × sign, -exp_sum), 指数超界由 ldexp 归 ±inf/±0(IEEE 语义)。
    除 0: exp_sum=-inf 时段内确有 u==0, 按 IEEE var/±0 = ±inf(var=0 时 NaN)单独处理。
    f64 精度: 5600 项 log 求和误差 ~1e-12 相对指数, 折算值误差远低于任何容差档。
    """
    slice_size = work_np.shape[1]
    # 执行域零 cast: 完全随 ttk Promote 下发域(fp32 var->f64, fp16 var->f32), 不自行上浮。
    w64 = work_np
    u64 = upd_np
    for r0, r1 in _chain_row_blocks(row_starts, counts, slice_size):
        a, b, nb, rb, seg, upd_pos = _chain_seq_layout(counts, row_starts, r0, r1)
        den = u64[order[a:b]]
        head = w64[r0:r1]
        # 符号与绝对值分解(u=0 的 log2(|0|)=-inf 会污染指数和, 单独走 IEEE 支路)
        # 符号必须按 IEEE signbit: -0.0 的 den<0 为 False 会丢掉"除以负零"的符号
        # (实测 L0_001: var/(-0.0) 应为 -inf, den<0 版给 +inf)。
        sgn = np.where(np.signbit(den), np.float64(-1.0), np.float64(1.0))
        absden = np.abs(den)
        ustart = (row_starts[r0:r1] - row_starts[r0]).astype(np.int64)
        q_sgn = np.multiply.reduceat(sgn.reshape(-1, slice_size), ustart, axis=0)
        d2 = den.reshape(-1, slice_size)
        # 必须是 add 归约: subtract.reduceat 给的是"首项减其余", 单个零会按位置给 ±1,
        # 两个零还会相消成 0 被误判成无零段。
        zero_cnt = np.add.reduceat((d2 == 0).astype(np.float64), ustart, axis=0)
        nan_cnt = np.add.reduceat(np.isnan(d2).astype(np.float64), ustart, axis=0)
        inf_cnt = np.add.reduceat(np.isinf(d2).astype(np.float64), ustart, axis=0)
        exp_sum = np.add.reduceat(
            np.log2(
                absden.reshape(-1, slice_size),
                out=np.zeros_like(absden.reshape(-1, slice_size)),
                where=(absden.reshape(-1, slice_size) != 0),
            ),
            ustart,
            axis=0,
        )
        res = np.empty((rb, slice_size), dtype=w64.dtype)
        nz = (zero_cnt == 0) & (nan_cnt == 0) & (inf_cnt == 0)
        # 与 kernel 的缩放域同构: 尾数 x 整数指数分开累, 中间量恒在 [0.5,2) 不会饱和。
        # (ldexp 单用会截掉 Σlog2 的小数位丢 2^frac; exp2 单用在 |Σ|>1024 时中间就溢出。)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            e = -exp_sum[nz]
            ei = np.floor(e)
            man, hex_ = np.frexp((head * q_sgn)[nz])
            res[nz] = np.ldexp(
                man * np.exp2(e - ei),
                np.clip(ei + hex_, -2100.0, 2100.0).astype(np.int32),
            )
        # 特殊段按 IEEE 定类: 0 与 inf 同段(或 0/0、inf/inf)给 NaN; 仅含 0 给 ±inf; 仅含 inf 给 ±0。
        zmask = ~nz
        if zmask.any():
            hz = head[zmask]
            zc = zero_cnt[zmask]
            ic = inf_cnt[zmask]
            sgn_h = (
                np.where(np.signbit(hz), np.float64(-1.0), np.float64(1.0))
                * q_sgn[zmask]
            )
            nan_seg = (
                (nan_cnt[zmask] > 0)
                | np.isnan(hz)
                | ((zc > 0) & ((ic > 0) | (hz == 0)))
                | ((ic > 0) & np.isinf(hz))
            )
            res[zmask] = np.where(
                nan_seg,
                np.float64(np.nan),
                np.where(zc > 0, sgn_h * np.float64(np.inf), sgn_h * np.float64(0.0)),
            )
        w64[r0:r1] = res
    work_np[...] = w64.astype(work_np.dtype, copy=False)


def _chain_intdiv_seq(work_np, upd_np, order, counts, row_starts):
    """整型链: torch 原生宽度的分层截断除 —— 每步在内核 AccT 域回绕, 与 kernel 逐位一致。

    语义依据: ttk 对整型不 Promote(降级两方 + binary_equal 逐位判据), 真值 = 内核 AccT
    域上逐步 C++ 截断除(INT_MIN÷-1 回绕为 INT_MIN, 之后 ÷-2 得 +2^30 ...)。
    不做任何宽化: int64 域的除法链不回绕, INT_MIN÷-1 之后符号全错 —— 实测 ScatterDiv_
    L1_133(var 全 INT32_MIN, 除数 -2/-1) 有 253,824/6,506,501 个元素与 kernel 不一致;
    早期版本在 int32 域取 abs 更是直接溢出(仍为负)。
    仓内范式: scatter_mul golden 委托 torch index_reduce_("prod") 拿到原生回绕; div 无
    对应竞品归约, 故用 torch 分层向量化复刻(_int_trunc_div: trunc + -1 走 neg 复刻回绕
    + 除 0 保原值)。分层 = 按"行内第 t 次作用"逐层批量, 层内全为独立行可并行。
    深链热点行的层循环代价按 b026 先例记录为 golden 限制。
    """
    slice_size = work_np.shape[1]
    for r0, r1 in _chain_row_blocks(row_starts, counts, slice_size):
        a, b = int(row_starts[r0]), int(row_starts[r1])
        cnt = counts[r0:r1]
        starts = (row_starts[r0:r1] - row_starts[r0]).astype(
            np.int64
        )  # 行首在块内的条目偏移
        depth = int(cnt.max()) if cnt.size else 0
        wt = torch.from_numpy(work_np[r0:r1])  # 共享内存, 原地写回
        upd_blk = upd_np[order[a:b]]
        blk = work_np[r0:r1]
        nz = (blk != 0).any(axis=1)  # 行是否还有非零元素
        for t in range(depth):
            # 第 t 层下标有闭式(行首偏移 + t), 避免每层全扫条目的 O(depth x 条目数)。
            # 0 在截断除下是吸收态(含除 0 保原值), 故全零行不必再走后续层。
            alive = np.flatnonzero((cnt > t) & nz)
            if alive.size == 0:
                break
            sel = starts[alive] + t
            rows = torch.from_numpy(alive)
            wt[rows] = _int_trunc_div(
                wt[rows], torch.from_numpy(np.ascontiguousarray(upd_blk[sel]))
            )
            nz[alive] = (blk[alive] != 0).any(axis=1)
        work_np[r0:r1] = wt.numpy()


def _int_trunc_div(row, denom):
    """整型截断除(向零), 除数为 0 保持原值。全程 torch, 停留在入参 dtype。

    除数为 -1 时改用 neg: torch 的原生整型除法遇 INT_MIN/-1 会触发 SIGFPE 硬件陷阱,
    而回绕语义下 x / -1 恒等于 -x(INT_MIN 处 neg 同样回绕为 INT_MIN), 逐位等价。
    """
    nz = denom != 0
    neg = denom == -1
    safe = torch.where(nz & ~neg, denom, torch.ones_like(denom))
    q = torch.div(row, safe, rounding_mode="trunc")
    q = torch.where(neg, torch.neg(row), q)
    return torch.where(nz, q, row)


def __golden_scatter_div(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out_dtypes = kwargs.get("output_dtypes", None)
    if out_dtypes:
        out_dt_str = out_dtypes[0]
    else:
        out_dt_str = str(var.dtype)
    is_int = out_dt_str in _INT_DTYPES

    var_first = var.shape[0]
    slice_shape = var.shape[1:]
    slice_size = int(np.prod(slice_shape)) if slice_shape else 1
    if slice_size == 0:
        # var 尾维为 0(空切片): 无元素可作用, 原值输出。与 scatter_min 的同名守护一致。
        # 少了这条, updates.reshape(-1, 0) 会因 numpy 无法推断 -1 而抛
        # "cannot reshape array of size 0 into shape (0)" —— golden 先于算子挂掉。
        return [var]  # 出口不 cast, TTK 负责

    # 计算 dtype 一律**跟随 NPU 的 AccT**, 唯一例外是"浮点 + 三方": 那时 TTK 已按
    # golden_mode=Promote 把入参抬档下发(fp16/bf16->fp32、fp32->fp64), golden 零 cast 直接算
    # 即为高精度真值; 自行 cast 会撤销 Promote, 真值塌到与三方腿同精度, 三比值分母被夹,
    # RMSE 比值假红。其余情形(整型恒是两方; 浮点在两方泛化下)TTK 不提升, 必须由 golden 自己
    # 复刻内核: AccT 对 fp16/bf16 是 float、对 int8/uint8 是 int32、对 fp32/int32 原生, 出口
    # 再复刻 NarrowStore 窄回。判 Promote 用 TTK 下发的 golden_mode, 不靠 dtype 猜。
    promoted = kwargs.get("golden_mode") == "Promote"
    sub_int = np.issubdtype(var.dtype, np.integer) and var.dtype.itemsize < 4
    narrow_fp = (not promoted) and var.dtype == np.float16
    need_narrow = sub_int or narrow_fp
    acc = torch.int32 if sub_int else (torch.float32 if narrow_fp else None)

    # 注(11-10): 此处不做任何 cast —— Promote 下 ttk 已把入参抬档(f32->f64)下发, golden
    # 在收到的域里直接算即为高精度真值(用户铁律, L0_014 先例)。曾试过"fp 链降回 f32 域算"
    # 来对齐 kernel 的粘性 inf 语义, 属于人为改真值域, 已回退; f64 真值与 f32 kernel 在
    # 溢出形态(var 全 ±FLT_MAX)下的 inf 语义鸿沟按 golden 限制如实记录(见 b026 先例)。

    # 只把"被索引命中的行"搬进 work: scatter 语义下未命中的行原值输出, 故无需为整个 var
    # 开 acc 缓冲。var 首维支持到 2^30 以上(两趟基数排序), 全尺寸 work 在 dim0=2^30、
    # int8 入参时(acc=int32)就要 4.3GB + 原拷贝 1GB, dim0=2^32 时 17GB —— golden 会先于
    # 算子 OOM, 宽档用例一例也验不了。
    # numpy 在此只做索引转换与行号重标号(规范: numpy 仅作输入输出辅助), 除法仍在 torch。
    out = var.copy()
    idx_flat = indices.reshape(-1).astype(np.int64)
    n = idx_flat.shape[0]
    upd_2d = updates.reshape(-1, slice_size)

    # 分层向量化: 串行依赖只在同一行内部, 按"本行第几次作用"分层后层内可批量相除。
    # stable 排序保证同行内次序不变, 与逐条相除逐位等价。
    if n and var_first:
        in_range = (idx_flat >= 0) & (idx_flat < var_first)
        # 全界内时 ix/src 都是恒等映射: 不物化这两个 k 级 int64 数组(2^31 件上各 17GB,
        # bound_034 的索引范围恰为 0..varFirst-1, 正属此类)。
        if bool(in_range.all()):
            ix, src = idx_flat, None
            del in_range
        else:
            ix = idx_flat[in_range]
            src = np.nonzero(in_range)[0]
        k = ix.shape[0]
        if k:
            # O(N) 计数定位替代 unique+多重 argsort: 2^31 级边界件上, np.unique 的排序+inverse 与
            # order/rank/lay/sel_all 四个 2^31 规模 int64 临时数组(各 17GB)既是内存峰值又是耗时
            # (实测 bound_030 golden 40min+ 不归, worker RSS 103GB)。bincount 计数 + 全量 cumsum
            # 重标号 + 一次 stable 基数 argsort, 产出与原机制逐位等价的 (touched, cix, order, rank)。
            counts_full = np.bincount(ix, minlength=var_first)
            touched = np.flatnonzero(counts_full)
            work = torch.from_numpy(
                np.ascontiguousarray(var.reshape(var_first, slice_size)[touched])
            )
            upd = torch.from_numpy(
                np.ascontiguousarray(upd_2d if src is None else upd_2d[src])
            )
            if acc is not None:
                work, upd = work.to(acc), upd.to(acc)
            # 行内稳定序只需要一次 stable argsort(int 键走基数排序); varFirst < 2^31 时键收窄到
            # int32 减半基数趟数。
            # argsort 交给 torch: CPU 并行排序(numpy stable 是单线程, 2^31 元素实测 10min+ 不归,
            # py-spy 栈实锤); torch.argsort stable=True 语义相同, 数十核并行下 2^31 为分钟级。
            ix_key = ix.astype(np.int32) if var_first < 2**31 else ix
            order = torch.argsort(torch.from_numpy(ix_key), stable=True).numpy()
            del ix_key
            counts = counts_full[touched]
            row_starts = np.empty(len(touched) + 1, dtype=np.int64)
            row_starts[0] = 0
            np.cumsum(counts, out=row_starts[1:])
            del counts_full
            # 分段 reduceat: 无填充、无转置拷贝、不需要 cix/rank, 峰值只与块大小有关。
            # 旧的填充立方体版在 bound_034 上被 OOM killer 打死(详见 _chain_div_seq 文档)。
            work_np, upd_np = work.numpy(), upd.numpy()
            if not is_int:
                _chain_div_seq(work_np, upd_np, order, counts, row_starts)
            else:
                _chain_intdiv_seq(work_np, upd_np, order, counts, row_starts)
            res_np = work.numpy()

            # 浮点出口不 cast —— TTK 负责窄回; 自行 astype 会把 Promote 出来的高精度真值砍回去。
            # 整型没有 Promote, TTK 也不会窄回, 故由本函数复刻内核的 NarrowStore(仅 int8/uint8)。
            if need_narrow:
                res_np = res_np.astype(var.dtype)  # 复刻 NarrowStore
            out.reshape(var_first, slice_size)[touched] = res_np

    return [out]


def __golden_scatter_div_e2e(var, indices, updates, use_locking=None, **kwargs):
    def _np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    return __golden_scatter_div(_np(var), _np(indices), _np(updates))


__golden__ = {
    "kernel": {"scatter_div": "__golden_scatter_div"},
    "e2e": {"tf.compat.v1.scatter_div": "__golden_scatter_div_e2e"},
}

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册: 保留原 golden，补三方标杆。
# third_party 用 torch / tf 竞品算子在设备侧跑，供 cross_check 比对。
# 不注册 customize_inputs —— 索引取值(含跨桶同低位、超 int32、INT64_MAX 等)直接写在
# 用例集的 input_data_ranges 多元组里，由 TTK 保证这些值必然出现在生成数据中。
# golden 只负责算参照值，不承担输入构造。
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


def _scatter_numel(t):
    """张量元素数; 对 torch / numpy / tf 的 eager 张量都成立(只读 shape)。"""
    shape = getattr(t, "shape", None)
    if shape is None:
        return None
    n = 1
    for d in tuple(shape):
        n *= int(d)
    return n


def _scatter_noop(var, updates):
    """空张量短路判据: var 或 updates 无元素时 scatter 是 no-op, 原样返回 var。

    必须在 updates 展平之前判。展平写的是 reshape((-1,) + var.shape[1:]), 当 var 的
    **非首维含 0**(切片宽度为 0)时, 0 个元素铺进 [-1, 0] 的 -1 无法唯一推断, torch 抛
    "cannot reshape tensor of 0 elements into shape [-1, 0] ... is ambiguous", tf 同理。
    合法入参下 updates 为空 <=> indices 为空(shape 约束 updates = indices + var[1:],
    且此时 var[1:] 全非 0), 两种情形都是 no-op, 故本判据既充分也不会误伤。

    kernel 通路那份 golden 早有这道短路(sliceSize==0 直接返回原 var), 是 aclnn spec
    与三方腿新增时漏带。算子侧同样是 no-op: scatter_reduce_common_tiling.cpp 对空张量
    按结构合法放行, scatter_reduce_common_simt.h 的 Init/Process 在 tiling_.sliceSize == 0
    时直接 return。
    """
    nv = _scatter_numel(var)
    nu = _scatter_numel(updates)
    return nv == 0 or nu == 0


def _tp_t(x):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。

    ⚠️ 不能把 fp16/bf16 升 fp32 再算。本算子对重复索引是**链式**运算, 误差随链长累积;
    三方若用更高精度跑, cross_check 拿到的就是"内核误差 / 一个 fp16 实现物理上够不到的
    参照"之比, 会系统性判红——实测 scatter_div fp16 链长中位 8 的用例 mare_ratio 12.78
    (限值 5), 改回与算子同 dtype 的语义后 0.65 通过, 内核本身的误差 99.99% 落在 fp16
    链式误差预算内。三方必须与算子同 dtype 语义, 内核误差才有可比对象。
    golden 反过来要保持高精度(fp32 链), 它是两条腿共同的参照点。
    """
    if isinstance(x, torch.Tensor):
        return x.clone()
    t = torch.as_tensor(np.asarray(x))  # 仅本地自测兜底: 框架侧不会走到
    return t.to(torch.float32) if t.dtype == torch.bfloat16 else t.clone()


def _tp_widen(t):
    """加宽到内核的累加类型(fp16/bf16 按 fp32 累算), 与内核同算法: 链式相除整条链
    都在加宽类型上做, 只在出口窄一次。整型不涉及: 整型判据是 binary_equal, need_3party=False, TTK 不执行 GPU 腿。"""
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.float()
    if t.dtype in (torch.int8, torch.uint8):
        return t.to(
            torch.int32
        )  # 内核 AccT 对 1 字节类型提升到 int32(SubwordWidenToI32)
    return t


def _tp_narrow(outs, dt):
    """算完窄回算子输出 dtype, 必须与 _tp_widen 成对。"""
    # 整型也要窄回: torch 在整型路径上会把中间量升到 int32, 只窄浮点会让 int8/uint8
    # 出参漏成 int32。三方腿出口必须与算子 dtype 一致(复刻内核 NarrowStore)。
    return [o.to(dt) for o in outs]


class _ScatterDivCompose:
    def __call__(self, var, indices, updates, **kwargs):
        work0 = _tp_t(var)
        if _scatter_noop(work0, updates):
            return [work0]
        _dt = work0.dtype
        work = _tp_widen(work0)
        upd = _tp_widen(_tp_t(updates)).reshape((-1,) + tuple(work.shape[1:]))
        it = (
            indices
            if isinstance(indices, torch.Tensor)
            else torch.as_tensor(np.asarray(indices))
        )
        idx = it.reshape(-1).to(torch.int64)
        valid = (idx >= 0) & (idx < work.shape[0])
        idx, upd = idx[valid], upd[valid]
        # div 无 index_reduce 归约模式: 按索引顺序逐条相除(重复索引累除, 与算子语义一致)。
        # 整型必须走截断除且除数为 0 时保持原值, 不能升 fp32——大 int32 经 float32 会静默丢精度。
        # 整型判定与 golden 同源(_INT_DTYPES = int32/int8/uint8, 即 def 注册的整型面),
        # 不用 dtype.is_floating_point 泛判, 避免与 golden 的分支口径分叉。
        # _INT_DTYPES = {int32, int8, uint8}（def 注册的整型面）的 torch 对应;
        # 不能用 work.numpy() 反查——CUDA 张量转不了 numpy。
        is_int = _dt in (torch.int32, torch.int8, torch.uint8)
        # 分层向量化: 串行依赖只在同一行内部, 不同行互相独立。按作用次序分层、层内
        # 批量相除, 循环次数由索引数降为单行最大重复次数; stable 排序保证同行内仍按
        # 原顺序, 与逐条相除逐位一致。
        n = int(idx.numel())
        if n:
            ix = idx.cpu().numpy()
            order = np.argsort(ix, kind="stable")
            s = ix[order]
            rank = np.arange(n) - np.searchsorted(s, s, side="left")
            lay = np.argsort(rank, kind="stable")
            sel_all = torch.as_tensor(order[lay], device=idx.device)
            row_all = torch.as_tensor(ix[order[lay]], device=idx.device).to(torch.int64)
            rk = rank[lay]
            bounds = np.searchsorted(rk, np.arange(int(rk.max()) + 2))
            for r in range(len(bounds) - 1):
                a, b = int(bounds[r]), int(bounds[r + 1])
                if a == b:
                    continue
                sel, rows = sel_all[a:b], row_all[a:b]
                d = upd.index_select(0, sel)
                cur = work.index_select(0, rows)
                if is_int:
                    nonzero = d != 0
                    safe = torch.where(nonzero, d, torch.ones_like(d))
                    q = torch.div(cur, safe, rounding_mode="trunc")
                    work[rows] = torch.where(nonzero, q, cur)
                else:
                    work[rows] = torch.div(cur, d)
        return _tp_narrow([work], _dt)


# ---------------------------------------------------------------------------
# 三方(GPU)腿按内核算法转写: 入口复刻 Cast<U, T, 0>(窄类型加宽到内核计算类型),
# 出口复刻 Cast<T, U, 1> / NarrowStore(窄回算子输出 dtype)。两者**必须成对**:
#   * 缺入口加宽 -> 三方在窄类型上逐步截断, 与融合内核(中间量全程 fp32)不是同一算法;
#     A100 实测 fp16 下 300*300 直接得 inf, 而 (a*a)/a 真值 300 明明存得下。
#   * 缺出口窄回 -> 三方停在 fp32, 与走 TTK Promote(fp16->fp32) 的 golden 逐位相等,
#     双标杆塌成单标杆, 三比值分母夹到精度标准 §4.5.1 的 err, 有量纲的 RMSE 比值随
#     输出量级线性放大而假红(真机实测: 不窄回 rmse 比值 2348.6, 窄回后 1.0000)。
# 整型不动: 内核对 int 也是原生/int32 累加, 走一趟 fp32 会把 >2^24 抹掉低位。
# 只作用于 third_party(GPU); CPU golden 由 TTK 的 Promote 单独喂高精度输入, 不受影响。
# ---------------------------------------------------------------------------


def _kf_widen(a, seen):
    if isinstance(a, (list, tuple)):
        return type(a)(_kf_widen(x, seen) for x in a)
    if isinstance(a, torch.Tensor):
        if a.dtype in (torch.float16, torch.bfloat16):
            seen.append(a.dtype)
            return a.float()
        return a
    if isinstance(a, np.ndarray) or (_KF_BF16 is not None and hasattr(a, "dtype")):
        n = np.asarray(a)
        if n.dtype.kind == "V" and _KF_BF16 is not None:
            n = n.view(_KF_BF16)
        if _KF_BF16 is not None and n.dtype == _KF_BF16:
            seen.append(torch.bfloat16)
            return n.astype(np.float32)
        if n.dtype == np.float16:
            seen.append(torch.float16)
            return n.astype(np.float32)
        return a
    return a


def _kf_narrow(o, dt):
    if isinstance(o, (list, tuple)):
        return type(o)(_kf_narrow(x, dt) for x in o)
    if isinstance(o, torch.Tensor) and o.is_floating_point():
        return o.to(dt)
    return o


class _TpKernelFaithful:
    _INNER = _ScatterDivCompose

    def __call__(self, *args, **kwargs):
        seen = []
        wa = [_kf_widen(a, seen) for a in args]
        wk = {k: _kf_widen(v, seen) for k, v in kwargs.items()}
        outs = self._INNER()(*wa, **wk)
        return outs if not seen else _kf_narrow(outs, seen[0])


# TTK 服务端按 __call__ 的**签名**做入参绑定(remote/server/executor.py::_bind ->
# _function_param_names)。包装类若只写 *args/**kwargs, 服务端取不到形参名, 会退化成
# 位置传参、同时又传同名关键字 -> "got multiple values for argument ..." 而三方腿整个不可用。
# 故把内层 compose 的签名透传出去。
try:
    _TpKernelFaithful.__call__.__signature__ = _kf_inspect.signature(
        _ScatterDivCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class _ScatterDivTfCompose:
    """tf 腿适配类。TF 侧对标算子是 ScatterDiv(tf_plugin 的 OriginOpType), 但它是
    **ref-variable 语义**: tf.raw_ops.ScatterDiv 首参名叫 ref 且必须是可变引用, 与
    def.cpp 的 var 既不同名、也不能直接喂 eager tensor, 故用适配类包 tf.Variable 再调。

    与 torch 腿逐条对齐的三处语义(不一致就会系统性假红):
    1) 非法下标(越界/负)按算子语义静默跳过——TF 自己会抛 InvalidArgumentError;
    2) 除数含 0 时不能直接调 TF 的 ScatterDiv: 它对**所有 dtype**硬性拒收零除数
       (实测 tf 2.21: InvalidArgumentError "updates must not contain 0", 浮点也拒),
       即"除数为 0"落在 TF 该算子的定义域之外。而本算子对零除数是有定义的:
       整型保持原值、浮点按普通除法给 ±inf(见 golden 与 torch 腿)。故按 dtype 分流——
         · 整型: 把 0 除数换成 1(x/1 == x, 等价于保持原值), 仍走单次 scatter_div;
         · 浮点且含 0 除数: 退化成按索引顺序的链式 tf.divide(TF 自己的除法核, ±inf
           与 nan 的产生方式与 TF 一致), 只有这条罕见分支走循环, 常规用例仍是单次调用;
    3) **不升精度**: 一律用算子自身 dtype 算(见 _tp_t 的说明——本算子对重复索引是链式
       运算, 三方升精度会让 cross_check 拿 fp16 实现够不到的参照去比, 系统性判红)。
    TF 的整型除法本身就是截断除, 与算子/golden 的 trunc 语义一致, 无需额外处理。
    """

    def __call__(self, var, indices, updates, **kwargs):
        import tensorflow as tf

        work = tf.convert_to_tensor(var)
        if _scatter_noop(work, updates):
            return [work]
        _tf_dt = work.dtype  # 算子输出 dtype, 出口窄回用
        if _tf_dt in (tf.float16, tf.bfloat16):
            work = tf.cast(work, tf.float32)  # 复刻内核 LoadWiden
        upd = tf.reshape(
            tf.convert_to_tensor(updates), tf.concat([[-1], tf.shape(work)[1:]], axis=0)
        )
        idx = tf.reshape(tf.cast(tf.convert_to_tensor(indices), tf.int64), [-1])
        keep = tf.logical_and(idx >= 0, idx < tf.cast(tf.shape(work)[0], tf.int64))
        idx, upd = tf.boolean_mask(idx, keep), tf.boolean_mask(upd, keep)
        if tf.size(idx) == 0:
            return [tf.cast(work, _tf_dt) if work.dtype != _tf_dt else work]
        upd = tf.cast(upd, work.dtype)
        has_zero = bool(tf.reduce_any(tf.equal(upd, tf.zeros_like(upd))))
        # 整型判定与 golden/torch 腿同源(def 注册的整型面), 不用 is_floating 泛判。
        is_int = work.dtype in (tf.int32, tf.int8, tf.uint8)

        if is_int and has_zero:
            upd = tf.where(tf.equal(upd, tf.zeros_like(upd)), tf.ones_like(upd), upd)
            has_zero = False

        if not has_zero:
            ref = tf.Variable(work)
            tf.compat.v1.scatter_div(ref, tf.cast(idx, tf.int32), upd)
            out = tf.convert_to_tensor(ref)
            return [tf.cast(out, _tf_dt) if out.dtype != _tf_dt else out]

        # 浮点 + 零除数: TF 的 ScatterDiv 拒收, 按索引顺序链式除(与算子/golden 同序)。
        rows = idx.numpy().tolist()
        for k, i in enumerate(rows):
            work = tf.tensor_scatter_nd_update(
                work, [[i]], tf.expand_dims(tf.divide(work[i], upd[k]), 0)
            )
        # 链式除同样要窄回: 上面的加宽只为复刻 LoadWiden, 不能泄漏到出参
        return [tf.cast(work, _tf_dt) if work.dtype != _tf_dt else work]


_GOLDEN_FN = __golden_scatter_div


class ScatterDivKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _TpKernelFaithful, "tf": _ScatterDivTfCompose}
    tolerance = _TOL_KERNEL


# e2e(TF 前端)通路判据: 与 kernel 腿同口径。不声明则回落默认的绝对容差判据,
# 输出量级接近 dtype 上限时 1 ULP 即超限。
_TOL_E2E = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


class _TpE2eDiv:
    """e2e 三方腿适配: 该通路按框架 API 形参名下发, 与 def 注册名不同, 故另立适配类
    按位置转调同一竞品类, 不改变竞品语义。"""

    def __call__(self, ref, indices, updates, use_locking=None, **kwargs):
        return _ScatterDivCompose()(ref, indices, updates)


class _TpE2eDivTf:
    """同上, tf provider 腿。"""

    def __call__(self, ref, indices, updates, use_locking=None, **kwargs):
        return _ScatterDivTfCompose()(ref, indices, updates)


class ScatterDivE2eSpec:
    """e2e 通路 spec: 三方腿与判据。"""

    third_party = {"torch": _TpE2eDiv, "tf": _TpE2eDivTf}
    tolerance = _TOL_E2E


class ScatterDivAclnnSpec:
    """aclnn 通路 spec。

    golden 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名对齐 aclnnScatterDivGetWorkspaceSize 的 varRef/indices/updates/useLocking;
    third_party 按**形参名**绑定, 用适配类把头文件的 varRef 接到竞品类的 def 注册名 var。
    """

    @staticmethod
    def golden(varRef, indices, updates, useLocking=None, **kwargs):
        """入参为 torch 张量/numpy 视图, 还原成 numpy 后转接 kernel 档 golden(单一真源)。

        类体内不能直接写 __golden_xxx —— 会被改写成 _ScatterDivAclnnSpec__golden_xxx, 故用 _GOLDEN_FN。
        """

        def _np(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

        return _GOLDEN_FN(_np(varRef), _np(indices), _np(updates), **kwargs)

    class _Compose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterDivCompose()(varRef, indices, updates, **kwargs)

    class _TfCompose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterDivTfCompose()(varRef, indices, updates, **kwargs)

    third_party = {"torch": _Compose, "tf": _TfCompose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# __spec__ 已注册: kernel + GEIR(复用 kernel spec) + aclnn + e2e(TF 前端);
#   三个 spec 的 third_party 均为 {"torch", "tf"}, tf 腿跑 --provider tf。
# TF 通路连通性: 算子目录下有 framework 的 tf_plugin(OriginOpType "ScatterDiv"), 但 NPU 侧要把
#   TF 图下沉到本算子需要 Ascend TF adapter(npu_device/tfplugin); 该组件未装时按
#   预生成 .pb + aclgrphParseTensorFlow 验证连通性。
# ONNX / 融合 pass: 均未交付。


__spec__ = {
    "scatter_div": "ScatterDivKernelSpec",
    "tf.compat.v1.scatter_div": "ScatterDivE2eSpec",
    "aclnnScatterDiv": "ScatterDivAclnnSpec",
}
