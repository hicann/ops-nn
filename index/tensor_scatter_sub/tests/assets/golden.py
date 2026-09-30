#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
tensor_scatter_sub Golden TestSpec

Golden 计算来源：SE 文档第 7 章"Golden 计算"
- 语义：y = copy(x)；按 indices 逐条将 updates 对应 slice 从 y 中减去
- 主路径：tf.tensor_scatter_nd_sub（与标杆项目 ScatterNdSub golden
  /home/developer/SIMT-LongTails/ScatterNdSub/deliverables/thirdparty_golden/golden.py
  的 _tf_scatter_nd_sub_any_k 同款实现，该项目最终 TTK 通过率 100%）
- 重复索引：TF CPU 串行逐条累减（确定性，与 SE §5.2 一致）
- 越界索引：kernel 内跳过；测试用例不构造越界索引（越界属于非法输入）

TTK 第5轮回调修复（golden 二次对齐标杆 _tf_scatter_nd_sub_any_k）：
- 主路径封装为 _tf_scatter_nd_sub_any_k，与标杆同款：
  ① rank-1 indices（shape (K,)）→ reshape (1,K) + tf.expand_dims(updates, 0)，
     同时覆盖标量 updates 退化形态（替代第4轮 numel 归一化中由该形态触发的
     特判分支；TTK 数据生成 ()→(1,) 的吸收逻辑仍保留在 golden 入口）
  ② K = indices.shape[-1] 超过 TF 实例化上限时按 stride 展平为线性索引
     （int64 计算防溢出），x/updates reshape 为 (N, slice) 后走 k=1 接口，
     再 reshape 回原 shape，全程 TF 计算不用 numpy。
     K 阈值取 5：标杆注释 TF 实例化 k<=7，本环境（TF 2.21）实测 k<=5，
     按"先尝试直接调、按 K 阈值走展平"原则以实测 5 为准
  ③ 空 tensor 守卫保留：x/indices/updates 任一 size==0 返回 x 拷贝
- third_party provider 由 torch 改为 tf（与标杆一致：同接口 tf.Tensor 进
  tf.Tensor 出，不做类型转换），Torch 实现删除
- 参数名按本算子 def.cpp：x/indices/updates，无 use_locking，out-of-place

保留的环境约束差异（相对标杆，禁止改动）：
- 容差保持现状：float 仅声明 rtol/atol（无 standard），int 显式
  binary_equal；禁止引入 cross_check/mix_tolerance（本环境无 XPU
  endpoint，cross_check 会阻塞全部 float 用例；二者均为 harness
  PROMOTE_GOLDEN_TOKENS，会触发 golden 升精度路径）

第6轮回调（golden 去 numpy 计算）：
- 超大 tensor（x 元素数 > 2^31）兜底路径由分块 numpy 改为纯 TF 分块
  实现 _tf_chunked_tensor_scatter_sub：按 slice 分块（每块 4096 条），
  每块直接调用 _tf_scatter_nd_sub_any_k（k<=5 走
  tf.tensor_scatter_nd_sub，k>5 走 TF stride 展平）顺序累减，块间
  顺序 = 输入顺序，确定性串行语义不变
- numpy 仅用于输入适配（ascontiguousarray/shape 判断/reshape 元数据）
  与最终 .numpy() 输出转换，全 golden 不再用 numpy 做任何减法/累加
  等数值计算

第7轮回调（int32 索引 + x>2^31 分块路径 int64 索引适配）：
- TF 内部对 params 元素数超 2^31 的 int32 索引报
  "params_shape[0] too large for int32 indexing"（int64 索引不受影响）。
  _tf_chunked_tensor_scatter_sub 块内调用 _tf_scatter_nd_sub_any_k 前，
  当 x 元素数 > 2^31 时将块索引 tf.cast 为 int64：int32 索引语义不变
  （索引值域不变，仅索引 dtype 提升），规避 TF 内部 int32 indexing
  限制；K>5 展平路径内部本就 cast int64，此处前置 cast 对其为无害
  恒等（int64→int64），两条块内分支（k<=5 直接调 / K>5 展平）统一
  覆盖。仅分块兜底路径受影响，主路径（x <= 2^31）行为完全不变。

第8轮回调（third_party 直调路径 updates 形状归一化）：
- GPU 侧 third_party __call__ 直调 _tf_scatter_nd_sub_any_k，绕过
  golden() 入口的 expected-shape 归一化保护；rank-1 indices 分支对
  updates 无条件 expand_dims(...,0)，当 updates 已物化为 (1,)（标量
  () 经 TTK/GPU 侧物化）且 K==rank(x) 时变成 (1,1)，TF 报
  "Inner dimensions of output shape must match inner dimensions of
  updates shape"。修复：_tf_scatter_nd_sub_any_k 内在 rank-1 规整后
  按 SE 公式 expected = indices.shape[:-1] + x.shape[K:] 对 updates
  做形状归一（numel 匹配才 reshape），对已正确形状的所有路径
  （rank>=2 indices、K>5 展平、标量 updates）为无害恒等，全程纯
  TF 计算。

第9轮扩展（TF E2E 通路 Spec）：
- 新增 TfTensorScatterNdSubSpec（api_name=tf.tensor_scatter_nd_sub）与
  TfCompatV1TensorScatterNdSubSpec（api_name=tf.compat.v1.tensor_scatter_nd_sub），
  对齐标杆 ScatterNdSub golden 的 TfScatterNdSubSpec/TfCompatV1ScatterNdSubSpec
  双 Spec 范式：
  ① golden/third_party 均收到 CPU 侧 tf.Tensor（E2E 链路），golden 返回
     numpy list（框架统一比对），third_party tensor 进 tensor 出不转换
  ② 签名按 TF 接口 (tensor, indices, updates, name=None)：
     tf.compat.v1.tensor_scatter_nd_sub 为函数式接口（无 ref Variable，
     与 v2 语义相同），两个 Spec 结构一致，仅注册 api_name 不同
  ③ 计算复用 _tf_scatter_nd_sub_any_k；超大 tensor（>2^31）复用
     _tf_chunked_tensor_scatter_sub 分块兜底；空 tensor 守卫与标杆
     逐字对齐（任一 numel==0 返回 identity）
  ④ tolerance 与 TensorScatterSubTestSpec 完全一致（float 仅 rtol/atol、
     int binary_equal，环境约束差异同上，禁止改动）
- 现有 TensorScatterSubTestSpec 不动；保持纯 TF 计算、不引入 numpy
  数值计算

容差策略（值保持 SE §5.5）：
- float16: rtol=1e-3/atol=1e-8；float32: rtol=1e-4/atol=1e-8
  （TestSpec dict 仅声明 rtol/atol、无 standard；CSV 为纯元组 ((rtol, atol),)）
- int32/int8/uint8: binary_equal（精确比对，等价 ((0.0, 0.0),)）

third_party（GEIR 远端派发）：
- provider: tf，与标杆一致，同接口 tensor 进 tensor 出不转换
"""

import gc
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

__spec__ = {
    "tensor_scatter_sub": "TensorScatterSubTestSpec",  # Kernel / GEIR 流程（按 op_name 查找）
    "tf.tensor_scatter_nd_sub": "TfTensorScatterNdSubSpec",  # E2E TF v2 流程（按 api_name 查找）
    "tf.compat.v1.tensor_scatter_nd_sub": "TfCompatV1TensorScatterNdSubSpec",  # E2E TF v1 流程
}

# 超大 tensor 阈值：x 元素数 > 2^31 时单次 tf.tensor_scatter_nd_sub
# 对全量 updates 一次性实例化内存不可控，走分块 TF 兜底（每块 4096 条
# slice 顺序累减，峰值内存 ≈ 2×sizeof(x)）
_TF_INT32_INDEX_LIMIT = 2**31
# tf.tensor_scatter_nd_sub 的 K = indices.shape[-1] 实例化上限。
# 标杆环境注释 k<=7；本环境 TF 2.21 实测 k<=5，以实测为准
_TF_MAX_K = 5
# 超大 tensor 分块兜底每块处理的 slice 条数
_CHUNK_SLICES = 4096


def _tf_scatter_nd_sub_any_k(x, indices, updates):
    """支持任意索引深度 K 的 tf.tensor_scatter_nd_sub（对标标杆实现）。

    rank-1 indices（shape (K,)）：TF 实现要求 indices 至少 rank 2，
    单索引一维形式会被误读，先 reshape 为 (1, K)、updates 加 batch 维
    （同时覆盖 K == rank(x) 时 updates 为标量的退化形态）。

    TF 仅实例化 K<=5（本环境实测；标杆注释 k<=7，阈值以实测 5 为准）；
    K>5 时按 stride 将多维索引展平为线性索引（int64 计算防溢出），
    x/updates reshape 为 (N, slice) 后走 K=1 接口（必然已实例化），
    再 reshape 回原 shape。全程 TF 计算，不用 numpy。
    语义与 tensor_scatter_nd_sub 一致（重复索引逐个累减）。

    updates 形状归一化（第8轮修复）：rank-1 分支对 updates 无条件
    expand_dims(..., 0)，当 updates 已物化为 (1,)（标量 () 经
    TTK/GPU 侧物化）且 K == rank(x) 时会变成 (1,1)，而 TF 期望
    (1,)（报 "Inner dimensions of output shape must match inner
    dimensions of updates shape"）。golden() 入口有 expected-shape
    归一化保护，但 third_party __call__ 直调本函数无此保护，故在
    本函数内按 SE 公式 expected = indices.shape[:-1] + x.shape[K:]
    统一归一：形状不符且 numel 匹配才 reshape，对已正确形状的
    所有路径（rank>=2 indices、K>5 展平、标量 updates）为无害恒等。
    """
    if indices.shape.rank == 1:
        indices = tf.reshape(indices, [1, -1])
        updates = tf.expand_dims(updates, 0)
    k = indices.shape[-1]
    expected = tuple(indices.shape[:-1]) + tuple(x.shape[k:])
    expected_numel = 1
    for d in expected:
        expected_numel *= int(d)
    if (
        tuple(updates.shape) != expected
        and updates.shape.num_elements() == expected_numel
    ):
        updates = tf.reshape(updates, expected)
    if k <= _TF_MAX_K:
        return tf.tensor_scatter_nd_sub(x, indices, updates)
    x_shape = tf.shape(x)
    head, tail = x_shape[:k], x_shape[k:]
    slice_size = tf.reduce_prod(tail)  # k == rank(x) 时 tail 为空，reduce_prod 得 1
    strides = tf.stack([tf.reduce_prod(head[i + 1 :]) for i in range(k)])
    flat_idx = tf.reduce_sum(
        tf.cast(indices, tf.int64) * tf.cast(strides, tf.int64), axis=-1
    )
    flat_x = tf.reshape(x, [-1, slice_size])
    flat_upd = tf.reshape(updates, [-1, slice_size])
    out = tf.tensor_scatter_nd_sub(flat_x, tf.reshape(flat_idx, [-1, 1]), flat_upd)
    return tf.reshape(out, x_shape)


def _tf_chunked_tensor_scatter_sub(x, indices, updates, chunk_slices=_CHUNK_SLICES):
    """超大 tensor 纯 TF 分块兜底（x 元素数 > 2^31）。

    语义与 tf.tensor_scatter_nd_sub 完全一致：y = copy(x)；按 indices
    逐条将 updates 对应 slice 从 y 中减去；重复索引按输入顺序串行累减
    （确定性，与 SE §5.2 一致）。

    - 分块策略：按 chunk_slices 条 slice 为一块，每块直接调用
      _tf_scatter_nd_sub_any_k（k<=5 走 tf.tensor_scatter_nd_sub，
      k>5 走 TF stride 展平，均为纯 TF 计算）顺序累减；块间顺序 =
      输入顺序，确定性串行语义不变
    - 无 K（indices.shape[-1]）上限，支持任意维度（含 8D）
    - 内存友好：全程仅持有 numpy 输入 x 与 TF 输出载体 y 两份全量
      数据；每块的索引/updates 临时张量处理完立即 del +
      gc.collect() 释放，无全量中间张量堆积，单条 golden 峰值内存
      ≈ 2×sizeof(x)（TF 输出载体 out-of-place 更新瞬时两份 y）
    - numpy 仅用于输入适配（shape 判断/reshape 元数据），减法/累加
      全部由 TF 完成；输出由调用方经 .numpy() 一次性转换
    - 第7轮修复：x 元素数 > 2^31 时，块内调用前将块索引 cast 为
      int64（int32 索引语义不变，仅规避 TF 内部
      "params_shape[0] too large for int32 indexing" 限制；K>5
      展平路径内部已 cast int64，前置 cast 对其为无害恒等）
    """
    y = tf.constant(x)  # 唯一一份全量 TF 拷贝（输出载体，不原地改 x）
    k = indices.shape[-1]
    idx = indices.reshape(-1, k)
    num_slices = idx.shape[0]
    slice_shape = tuple(x.shape[k:])
    slice_size = 1
    for d in slice_shape:
        slice_size *= int(d)
    upd2d = updates.reshape(num_slices, slice_size)
    # x 元素数 > 2^31 时 TF 内部 int32 indexing 溢出，块索引统一提为
    # int64（仅 dtype 提升，索引值与语义不变）。本函数即由该超大分支
    # 触发，条件恒真；保留条件判断以防御未来其它调用方复用本函数。
    cast_idx_int64 = x.size > _TF_INT32_INDEX_LIMIT
    for start in range(0, num_slices, chunk_slices):
        end = min(start + chunk_slices, num_slices)
        idx_blk = tf.constant(idx[start:end])
        if cast_idx_int64:
            idx_blk = tf.cast(idx_blk, tf.int64)
        upd_blk = tf.constant(upd2d[start:end].reshape((end - start,) + slice_shape))
        y = _tf_scatter_nd_sub_any_k(y, idx_blk, upd_blk)
        del idx_blk, upd_blk
        gc.collect()
    del idx, upd2d
    gc.collect()
    return y


class TensorScatterSubTestSpec:
    """One TestSpec shared by Kernel and GEIR.

    Kernel/GEIR golden 输入为 numpy.ndarray。主路径使用
    _tf_scatter_nd_sub_any_k（对齐 ScatterNdSub 标杆 golden 的同款
    实现：rank-1 indices 规整 + K>5 stride 展平，全程 TF）；超大
    tensor（x 元素数 > 2^31）走纯 TF 分块兜底
    _tf_chunked_tensor_scatter_sub（每块 4096 条 slice 顺序调用
    _tf_scatter_nd_sub_any_k 累减），语义与 TF 逐条 slice 减法一致。
    全 golden 仅用 TF 做数值计算，numpy 只做输入适配与输出转换。
    third_party 为 tf provider（tensor 进 tensor 出，不做类型转换）。
    """

    def golden(x, indices, updates, **kwargs):
        x = np.ascontiguousarray(x)
        indices = np.ascontiguousarray(indices)
        updates = np.ascontiguousarray(updates)

        # 空 tensor 守卫（与标杆 golden 逐字对齐）：
        # x/indices/updates 任一 size==0 时返回 x 拷贝
        # （tf.tensor_scatter_nd_sub 不接受空 tensor，语义退化为恒等）
        if x.size == 0 or indices.size == 0 or updates.size == 0:
            return [np.array(x, copy=True)]

        # TTK 数据生成 ()→(1,) 吸收逻辑 + updates numel 归一化：
        # TTK 数据生成将标量 shape () 物化为 (1,)（eliminate_scalar_shapes），
        # 按 SE 公式 updates.shape == indices.shape[:-1] + x.shape[K:]
        # 规范化，仅在 numel 匹配时 reshape，不改变任何计算语义。
        # 标量 updates 退化形态（indices (K,) + K==rank(x)，expected==()）
        # 由此吸收回 0-d，随后由 _tf_scatter_nd_sub_any_k 的 rank-1 分支
        # （reshape (1,K) + expand_dims(updates,0)）统一处理走 TF 主路径。
        k = indices.shape[-1] if indices.ndim > 0 else 0
        expected = tuple(indices.shape[:-1]) + tuple(x.shape[k:])
        expected_numel = 1
        for d in expected:
            expected_numel *= int(d)
        if tuple(updates.shape) != expected and updates.size == expected_numel:
            updates = updates.reshape(expected)

        # 超大 tensor 兜底：x 元素数 > 2^31 时走纯 TF 分块实现
        # _tf_chunked_tensor_scatter_sub（每块 4096 条 slice 顺序调用
        # _tf_scatter_nd_sub_any_k 累减；K>5 由块内 stride 展平路径
        # 覆盖）。全程 TF 数值计算，numpy 仅做输入适配与 .numpy()
        # 输出转换。
        if x.size > _TF_INT32_INDEX_LIMIT:
            out = _tf_chunked_tensor_scatter_sub(x, indices, updates)
            return [out.numpy()]

        # TF 主路径（_tf_scatter_nd_sub_any_k，与标杆同款实现）
        out = _tf_scatter_nd_sub_any_k(
            tf.constant(x), tf.constant(indices), tf.constant(updates)
        )
        return [out.numpy()]

    class ThirdPartyImpl:
        """TF provider（与标杆一致）：同接口 tf.Tensor 进 tf.Tensor 出，
        不做类型转换，不反调 Golden wrapper。"""

        def __init__(self, x=None, indices=None, updates=None, **kwargs):
            pass

        def __call__(self, x, indices, updates, **kwargs):
            if (
                x.shape.num_elements() == 0
                or indices.shape.num_elements() == 0
                or updates.shape.num_elements() == 0
            ):
                return [tf.identity(x)]
            return [_tf_scatter_nd_sub_any_k(x, indices, updates)]

    # GEIR remote dispatch needs an explicit provider dict（provider 由
    # torch 改为 tf，与标杆一致）。
    third_party = {"tf": ThirdPartyImpl}

    # TTK 第4轮回调修复：去除 mix_tolerance/cross_check 声明（二者均为
    # harness PROMOTE_GOLDEN_TOKENS，会触发 golden 升精度路径；且本环境
    # 无 XPU endpoint，cross_check 会阻塞全部 float 用例），容差值保持
    # SE §5.5 不变，与 CSV precision_tolerances 纯元组 ((rtol, atol),) 对齐。
    # 注：TTK test_spec validator 强制 tolerance[dtype] 为 dict（字面元组会
    # 抛 InvalidSpecError），故 float 采用"仅 rtol/atol、无 standard 声明"
    # 的合规等价形态；int 显式 binary_equal（非 Promote token，等价 (0.0,0.0)）。
    tolerance = {
        "float16": {"rtol": 0.001, "atol": 1e-8},
        "float32": {"rtol": 0.0001, "atol": 1e-8},
        "int32": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
    }


class TfTensorScatterNdSubSpec:
    """E2E TF v2 流程 — 按 api_name tf.tensor_scatter_nd_sub 注册。

    golden / third_party 均收到 CPU 侧 tf.Tensor（按
    tf.tensor_scatter_nd_sub 签名参数名 tensor/indices/updates 注入，
    签名中的 name 参数框架会按位置传入，需以 name=None 接收），
    golden 返回 numpy list 由框架统一比对；third_party 返回 tf.Tensor，
    不做类型转换（tensor 进 tensor 出）。

    计算复用 Kernel 链路同款 _tf_scatter_nd_sub_any_k（rank-1 indices
    规整 + updates 形状归一 + K>5 stride 展平，全程 TF）；超大 tensor
    （元素数 > 2^31）复用 _tf_chunked_tensor_scatter_sub 分块兜底
    （E2E 用例均为中小 shape，该分支为防御性兜底，正常不触发）。
    空 tensor 守卫与标杆逐字对齐：tensor/indices/updates 任一
    numel==0 时返回 tensor 恒等拷贝（语义退化为恒等变换）。
    tolerance 与 TensorScatterSubTestSpec 完全一致。
    """

    def golden(tensor, indices, updates, name=None, **kwargs):
        if (
            tensor.shape.num_elements() == 0
            or indices.shape.num_elements() == 0
            or updates.shape.num_elements() == 0
        ):
            return [tf.identity(tensor).numpy()]
        if tensor.shape.num_elements() > _TF_INT32_INDEX_LIMIT:
            out = _tf_chunked_tensor_scatter_sub(
                tensor.numpy(), indices.numpy(), updates.numpy()
            )
            return [out.numpy()]
        return [_tf_scatter_nd_sub_any_k(tensor, indices, updates).numpy()]

    class ThirdPartyImpl:
        """TF provider（与标杆一致）：同接口 tf.Tensor 进 tf.Tensor 出，
        不做类型转换，不反调 Golden wrapper。"""

        def __init__(self, tensor, indices, updates, name=None, **kwargs):
            pass

        def __call__(self, tensor, indices, updates, **kwargs):
            if (
                tensor.shape.num_elements() == 0
                or indices.shape.num_elements() == 0
                or updates.shape.num_elements() == 0
            ):
                return [tf.identity(tensor)]
            if tensor.shape.num_elements() > _TF_INT32_INDEX_LIMIT:
                return [
                    _tf_chunked_tensor_scatter_sub(
                        tensor.numpy(), indices.numpy(), updates.numpy()
                    )
                ]
            return [_tf_scatter_nd_sub_any_k(tensor, indices, updates)]

    third_party = {"tf": ThirdPartyImpl}

    # 与 TensorScatterSubTestSpec 完全一致（环境约束：无 XPU endpoint，
    # float 仅声明 rtol/atol、无 standard；int 显式 binary_equal）。
    tolerance = {
        "float16": {"rtol": 0.001, "atol": 1e-8},
        "float32": {"rtol": 0.0001, "atol": 1e-8},
        "int32": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
    }


class TfCompatV1TensorScatterNdSubSpec:
    """E2E TF v1 流程 — 按 api_name tf.compat.v1.tensor_scatter_nd_sub 注册。

    tf.compat.v1.tensor_scatter_nd_sub 为函数式接口，签名为
    (tensor, indices, updates, name=None)，无 ref Variable（与
    tf.compat.v1.scatter_nd_sub 的 ref 形式不同），数学语义与 v2 相同：
    out = tensor - scatter_nd(indices, updates, tensor.shape)。
    golden/third_party 侧收到普通 CPU 侧 tf.Tensor，结构与
    TfTensorScatterNdSubSpec 完全一致，仅注册 api_name 不同。
    """

    def golden(tensor, indices, updates, name=None, **kwargs):
        if (
            tensor.shape.num_elements() == 0
            or indices.shape.num_elements() == 0
            or updates.shape.num_elements() == 0
        ):
            return [tf.identity(tensor).numpy()]
        if tensor.shape.num_elements() > _TF_INT32_INDEX_LIMIT:
            out = _tf_chunked_tensor_scatter_sub(
                tensor.numpy(), indices.numpy(), updates.numpy()
            )
            return [out.numpy()]
        return [_tf_scatter_nd_sub_any_k(tensor, indices, updates).numpy()]

    class ThirdPartyImpl:
        """TF provider（与标杆一致）：同接口 tf.Tensor 进 tf.Tensor 出，
        不做类型转换，不反调 Golden wrapper。"""

        def __init__(self, tensor, indices, updates, name=None, **kwargs):
            pass

        def __call__(self, tensor, indices, updates, **kwargs):
            if (
                tensor.shape.num_elements() == 0
                or indices.shape.num_elements() == 0
                or updates.shape.num_elements() == 0
            ):
                return [tf.identity(tensor)]
            if tensor.shape.num_elements() > _TF_INT32_INDEX_LIMIT:
                return [
                    _tf_chunked_tensor_scatter_sub(
                        tensor.numpy(), indices.numpy(), updates.numpy()
                    )
                ]
            return [_tf_scatter_nd_sub_any_k(tensor, indices, updates)]

    third_party = {"tf": ThirdPartyImpl}

    # 与 TensorScatterSubTestSpec 完全一致（环境约束：无 XPU endpoint，
    # float 仅声明 rtol/atol、无 standard；int 显式 binary_equal）。
    tolerance = {
        "float16": {"rtol": 0.001, "atol": 1e-8},
        "float32": {"rtol": 0.0001, "atol": 1e-8},
        "int32": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
    }
