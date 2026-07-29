#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
Golden script for scatter_max_with_argmax.

算子功能：沿第0维执行 scatter max 操作，同时记录每个位置最大值来自哪个 update（argmax 索引）。

输入：
    x:       [N, D1, D2, ...] float32  # 仅用于确定输出 shape，不参与计算
    indices: [M] int32                  # scatter 索引，0 <= indices[i] < N
    updates: [M, D1, D2, ...] float32  # scatter 源数据

输出：
    y:      [N, D1, D2, ...] float32   # scatter max 结果，未被指到位置为 0（Phase 3 补零）
    argmax: [N, D1, D2, ...] int32     # 最大值来源索引（0~M-1），未被指到位置为 M

三阶段语义（与 Ascend TBE 实现一致）：
    Phase 1: y 初始化为 -3.4e+38（-inf 近似），argmax 初始化为 M
    Phase 2: scatter max + GT 严格大于比较（相等不更新 argmax，保证确定性）
    Phase 3: 对未被 indices 覆盖的位置（argmax 仍为 M），将 y 改写为 0（argmax 保持 M）

实现说明：
    本脚本采用 SE 文档第6章 Golden B（numpy 自定义实现）作为主 golden 函数，
    无额外依赖，确定性最强，适用于所有 TTK 测试环境。
    供交叉验证使用（需 pip install torch_scatter）。

参考：SE 文档第6章 Golden 计算（两个逻辑等价的 golden 实现用于交叉验证）。
"""

import numpy as np

__golden__ = {
    "kernel": {"scatter_max_with_argmax": "scatter_max_with_argmax_golden"},
    "aclnn": {"aclnnScatterMaxWithArgmax": "aclnn_scatter_max_with_argmax_golden"},
}

# float32 的 -inf 近似值，与 Ascend TBE 实现的 MINUS_INF = -3.4e+38 保持一致
_MINUS_INF = np.float32(-3.4e38)


def _scatter_max_with_argmax_numpy(x, indices, updates):
    """
    Golden B：基于 numpy 的自定义实现（无额外依赖）。

    功能：沿第0维执行 scatter max 操作，同时记录每个位置最大值来自哪个 update。
    语义等同于 torch_scatter.scatter_max(src=updates, index=indices, dim=0)。

    三阶段语义（与 Ascend TBE 实现一致）：
      Phase 1: y 初始化为 -3.4e+38（-inf 近似），argmax 初始化为 M（"无更新"标记值）
      Phase 2: 严格大于（>）比较，相等时不更新 argmax（保证确定性）；NaN > 任意值为 false，NaN 被忽略
      Phase 3: 对未被 indices 覆盖的位置（argmax 仍为 M），将 y 改写为 0（argmax 保持 M）

    越界索引被跳过（与 SE 文档"越界行为未定义"保持一致，golden 采取安全跳过策略）。
    最终未覆盖位置输出：y=0, argmax=M。
    """
    x = np.asarray(x, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int32)
    updates = np.asarray(updates, dtype=np.float32)

    out_shape = x.shape
    N = out_shape[0]
    M = indices.shape[0]

    # Phase 1: 初始化 y 为 -3.4e+38（-inf 近似），argmax 为 M（表示"无更新"标记值）
    y = np.full(out_shape, _MINUS_INF, dtype=np.float32)
    argmax = np.full(out_shape, M, dtype=np.int32)

    # Phase 2: scatter max + argmax 更新（严格大于比较，相等不更新）
    if M > 0:
        if x.ndim > 1:
            rest_shape = x.shape[1:]
            rest_size = int(np.prod(rest_shape)) if rest_shape else 1
            y_flat = y.reshape(N, rest_size)
            argmax_flat = argmax.reshape(N, rest_size)
            updates_flat = updates.reshape(M, rest_size)

            for i in range(M):
                idx = int(indices[i])
                if idx < 0 or idx >= N:
                    continue
                # 严格大于比较；NaN > 任意值 为 false，故 NaN 不会更新（被忽略）
                mask = updates_flat[i] > y_flat[idx]
                y_flat[idx] = np.where(mask, updates_flat[i], y_flat[idx])
                argmax_flat[idx] = np.where(mask, i, argmax_flat[idx])

            y = y_flat.reshape(out_shape)
            argmax = argmax_flat.reshape(out_shape)
        else:
            for i in range(M):
                idx = int(indices[i])
                if idx < 0 or idx >= N:
                    continue
                if updates[i] > y[idx]:
                    y[idx] = updates[i]
                    argmax[idx] = i

    # Phase 3: 补零阶段
    # 对未被 indices 覆盖的位置（argmax 仍为 M），将 y 从 -3.4e+38 改写为 0（argmax 保持 M 不变）
    # 与 Ascend TBE 实现的 Phase 3 补零语义一致：未覆盖位置最终输出 y=0, argmax=M
    mask_uncovered = argmax == M
    y = np.where(mask_uncovered, np.float32(0.0), y)

    return y, argmax


def scatter_max_with_argmax_golden(x, indices, updates, **kwargs):
    """
    Golden function for scatter_max_with_argmax.
    All the parameters (names and order) follow ScatterMaxWithArgmax prototype definition
    (REG_OP: x, indices, updates) without outputs.
    All the input Tensors are numpy.ndarray.

    计算逻辑（三阶段，与 Ascend TBE 实现一致）：
    - Phase 1: y 初始化为 -3.4e+38（float32 -inf 近似），argmax 初始化为 M（indices 数量，"无更新"标记值）
    - Phase 2: 严格大于（>）比较，相等时不更新 argmax（保证确定性）
    - Phase 3: 对未被 indices 覆盖的位置（argmax 仍为 M），将 y 改写为 0（argmax 保持 M 不变）
    - 最终未覆盖位置输出：y=0, argmax=M

    Args:
        x: np.ndarray, float32, shape [N, D1, D2, ...]，仅用于确定输出 shape，不参与计算。
        indices: np.ndarray, int32, shape [M]，scatter 索引，取值范围 [0, N)。
        updates: np.ndarray, float32, shape [M, D1, D2, ...]，scatter 源数据。
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        tuple: (y, argmax)
            y: np.ndarray, float32, shape [N, D1, D2, ...]，scatter max 结果，未覆盖位置为 0。
            argmax: np.ndarray, int32, shape [N, D1, D2, ...]，最大值来源索引，未覆盖位置为 M。
    """
    return _scatter_max_with_argmax_numpy(x, indices, updates)


def aclnn_scatter_max_with_argmax_golden(x, indices, updates, y, argmax, **kwargs):
    """
    Aclnn golden for aclnnScatterMaxWithArgmax.
    All the parameters (name & order) follow
        function `aclnnScatterMaxWithArgmaxGetWorkspaceSize` in @aclnn_scatter_max_with_argmax.h
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch,
        the Tensors in the parameters are all torch.Tensor.
    Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        x: 输出 shape 参考张量，float32，不参与计算。
        indices: scatter 索引张量，int32，取值范围 [0, N)。
        updates: scatter 源数据张量，float32。
        y: 输出张量（用于确定 dtype/shape），float32。
        argmax: 输出张量（用于确定 dtype/shape），int32。
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        tuple: (y_result, argmax_result)
    """

    # 统一转为 numpy.ndarray 进行计算
    def _to_numpy(t):
        if isinstance(t, np.ndarray):
            return t
        try:
            return t.numpy()
        except AttributeError:
            return np.asarray(t)

    y_np, argmax_np = _scatter_max_with_argmax_numpy(
        _to_numpy(x), _to_numpy(indices), _to_numpy(updates)
    )
    return y_np, argmax_np
