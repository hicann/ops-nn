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

import numpy as np

__input__ = {
    "kernel": {"quant_update_scatter": "quant_update_scatter_input"},
    "aclnn": {
        "aclnnInplaceQuantScatter": "aclnn_quant_scatter_input",
        "aclnnInplaceQuantScatterV2": "aclnn_quant_scatter_v2_input",
    },
}


def quant_update_scatter_input(
    var,
    indices,
    updates,
    quant_scales,
    quant_zero_points=None,
    *,
    reduce,
    axis=-2,
    quant_axis=-1,
    reciprocal_scale=False,
    round_mode="rint",
    **kwargs,
):
    """
    Input function for quant_update_scatter.
    All the parameters (names and order) follow @quant_update_scatter_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    shape, dtype = indices.shape, indices.dtype
    if len(indices.shape) == 1:
        indices = np.random.uniform(
            0, var.shape[axis] - updates.shape[axis], shape
        ).astype(dtype)
    else:
        indices = []
        batch_indices_map = {}
        while True:
            batch = np.random.uniform(0, var.shape[0], (1,)).astype(dtype).item()
            if batch in batch_indices_map.keys():
                available_range = set(range(1, var.shape[axis] - updates.shape[axis]))
                for indices_start in batch_indices_map[batch]:
                    available_range -= set(
                        range(
                            indices_start - updates.shape[axis],
                            indices_start + updates.shape[axis],
                        )
                    )
                if len(available_range) > 0:
                    indices_random = np.random.choice(
                        np.array(list(available_range), dtype=object)
                    )
                    batch_indices_map[batch].append(indices_random)
                else:
                    continue
            else:
                indices_random = np.random.uniform(
                    0, var.shape[axis] - updates.shape[axis], (1,)
                ).astype(dtype)
                batch_indices_map[batch] = [indices_random.item()]
            indices.append(batch)
            if not isinstance(indices_random, int):
                indices_random = indices_random.tolist()[0]
            indices.append(indices_random)
            if len(indices) == shape[0] * 2:
                break
        indices = np.reshape(indices, (shape[0], 2))
    indices = indices.astype(dtype, copy=False)
    return [var, indices, updates, quant_scales, quant_zero_points]


def aclnn_quant_scatter_input(*args, **kwargs):
    """
    Input function for aclnnInplaceQuantScatter.
    TTK passes all tensor args; process indices.
    """
    import torch
    import numpy as np

    var = args[0] if len(args) > 0 else kwargs.get("selfRef")
    indices = args[1] if len(args) > 1 else kwargs.get("indices")
    updates = args[2] if len(args) > 2 else kwargs.get("updates")
    quant_scales = args[3] if len(args) > 3 else kwargs.get("quantScales")
    quant_zero_points = args[4] if len(args) > 4 else kwargs.get("quantZeroPoints")

    axis = kwargs.get("attributes", {}).get("axis", -2)

    if indices is not None:
        import torch
        import numpy as np

        def _to_np(t):
            if t is None:
                return None
            if isinstance(t, np.ndarray):
                return t
            if not hasattr(t, "dtype"):
                return np.array(t)
            dt = t.dtype
            dt_str = str(dt)
            if dt == torch.bfloat16:
                return t.to(torch.float32).numpy()
            if "float8" in dt_str or "hifloat" in dt_str or "HiFloat" in dt_str:
                try:
                    return t.view(torch.uint8).numpy()
                except (TypeError, RuntimeError):
                    try:
                        return t.to(torch.uint8).numpy()
                    except Exception:
                        return np.frombuffer(t.numpy(), dtype=np.uint8).reshape(t.shape)
            try:
                return t.numpy()
            except (TypeError, RuntimeError):
                return t.to(torch.uint8).numpy()

        var_np = _to_np(var)
        upd_np = _to_np(updates)
        idx_np = _to_np(indices)
        shape, dtype = idx_np.shape, idx_np.dtype
        if len(idx_np.shape) == 1:
            idx_np = np.random.uniform(
                0, var_np.shape[axis] - upd_np.shape[axis], shape
            ).astype(dtype)
        else:
            idx_list = []
            batch_map = {}
            while True:
                batch = np.random.uniform(0, var_np.shape[0], (1,)).astype(dtype).item()
                if batch in batch_map:
                    avail = set(range(1, var_np.shape[axis] - upd_np.shape[axis]))
                    for s in batch_map[batch]:
                        avail -= set(
                            range(s - upd_np.shape[axis], s + upd_np.shape[axis])
                        )
                    if len(avail) > 0:
                        idx_r = np.random.choice(np.array(list(avail), dtype=object))
                        batch_map[batch].append(idx_r)
                    else:
                        continue
                else:
                    idx_r = np.random.uniform(
                        0, var_np.shape[axis] - upd_np.shape[axis], (1,)
                    ).astype(dtype)
                    batch_map[batch] = [idx_r.item()]
                idx_list.append(batch)
                if not isinstance(idx_r, int):
                    idx_r = idx_r.tolist()[0]
                idx_list.append(idx_r)
                if len(idx_list) == shape[0] * 2:
                    break
            idx_np = np.reshape(idx_list, (shape[0], 2)).astype(dtype)
        indices = torch.from_numpy(idx_np.astype(dtype))

    return [var, indices, updates, quant_scales, quant_zero_points]


def aclnn_quant_scatter_v2_input(*args, **kwargs):
    """
    Input function for aclnnInplaceQuantScatterV2.
    Same logic as V1.
    """
    return aclnn_quant_scatter_input(*args, **kwargs)
