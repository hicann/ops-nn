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
"""Kernel-level golden for TransposeBatchMatMul.

op: transpose_batch_mat_mul   formula: y = perm_y(perm_x1(x1) @ perm_x2(x2))

Kernel golden receives numpy.ndarray inputs/outputs.  The FRACTAL_NZ layout of
x2 (when input_formats[1] == 'FRACTAL_NZ') is converted back to logical ND via
``matmul_golden_util.nz_to_nd`` before the matmul.  HF32 execution is simulated
when ``enable_hf32`` is True.  Scale (INT64/UINT64) is decoded via
``matmul_golden_util.u64_to_deq_scale`` before quantization.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util


def _kernel_compute(
    x1,
    x2,
    bias=None,
    scale=None,
    *,
    perm_x1=(0, 1, 2),
    perm_x2=(0, 1, 2),
    perm_y=(1, 0, 2),
    enable_hf32=False,
    batch_split_factor=1,
    **kwargs,
):
    """Core transpose_batch_mat_mul numpy simulation.  x1 and x2 are already in ND format."""
    x1_dtype = x1.dtype

    if enable_hf32 and x1_dtype == np.float32:
        x1 = _util.hf32_truncate_np(x1)
        x2 = _util.hf32_truncate_np(x2)

    if x1_dtype in (np.float16, _util.np_bfloat16):
        x1 = x1.astype(np.float32)
        x2 = x2.astype(np.float32)
        comp_dtype = np.float32
    else:
        x1 = x1.astype(np.float64)
        x2 = x2.astype(np.float64)
        comp_dtype = np.float64

    x1_t = np.transpose(x1, axes=list(perm_x1))

    x2_t = np.transpose(x2, axes=list(perm_x2))

    mm_out = np.matmul(x1_t, x2_t)
    if bias is not None:
        # Apply the shared logical-N bias before output permutation, batch split or quantization.
        mm_out = mm_out + bias.astype(comp_dtype).reshape(1, 1, x2_t.shape[-1])

    if scale is not None:
        if scale.dtype in (np.int64, np.uint64):
            scale = _util.u64_to_deq_scale(scale)
        mm_out = np.transpose(mm_out, axes=list(perm_y))
        M, B, N = mm_out.shape
        mm_out = mm_out.reshape(M, B * N)
        scale_f = scale.astype(comp_dtype).reshape(1, B * N)
        mm_out = mm_out * scale_f
        mm_out = np.sign(mm_out) * np.floor(np.abs(mm_out) + 0.5)
        mm_out = mm_out.clip(-128, 127).astype(np.int8)
        mm_out = mm_out.reshape(M, 1, B * N)
    elif batch_split_factor > 1:
        B, M, N = mm_out.shape
        inner_batch = B // batch_split_factor
        mm_out = mm_out.reshape(batch_split_factor, inner_batch, M, N)
        mm_out = mm_out.transpose(0, 2, 1, 3)
        mm_out = mm_out.reshape(batch_split_factor, M, inner_batch * N)
    else:
        mm_out = np.transpose(mm_out, axes=list(perm_y))

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        mm_out = _util.cast_output_dtype(mm_out, output_dtypes[0])

    return [mm_out]


def _npu_compute(
    x1,
    x2,
    bias=None,
    scale=None,
    *,
    perm_x1=(0, 1, 2),
    perm_x2=(0, 1, 2),
    perm_y=(1, 0, 2),
    enable_hf32=False,
    batch_split_factor=1,
    device=None,
    **kwargs,
):
    """Profile no-bias TBMM and a separate bias Add as individual operators."""
    if scale is not None:
        # TTK falls back to the existing CPU golden for quantized cases.
        raise NotImplementedError("NPU TBMM+Add profiling requires scale=None")
    import torch

    formats = kwargs.get("input_formats", kwargs.get("tensor_formats", ()))
    is_weight_nz = len(formats) > 1 and formats[1] == "FRACTAL_NZ"
    if is_weight_nz:
        import torch_npu

        if isinstance(x2, np.ndarray):
            # Kernel inputs contain physical NZ data; restore the logical ND shape.
            ori_shapes = kwargs.get("input_ori_shapes", ())
            ori_shape = ori_shapes[1] if len(ori_shapes) > 1 else None
            x2 = _util.nz_to_nd(x2, ori_shape)
        elif not kwargs.get("backend") and torch_npu.get_npu_format(x2) != 29:
            # ACLNN inputs carry raw NZ storage; copy the effective storage to CPU.
            offset = x2.storage_offset()
            storage_size = x2.untyped_storage().nbytes() // x2.element_size() - offset
            storage = x2.as_strided((storage_size,), (1,), offset).cpu()
            x2_host = storage.as_strided(x2.shape, x2.stride(), 0)
            x2 = _util.nz_input_to_nd(x2_host, formats, 1)
        # E2E inputs are logical ND tensors; the framework casts only the DUT side.

    inputs = []
    for tensor in (x1, x2, bias):
        if isinstance(tensor, np.ndarray):
            dtype = _util.DTypeTable.get_torch_dtype(tensor.dtype)
            if dtype == torch.bfloat16:
                tensor = tensor.astype(np.float32)
            tensor = torch.from_numpy(tensor).to(dtype=dtype)
        inputs.append(tensor.to(device) if tensor is not None else None)
    x1, x2, bias = inputs
    if is_weight_nz and torch_npu.get_npu_format(x2) != 29:
        x2 = torch_npu.npu_format_cast(x2, 29)

    import cann_ops_nn  # noqa: F401

    y = torch.ops.cann_ops_nn.transpose_batch_mat_mul(
        x1,
        x2,
        bias=None,
        scale=None,
        perm_x1=list(perm_x1),
        perm_x2=list(perm_x2),
        perm_y=list(perm_y),
        enable_hf32=enable_hf32,
        batch_split_factor=batch_split_factor,
    )
    output_shape = y.shape
    if batch_split_factor > 1:
        # Restore N for shared bias broadcasting over the original batches.
        batch = x1.shape[perm_x1[0]]
        m = x1.shape[perm_x1[1]]
        n = x2.shape[perm_x2[2]]
        y = y.reshape(batch_split_factor, m, batch // batch_split_factor, n)
    output_dtypes = kwargs.get("output_dtypes")
    output_dtype = (
        _util.DTypeTable.get_torch_dtype(output_dtypes[0]) if output_dtypes else y.dtype
    )
    if bias is not None:
        y = torch.add(y, bias, out=torch.empty_like(y, dtype=output_dtype))
    elif y.dtype != output_dtype:
        y = y.to(output_dtype)
    # BF16 conversion stays on CPU, without an extra NPU Cast for NumPy output.
    return [_util.torch_to_numpy(y.reshape(output_shape))]


class TransposeBatchMatMulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        scale=None,
        *,
        perm_x1=(0, 1, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        enable_hf32=False,
        batch_split_factor=1,
        **kwargs,
    ):
        """Kernel golden: NZ->ND conversion + core matmul computation."""
        if str(kwargs.get("device", "cpu")).startswith("npu"):
            return _npu_compute(
                x1,
                x2,
                bias,
                scale,
                perm_x1=perm_x1,
                perm_x2=perm_x2,
                perm_y=perm_y,
                enable_hf32=enable_hf32,
                batch_split_factor=batch_split_factor,
                **kwargs,
            )
        input_formats = kwargs.get("input_formats", ())
        input_ori_shapes = kwargs.get("input_ori_shapes", ())

        if len(input_formats) > 1 and input_formats[1] == "FRACTAL_NZ":
            ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
            if ori_shape is not None and tuple(x2.shape) != tuple(ori_shape):
                x2 = _util.nz_to_nd(x2, ori_shape)

        return _kernel_compute(
            x1,
            x2,
            bias,
            scale,
            perm_x1=perm_x1,
            perm_x2=perm_x2,
            perm_y=perm_y,
            enable_hf32=enable_hf32,
            batch_split_factor=batch_split_factor,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        bias=None,
        scale=None,
        *,
        perm_x1=(0, 1, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        enable_hf32=False,
        batch_split_factor=1,
        **kwargs,
    ):
        """Input preprocessing: scale generation only. NZ->ND is handled in golden()."""
        if scale is not None and scale.dtype in (np.int64, np.uint64):
            scale_shape = scale.shape
            scale_orig_dtype = scale.dtype
            fp32_scale = np.random.uniform(low=-5, high=5, size=scale_shape).astype(
                np.float32
            )
            new_scale = _util.pack_u64_scale(_util.scale_generate(fp32_scale))
            if scale_orig_dtype == np.int64:
                new_scale = new_scale.astype(np.int64)
            scale = new_scale

        return x1, x2, bias, scale


__spec__ = {
    "transpose_batch_mat_mul": "TransposeBatchMatMulTestSpec",
}
