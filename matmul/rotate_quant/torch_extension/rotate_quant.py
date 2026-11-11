# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


# Native PyTorch ScalarType IDs accepted by the int? dst_dtype schema.
_INT8 = 1
_INT4 = 16
_FP8_E5M2 = 23
_FP8_E4M3FN = 24
# torch_npu extended DType IDs use the ACL dtype ID plus 256.
_NPU_FP8_E5M2 = 291
_NPU_FP8_E4M3FN = 292
_FP4 = 296
_MX_SCALE_DTYPE = getattr(torch, "float8_e8m0fnu", None)

_MIN_INPUT_RANK = 1
_MAX_INPUT_RANK = 7
_MATRIX_RANK = 2
_BATCHED_MATRIX_RANK = 3
_GROUPED_BATCHED_MATRIX_RANK = 4
_MIN_GROUP_NUM = 1
_MAX_GROUP_NUM = 1024
_GROUP_LIST_TYPE_CUMSUM = 0
_GROUP_LIST_TYPE_COUNT = 1
_SUPPORTED_MX_ROT_K_SIZES = (32, 64, 128, 256)
_SCALE_ALG_OCP = 0
_SCALE_ALG_CUBLAS = 1
_SCALE_ALG_DYNAMIC = 2
_MIN_DST_TYPE_MAX = 6.0
_MAX_DST_TYPE_MAX = 12.0
_INT4_VALUES_PER_INT32 = 8
_FP4_VALUES_PER_BYTE = 2
# Two E8M0 scale codes describe a pair of 32-element MX blocks.
_MX_SCALE_PAIR_SIZE = 64
_MX_SCALES_PER_PAIR = 2

_OUTPUT_DTYPES = {
    _INT8: torch.int8,
    _INT4: torch.int32,
    _FP8_E5M2: torch.float8_e5m2,
    _FP8_E4M3FN: torch.float8_e4m3fn,
    _NPU_FP8_E5M2: torch.float8_e5m2,
    _NPU_FP8_E4M3FN: torch.float8_e4m3fn,
    _FP4: torch.uint8,
}


def _check_params(
    x,
    rotation,
    alpha,
    dst_dtype,
    axis,
    round_mode,
    scale_alg,
    dst_type_max,
    transpose_y,
    group_list,
    group_list_type,
):
    """Check metadata only; device group sizes and platform limits are checked by ACLNN/kernel."""
    torch._check(
        _MIN_INPUT_RANK <= x.dim() <= _MAX_INPUT_RANK,
        lambda: "x must have between 1 and 7 dimensions",
    )
    torch._check(
        x.dtype in (torch.float16, torch.bfloat16),
        lambda: "x dtype must be float16 or bfloat16",
    )
    torch._check(rotation.dtype == x.dtype, lambda: "rotation dtype must match x dtype")
    torch._check(
        dst_dtype in _OUTPUT_DTYPES, lambda: "unsupported dst_dtype for rotate_quant"
    )
    torch._check(
        axis in (-1, x.dim() - 1), lambda: "axis must be -1 or the last dimension index"
    )
    torch._check(not transpose_y, lambda: "transpose_y only supports False")
    torch._check(round_mode in ("rint", "round", "floor"), lambda: "invalid round_mode")
    torch._check(
        group_list_type in (_GROUP_LIST_TYPE_CUMSUM, _GROUP_LIST_TYPE_COUNT),
        lambda: "group_list_type must be 0 (cumsum) or 1 (count)",
    )

    is_mx = dst_dtype not in (_INT8, _INT4)
    if group_list is None:
        torch._check(
            rotation.dim() in (_MATRIX_RANK, _BATCHED_MATRIX_RANK)
            if is_mx
            else rotation.dim() == _MATRIX_RANK,
            lambda: "ordinary rotation must be [K,K] or, for MX output, [N/K,K,K]",
        )
    else:
        torch._check(
            is_mx, lambda: "grouped rotate_quant only supports MXFP4/MXFP8 output"
        )
        torch._check(
            x.dim() == _MATRIX_RANK, lambda: "grouped x must be 2-dimensional [M,N]"
        )
        torch._check(
            rotation.dim() in (_BATCHED_MATRIX_RANK, _GROUPED_BATCHED_MATRIX_RANK),
            lambda: "grouped rotation must be [E,K,K] or [E,N/K,K,K]",
        )
        torch._check(
            group_list.dim() == 1 and group_list.dtype == torch.int64,
            lambda: "group_list must be a 1D int64 tensor",
        )
        torch._check(
            _MIN_GROUP_NUM <= group_list.shape[0] <= _MAX_GROUP_NUM,
            lambda: "group count must be in [1,1024]",
        )
        torch._check(
            rotation.shape[0] == group_list.shape[0],
            lambda: "rotation group count must equal group_list length",
        )
        torch._check(x.shape[-1] > 0, lambda: "grouped N must be positive")

    n, k = x.shape[-1], rotation.shape[-1]
    torch._check(k > 0, lambda: "rotation K must be positive")
    torch._check(
        rotation.shape[-2] == k, lambda: "rotation last two dimensions must be [K,K]"
    )
    torch._check(n % k == 0, lambda: "x last dimension must be divisible by K")
    if is_mx:
        torch._check(
            _MX_SCALE_DTYPE is not None,
            lambda: "MX output requires torch.float8_e8m0fnu support",
        )
        torch._check(
            k in _SUPPORTED_MX_ROT_K_SIZES,
            lambda: "MX rotation K must be 32, 64, 128 or 256",
        )
        if (
            group_list is None and rotation.dim() == _BATCHED_MATRIX_RANK
        ) or rotation.dim() == _GROUPED_BATCHED_MATRIX_RANK:
            torch._check(
                rotation.shape[-3] == n // k,
                lambda: "rotation N-segment dimension must equal N/K",
            )
        allowed_algs = (
            (_SCALE_ALG_OCP, _SCALE_ALG_DYNAMIC)
            if dst_dtype == _FP4
            else (_SCALE_ALG_OCP, _SCALE_ALG_CUBLAS)
        )
        torch._check(
            scale_alg in allowed_algs,
            lambda: "scale_alg is not supported for dst_dtype",
        )
        if dst_dtype != _FP4:
            torch._check(
                round_mode == "rint", lambda: "MXFP8 output only supports rint"
            )
    else:
        torch._check(
            x.dim() == _MATRIX_RANK, lambda: "INT8/INT4 x must be 2-dimensional [M,N]"
        )
        torch._check(
            n % _INT4_VALUES_PER_INT32 == 0,
            lambda: "INT8/INT4 N must be divisible by 8",
        )
        torch._check(
            scale_alg == _SCALE_ALG_OCP, lambda: "INT8/INT4 only supports scale_alg=0"
        )
        torch._check(alpha is None, lambda: "alpha must be None for INT8/INT4 output")

    if alpha is not None:
        torch._check(
            alpha.dtype == torch.bfloat16 and alpha.dim() == 1 and alpha.shape[0] == 1,
            lambda: "alpha must be a bfloat16 tensor of shape [1]",
        )
    torch._check(
        _MIN_DST_TYPE_MAX <= dst_type_max <= _MAX_DST_TYPE_MAX
        if scale_alg == _SCALE_ALG_DYNAMIC
        else dst_type_max == 0.0,
        lambda: "dst_type_max must be in [6,12] for scale_alg=2, otherwise 0",
    )


class RotateQuantOpBuilder(OpBuilder):
    """Build the eager RotateQuant binding for ordinary and M-axis grouped rotation."""

    def __init__(self):
        super().__init__("rotate_quant")

    def sources(self) -> list:
        return [self.resolve_source("rotate_quant.cpp")]

    def schema(self) -> str:
        return (
            "rotate_quant(Tensor x, Tensor rotation, *, Tensor? alpha=None, "
            "int? dst_dtype=None, int? axis=-1, str? round_mode='rint', "
            "int? scale_alg=0, float? dst_type_max=0.0, bool? transpose_y=False, "
            "Tensor? group_list=None, int group_list_type=0) -> (Tensor y, Tensor scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def rotate_quant_meta(
            x: torch.Tensor,
            rotation: torch.Tensor,
            *,
            alpha: Optional[torch.Tensor] = None,
            dst_dtype: Optional[int] = None,
            axis: Optional[int] = -1,
            round_mode: Optional[str] = "rint",
            scale_alg: Optional[int] = 0,
            dst_type_max: Optional[float] = 0.0,
            transpose_y: Optional[bool] = False,
            group_list: Optional[torch.Tensor] = None,
            group_list_type: int = 0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            dst_dtype = _INT8 if dst_dtype is None else dst_dtype
            _check_params(
                x,
                rotation,
                alpha,
                dst_dtype,
                -1 if axis is None else axis,
                "rint" if round_mode is None else round_mode,
                0 if scale_alg is None else scale_alg,
                0.0 if dst_type_max is None else dst_type_max,
                False if transpose_y is None else transpose_y,
                group_list,
                group_list_type,
            )
            out_shape = list(x.shape)
            if dst_dtype == _INT4:
                out_shape[-1] //= _INT4_VALUES_PER_INT32
            elif dst_dtype == _FP4:
                out_shape[-1] //= _FP4_VALUES_PER_BYTE
            if dst_dtype in (_INT8, _INT4):
                scale_shape, scale_dtype = [x.shape[0]], torch.float32
            else:
                scale_shape = list(x.shape[:-1]) + [
                    (x.shape[-1] + _MX_SCALE_PAIR_SIZE - 1) // _MX_SCALE_PAIR_SIZE,
                    _MX_SCALES_PER_PAIR,
                ]
                scale_dtype = _MX_SCALE_DTYPE
            return (
                x.new_empty(out_shape, dtype=_OUTPUT_DTYPES[dst_dtype]),
                x.new_empty(scale_shape, dtype=scale_dtype),
            )


rotate_quant_builder = RotateQuantOpBuilder()
rotate_quant_builder._ensure_initialized()


@impl(get_as_library(), rotate_quant_builder.name, "PrivateUse1")
def rotate_quant(
    x: torch.Tensor,
    rotation: torch.Tensor,
    *,
    alpha: Optional[torch.Tensor] = None,
    dst_dtype: Optional[int] = None,
    axis: Optional[int] = -1,
    round_mode: Optional[str] = "rint",
    scale_alg: Optional[int] = 0,
    dst_type_max: Optional[float] = 0.0,
    transpose_y: Optional[bool] = False,
    group_list: Optional[torch.Tensor] = None,
    group_list_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate the last-dimension K-sized segments and quantize on NPU (eager only).

    Args:
        x: float16/bfloat16 input, [*, N]; grouped input must be [M, N].
        rotation: Same dtype as x. Ordinary: [K, K] or [N/K, K, K].
            Grouped: [E, K, K] or [E, N/K, K, K], with 1 <= E <= 1024.
        alpha: Optional bfloat16 [1] clipping factor for MX output.
        dst_dtype: INT8 (default), packed INT4, MXFP4 or MXFP8 target enum.
            FP4 uses torch_npu.float4_e2m1fn_x2.
            Ascend 950 requires an explicit MX target; A2/A3 support INT8/INT4.
        axis: -1 or x.dim()-1, default -1.
        round_mode: 'rint', 'round' or 'floor'; MXFP8 requires 'rint'.
        scale_alg: INT8/INT4: 0; MXFP4: 0 or 2; MXFP8: 0 or 1.
        dst_type_max: In [6, 12] for scale_alg=2, otherwise 0.
        transpose_y: Only False is supported.
        group_list: Optional NPU int64 [E] M-axis partition, Ascend 950 only.
            Empty groups and partial coverage of M are supported.
        group_list_type: 0 for cumulative row counts, 1 for row counts.

    Returns:
        (y, scale) on x.device. INT4 uses int32 [M, N/8]; FP4 uses packed
        uint8 [*, N/2] and requires unpacking. Other targets retain x.shape.
        INT scale is float32 [M]; MX scale is native float8_e8m0fnu [*, ceil(N/64), 2].
        Uncovered grouped rows have unspecified contents.
        See docs/torchapi_rotate_quant.md for limits.
    """
    _check_params(
        x,
        rotation,
        alpha,
        _INT8 if dst_dtype is None else dst_dtype,
        -1 if axis is None else axis,
        "rint" if round_mode is None else round_mode,
        0 if scale_alg is None else scale_alg,
        0.0 if dst_type_max is None else dst_type_max,
        False if transpose_y is None else transpose_y,
        group_list,
        group_list_type,
    )
    return rotate_quant_builder.load().rotate_quant(
        x,
        rotation,
        alpha,
        dst_dtype,
        axis,
        round_mode,
        scale_alg,
        dst_type_max,
        transpose_y,
        group_list,
        group_list_type,
    )
