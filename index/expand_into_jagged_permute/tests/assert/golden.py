#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FBGEMM reference for ExpandIntoJaggedPermute.

Nonempty inputs are validated against the shared FBGEMM safety contract before
calling the native operator. The P=0 result is the independent Ascend
empty-output contract because FBGEMM rejects it.
"""

from typing import Any, Tuple

import numpy as np
import torch

__spec__ = {
    "expand_into_jagged_permute": "ExpandIntoJaggedPermuteSpec",
    "aclnnExpandIntoJaggedPermute": "AclnnExpandIntoJaggedPermuteSpec",
}

_EXCEPTION_CASE_MARKERS = (
    "_L2_",
    "_SUPP_rank9_permute",
    "_SUPP_output_size_negative",
    "_SUPP_output_size_over_int32_small",
    "_SUPP_p0_nonzero_output",
    "_SUPP_output_zero_stride",
)
_INT32_MAX = np.iinfo(np.int32).max
_VALIDATION_CHUNK_ELEMS = 1 << 20


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _as_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    return torch.from_numpy(np.ascontiguousarray(np.asarray(value)))


def _load_fbgemm() -> None:
    try:
        import fbgemm_gpu  # noqa: F401 -- registers torch.ops.fbgemm
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "ExpandIntoJaggedPermute nonempty Golden requires an fbgemm-gpu-cpu/fbgemm-gpu "
            "package compatible with the active PyTorch; no formula fallback is allowed"
        ) from exc


def _validate_fbgemm_inputs(
    permute: torch.Tensor,
    input_offsets: torch.Tensor,
    output_offsets: torch.Tensor,
    output_size: int,
) -> None:
    for name, value in (
        ("permute", permute),
        ("input_offsets", input_offsets),
        ("output_offsets", output_offsets),
    ):
        if value.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got rank {value.ndim}")
        if value.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype int32, got {value.dtype}")

    segment_count = permute.numel()
    expected_offset_count = segment_count + 1
    if input_offsets.numel() != expected_offset_count:
        raise ValueError(
            "len(input_offsets) must equal len(permute) + 1, "
            f"got {input_offsets.numel()} and {segment_count}"
        )
    if output_offsets.numel() != expected_offset_count:
        raise ValueError(
            "len(output_offsets) must equal len(permute) + 1, "
            f"got {output_offsets.numel()} and {segment_count}"
        )
    if output_size < 0 or output_size > _INT32_MAX:
        raise ValueError(f"output_size must be in [0, INT32_MAX], got {output_size}")
    if segment_count == 0:
        if output_size != 0:
            raise ValueError("P == 0 requires output_size == 0")
    elif output_size < segment_count:
        raise ValueError(
            f"P > 0 requires output_size >= P, got P={segment_count}, "
            f"output_size={output_size}"
        )

    if input_offsets[0].item() != 0 or output_offsets[0].item() != 0:
        raise ValueError("input_offsets and output_offsets must start at 0")
    if output_offsets[-1].item() != output_size:
        raise ValueError("output_offsets[-1] must equal output_size")

    if segment_count == 0:
        return
    if torch.any(input_offsets[1:] <= input_offsets[:-1]).item():
        raise ValueError("input_offsets must be strictly increasing")
    if torch.any(output_offsets[1:] <= output_offsets[:-1]).item():
        raise ValueError("output_offsets must be strictly increasing")
    if torch.any((permute < 0) | (permute >= segment_count)).item():
        raise ValueError("permute values must be in [0, P)")

    for start in range(0, segment_count, _VALIDATION_CHUNK_ELEMS):
        end = min(start + _VALIDATION_CHUNK_ELEMS, segment_count)
        source_indices = permute[start:end].to(torch.int64)
        source_begins = input_offsets.index_select(0, source_indices).to(torch.int64)
        target_lengths = output_offsets[start + 1 : end + 1].to(
            torch.int64
        ) - output_offsets[start:end].to(torch.int64)
        if torch.any(source_begins + target_lengths - 1 > _INT32_MAX).item():
            raise OverflowError("generated index exceeds INT32_MAX")


def _compute_fbgemm(
    permute: Any,
    input_offsets: Any,
    output_offsets: Any,
    output_size: Any,
) -> np.ndarray:
    permute_tensor = _as_cpu_tensor(permute)
    input_offsets_tensor = _as_cpu_tensor(input_offsets)
    output_offsets_tensor = _as_cpu_tensor(output_offsets)
    output_size = _as_int(output_size)
    _validate_fbgemm_inputs(
        permute_tensor, input_offsets_tensor, output_offsets_tensor, output_size
    )

    if permute_tensor.numel() == 0:
        return torch.empty((0,), dtype=torch.int32).numpy()

    _load_fbgemm()
    result = torch.ops.fbgemm.expand_into_jagged_permute(
        permute_tensor,
        input_offsets_tensor,
        output_offsets_tensor,
        output_size,
    )
    return result.detach().cpu().numpy()


def _first_range(
    input_ranges: Any, index: int, fallback: Tuple[int, int]
) -> Tuple[int, int]:
    try:
        value = input_ranges[index]
    except (IndexError, TypeError):
        return fallback
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return fallback
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return fallback


def _build_valid_inputs(
    segment_count: int,
    output_size: int,
    permute_bounds: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if segment_count == 0:
        return (
            torch.empty((0,), dtype=torch.int32).numpy(),
            torch.zeros((1,), dtype=torch.int32).numpy(),
            torch.zeros((1,), dtype=torch.int32).numpy(),
        )
    lower = max(0, min(segment_count - 1, permute_bounds[0]))
    upper = max(lower, min(segment_count - 1, permute_bounds[1]))
    width = upper - lower + 1
    permute = lower + ((torch.arange(segment_count, dtype=torch.int64) + 1) % width)

    target_base, target_remainder = divmod(output_size, segment_count)
    target_lengths = torch.full((segment_count,), target_base, dtype=torch.int64)
    target_lengths[:target_remainder] += 1
    output_offsets = torch.empty((segment_count + 1,), dtype=torch.int64)
    output_offsets[0] = 0
    output_offsets[1:] = torch.cumsum(target_lengths, dim=0)

    # Unit source segments deliberately exercise that the next source offset is
    # not a generation bound for the corresponding target segment.
    input_offsets = torch.arange(segment_count + 1, dtype=torch.int64)
    return (
        permute.to(torch.int32).numpy(),
        input_offsets.to(torch.int32).numpy(),
        output_offsets.to(torch.int32).numpy(),
    )


def _copy_values(destination: Any, values: np.ndarray) -> None:
    if isinstance(destination, np.ndarray):
        source = np.asarray(values, dtype=destination.dtype)
        if source.shape != destination.shape:
            raise ValueError(
                f"shape mismatch: destination={destination.shape}, "
                f"source={source.shape}"
            )
        if (
            destination.ndim == 1
            and destination.size > 1
            and destination.strides[0] == 0
        ):
            if not np.all(source == source.flat[0]):
                raise ValueError(
                    "cannot write distinct values to a multi-element zero-stride view"
                )
            destination[0] = source.flat[0]
            return
        destination[...] = source
        return
    if isinstance(destination, torch.Tensor):
        source = torch.as_tensor(
            values, dtype=destination.dtype, device=destination.device
        )
        if tuple(source.shape) != tuple(destination.shape):
            raise ValueError(
                f"shape mismatch: destination={tuple(destination.shape)}, "
                f"source={tuple(source.shape)}"
            )
        if (
            destination.ndim == 1
            and destination.numel() > 1
            and destination.stride(0) == 0
        ):
            if not torch.equal(source, source[0].expand_as(source)):
                raise ValueError(
                    "cannot write distinct values to a multi-element zero-stride view"
                )
            destination[0].copy_(source[0])
            return
        destination.copy_(source)
        return
    raise TypeError(f"unsupported input container: {type(destination).__name__}")


def _should_customize_inputs(testcase_name: Any) -> bool:
    name = str(testcase_name or "")
    return not any(marker in name for marker in _EXCEPTION_CASE_MARKERS)


def _customize_correlated_inputs(
    permute: Any,
    input_offsets: Any,
    output_offsets: Any,
    output_size: Any,
    input_ranges: Any,
) -> Tuple[Any, Any, Any]:
    segment_count = int(permute.shape[0])
    bounds = _first_range(input_ranges, 0, (0, max(0, segment_count - 1)))
    values = _build_valid_inputs(segment_count, _as_int(output_size), bounds)
    _copy_values(permute, values[0])
    _copy_values(input_offsets, values[1])
    _copy_values(output_offsets, values[2])
    return permute, input_offsets, output_offsets


def _fbgemm_third_party(permute, input_offsets, output_offsets, output_size):
    """Run the official FBGEMM implementation on the provider device."""
    output_size = _as_int(output_size)
    if permute.numel() == 0 and output_size == 0:
        return torch.empty((0,), dtype=permute.dtype, device=permute.device)
    _load_fbgemm()
    return torch.ops.fbgemm.expand_into_jagged_permute(
        permute,
        input_offsets,
        output_offsets,
        output_size,
    )


class ExpandIntoJaggedPermuteThirdParty:
    def __init__(self, *, output_size, **kwargs):
        del kwargs
        self.output_size = output_size

    def __call__(self, permute, input_offsets, output_offsets, **kwargs):
        del kwargs
        return _fbgemm_third_party(
            permute, input_offsets, output_offsets, self.output_size
        )


class AclnnExpandIntoJaggedPermuteThirdParty:
    def __init__(self, outputSize, **kwargs):
        del kwargs
        self.output_size = outputSize

    def __call__(self, permute, inputOffsets, outputOffsets, **kwargs):
        del kwargs
        return _fbgemm_third_party(
            permute, inputOffsets, outputOffsets, self.output_size
        )


class ExpandIntoJaggedPermuteSpec:
    third_party = {"torch": ExpandIntoJaggedPermuteThirdParty}
    tolerance = {"int32": {"standard": "binary_equal"}}

    @staticmethod
    def golden(
        permute,
        input_offsets,
        output_offsets,
        *,
        output_size,
        **kwargs,
    ):
        del kwargs
        return [_compute_fbgemm(permute, input_offsets, output_offsets, output_size)]

    @staticmethod
    def customize_inputs(
        permute,
        input_offsets,
        output_offsets,
        *,
        output_size,
        input_ranges=None,
        testcase_name="",
        **kwargs,
    ):
        del kwargs
        if not _should_customize_inputs(testcase_name):
            return permute, input_offsets, output_offsets
        return _customize_correlated_inputs(
            permute, input_offsets, output_offsets, output_size, input_ranges
        )


class AclnnExpandIntoJaggedPermuteSpec:
    third_party = {"torch": AclnnExpandIntoJaggedPermuteThirdParty}
    tolerance = {"int32": {"standard": "binary_equal"}}

    @staticmethod
    def golden(
        permute,
        inputOffsets,
        outputOffsets,
        outputSize,
        out,
        **kwargs,
    ):
        del kwargs
        output_size = _as_int(outputSize)
        result = _compute_fbgemm(permute, inputOffsets, outputOffsets, output_size)
        _copy_values(out, result)
        return [out]

    @staticmethod
    def customize_inputs(
        permute,
        inputOffsets,
        outputOffsets,
        outputSize,
        out,
        input_ranges=None,
        testcase_name="",
        **kwargs,
    ):
        del out, kwargs
        if not _should_customize_inputs(testcase_name):
            return None
        _customize_correlated_inputs(
            permute, inputOffsets, outputOffsets, outputSize, input_ranges
        )
        return None

    @staticmethod
    def pre_compare(npu_output, golden_output):
        return np.ascontiguousarray(npu_output), np.ascontiguousarray(golden_output)
