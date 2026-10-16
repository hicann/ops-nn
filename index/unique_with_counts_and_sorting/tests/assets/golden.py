#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Separate values-only kernel and full-output GEIR references in one plugin.

GEIR checks three operator outputs. Unsorted outputs follow first occurrence.
No device result is used by this reference.

Sorted inputs use torch.unique when its CPU dtype is supported. NumPy remains
the fallback for extended unsigned types and first-occurrence unsorted order.
The native third-party baseline supports sorted order only; it performs no
NumPy calculation, dtype conversion or device transfer in the timed call.
"""

import numpy as np
import torch

from ttk.switches import get_global_storage

__spec__ = {
    "unique_with_counts_and_sorting": "UniqueWithCountsAndSortingSpec",
}

_EXACT = {
    dtype: {"standard": "binary_equal"}
    for dtype in (
        "bool",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    )
}


def _as_numpy(x):
    x = np.asarray(x)
    return x.astype(np.float32) if x.dtype.name == "bfloat16" else x


def _unique(x, sorted_flag):
    """Treat equal signed zeros as one value and each NaN as distinct."""
    flat = _as_numpy(x).reshape(-1)
    if sorted_flag and flat.dtype not in (
        np.dtype("uint16"),
        np.dtype("uint32"),
        np.dtype("uint64"),
    ):
        values, inverse, counts = torch.unique(
            torch.from_numpy(flat), sorted=True, return_inverse=True, return_counts=True
        )
        # Equal NaN keys may be permuted by torch's sort. GEIR assigns their
        # distinct inverse indices in input order.
        nan_mask = np.isnan(flat) if flat.dtype.kind == "f" else None
        values, inverse, counts = values.numpy(), inverse.numpy(), counts.numpy()
        if nan_mask is not None and np.any(nan_mask):
            inverse[nan_mask] = np.flatnonzero(np.isnan(values))
        return values, inverse.reshape(np.shape(x)), counts
    values, first, inverse, counts = np.unique(
        flat,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        equal_nan=False,
    )
    if not sorted_flag:
        order = np.argsort(first, kind="stable")
        ranks = np.empty(order.size, dtype=np.int64)
        ranks[order] = np.arange(order.size)
        values = values[order]
        inverse = ranks[inverse.reshape(-1)]
        counts = counts[order]
    return values, inverse.reshape(np.shape(x)), counts


def _customize(x, **kwargs):
    name = kwargs.get("testcase_name", "")
    pattern = None
    if "order_pattern" in name:
        pattern = [3, 1, 3, 2, 1, -1]
    elif "signed_zero" in name:
        pattern = [-0.0, 0.0, -2.0, 1.0, 1.0]
    elif "nan_inf" in name:
        pattern = [np.nan, 2.0, np.inf, -np.inf, np.nan, -0.0, 0.0]
    elif "large_integer" in name:
        pattern = [2**60 + 1, 2**60, 2**60 + 1, 2**60 - 1]
    if pattern is None:
        return [x]
    array = np.resize(np.asarray(pattern, dtype=_as_numpy(x).dtype), np.shape(x))
    array = array.astype(np.asarray(x).dtype)
    return [array]


def _equal(actual, expected):
    actual_dtype = str(actual.dtype).removeprefix("torch.")
    expected_dtype = str(expected.dtype).removeprefix("torch.")
    if actual_dtype != expected_dtype:
        return False
    actual, expected = _as_numpy(actual), _as_numpy(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    equal = actual == expected
    if expected.dtype.kind == "f":
        equal |= np.isnan(actual) & np.isnan(expected)
    return bool(np.all(equal))


def _shape_equal(actual, expected):
    """Compare rank/dimensions; encoding tags and unused padding are not shape."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != np.uint64 or actual.shape != expected.shape or actual.size % 9:
        return False
    for a, e in zip(actual.reshape(-1, 9), expected.reshape(-1, 9)):
        # TTK encodes the uint64 marker in the high bit of the low 32 bits.
        rank, expected_rank = int(a[0]) & 0x7FFFFFFF, int(e[0]) & 0x7FFFFFFF
        if rank != expected_rank or rank > 8:
            return False
        if not np.array_equal(a[1 : rank + 1], e[1 : rank + 1]):
            return False
    return True


class UniqueGeirCompetitor:
    def __init__(
        self, sorted=True, return_inverse=False, return_counts=False, **kwargs
    ):
        if not sorted:
            raise NotImplementedError(
                "Native torch.unique does not preserve GEIR first-occurrence order"
            )
        self.return_inverse = return_inverse
        self.return_counts = return_counts

    def __call__(self, x, **kwargs):
        if x.dtype in (torch.uint16, torch.uint32, torch.uint64):
            raise NotImplementedError(
                "Native torch.unique baseline does not support extended unsigned dtypes"
            )
        result = torch.unique(
            x,
            sorted=True,
            return_inverse=self.return_inverse,
            return_counts=self.return_counts,
        )
        if not self.return_inverse and not self.return_counts:
            return [result, None, None]
        return [
            result[0],
            result[1] if self.return_inverse else None,
            result[-1] if self.return_counts else None,
        ]


class UniqueGeirSpec:
    """Check dynamic output shapes and every enabled payload exactly."""

    customize_inputs = staticmethod(_customize)
    tolerance = _EXACT
    third_party = {"torch": UniqueGeirCompetitor}

    @staticmethod
    def golden(x, *, sorted=True, return_inverse=False, return_counts=False, **kwargs):
        values, inverse, counts = _unique(x, sorted)
        dtypes = kwargs.get("output_dtypes", (np.asarray(x).dtype, "int64", "int64"))
        placeholder = np.ones((1,), dtype=np.int64)
        return [
            values.astype(dtypes[0]),
            inverse.astype(dtypes[1]) if return_inverse else placeholder.copy(),
            counts.astype(dtypes[2]) if return_counts else placeholder.copy(),
        ]

    @staticmethod
    def compare(*outputs_and_goldens, compare_context, **kwargs):
        # TTK appends one uint64 shape-metadata tensor for the enabled
        # output_shape_unknown_indexes declared by the GEIR CSV.
        if len(outputs_and_goldens) != 8:
            raise ValueError(
                "Expected three GEIR outputs plus shape metadata and their goldens"
            )
        return_inverse = compare_context.attributes.get("return_inverse", False)
        return_counts = compare_context.attributes.get("return_counts", False)
        results = []
        for index, enabled in enumerate((True, return_inverse, return_counts)):
            if not enabled:
                results.append({"pass": True, "precision": "N/A (disabled output)"})
                continue
            passed = _equal(outputs_and_goldens[index], outputs_and_goldens[index + 4])
            results.append({"pass": passed, "precision": "100%" if passed else "FAIL"})
        passed = _shape_equal(outputs_and_goldens[3], outputs_and_goldens[7])
        results.append({"pass": passed, "precision": "100%" if passed else "FAIL"})
        return results


class UniqueKernelSpec:
    """The AICore route sorts flattened input and emits no inverse/count data."""

    @staticmethod
    def customize_inputs(x, **kwargs):
        if kwargs.get("testcase_name") != "unique_values_signed_zero_radix_f32":
            return [x]
        # Deterministically exercise radix ordering and UC's signed-zero
        # equivalence. Keep the first zero negative to check its representative.
        pattern = np.array([-0.0, +0.0, -2.0, 1.0, 1.0], dtype=np.float32)
        return [np.resize(pattern, np.asarray(x).shape)]

    @staticmethod
    def compare(*outputs_and_goldens, **kwargs):
        """Check effective values and shape metadata, not unused capacity."""
        output_count = len(outputs_and_goldens) // 2
        # Three data outputs plus TTK's dynamic-shape metadata output.
        if len(outputs_and_goldens) != 8:
            raise ValueError("Expected three data outputs and dynamic-shape metadata")
        results = []
        for index in range(output_count):
            if index in (1, 2):
                # Disabled payloads are unspecified; their shapes are still
                # checked in the separate metadata output below.
                results.append({"pass": True, "precision": "N/A (disabled payload)"})
                continue
            actual = np.asarray(outputs_and_goldens[index]).reshape(-1)
            expected = np.asarray(outputs_and_goldens[output_count + index]).reshape(-1)
            if index == 0:
                # Allocation must retain the declared input-size capacity.
                # The kernel only defines the first unique-count values.
                actual = actual[: expected.size]
            passed = actual.size == expected.size
            if passed:
                equal = actual == expected
                if expected.dtype.kind == "f" or str(expected.dtype) == "bfloat16":
                    equal |= np.isnan(actual) & np.isnan(expected)
                passed = bool(equal.all())
                # Preserve the deterministic radix signed-zero regression.
                if (
                    index == 0
                    and expected.size == 3
                    and np.array_equal(np.abs(expected), [2.0, 0.0, 1.0])
                ):
                    zeros = expected == 0
                    passed &= bool(
                        np.array_equal(
                            np.signbit(actual[zeros]), np.signbit(expected[zeros])
                        )
                    )
            results.append({"pass": passed, "precision": "100%" if passed else "FAIL"})
        return results

    @staticmethod
    def golden(x, **kwargs):
        flat = np.asarray(x).reshape(-1)
        if flat.size == 0:
            values = flat.copy()
        else:
            values = None
            if torch is not None:
                try:
                    values = (
                        torch.unique(torch.as_tensor(flat), sorted=True).cpu().numpy()
                    )
                except (RuntimeError, TypeError, ValueError):
                    # Keep TTK usable for dtypes unsupported by the local CPU
                    # PyTorch build (for example, some unsigned/bfloat16 types).
                    values = None
            if values is None:
                order = np.argsort(flat, kind="stable")
                ordered = flat[order]
                starts = np.empty(ordered.shape, dtype=np.bool_)
                starts[0] = True
                starts[1:] = np.not_equal(ordered[1:], ordered[:-1])
                values = ordered[starts]
            if kwargs.get("testcase_name") == "unique_values_signed_zero_radix_f32":
                # The radix path restores the sign of the first zero in input
                # order; CPU torch.unique may choose the other equal zero.
                zero = flat[np.flatnonzero(flat == 0)[0]]
                values[values == 0] = zero
        # The direct GE op keeps disabled inverse/count outputs as private
        # one-element placeholders. ACLNN does not expose these tensors.
        placeholder = np.ones((1,), dtype=np.int64)
        return [values, placeholder, placeholder.copy()]


def _path_spec():
    # TTK TestSpec registration is keyed by operator name, not pathway.
    # Dispatch by the CLI mode instead of testcase names or output contents.
    mode = get_global_storage().test_mode
    if mode == "geir":
        return UniqueGeirSpec
    if mode == "op":
        return UniqueKernelSpec
    raise ValueError(f"Unsupported UniqueWithCountsAndSorting test mode: {mode}")


class UniqueWithCountsAndSortingSpec:
    """One plugin file, separate kernel and GEIR contracts."""

    tolerance = _EXACT
    third_party = {"torch": UniqueGeirCompetitor}

    @staticmethod
    def customize_inputs(x, **kwargs):
        return _path_spec().customize_inputs(x, **kwargs)

    @staticmethod
    def golden(x, **kwargs):
        return _path_spec().golden(x, **kwargs)

    @staticmethod
    def compare(*outputs_and_goldens, compare_context=None, **kwargs):
        spec = _path_spec()
        if spec is UniqueGeirSpec:
            return spec.compare(
                *outputs_and_goldens, compare_context=compare_context, **kwargs
            )
        return spec.compare(*outputs_and_goldens, **kwargs)
