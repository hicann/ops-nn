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

"""TTK CPU goldens for the public aclnnUnique APIs.

These are API-level references.  The direct values-only
``UniqueWithCountsAndSorting`` golden is maintained in that operator's test
directory instead.

NumPy is a CPU golden fallback for unsigned dtypes missing from torch.unique
and for BOOL/FLOAT64 first-occurrence ordering. It is never a timed competitor.
Third-party adapters call native torch.unique without casts, copies or padding;
unsupported dtypes/orderings are reported explicitly. Native NaN permutations
may differ from the stable inverse convention used by the CPU golden.
"""

import numpy as np
import torch

from ttk.core_modules.common.comparison.binary_equal import BinaryComparison
from ttk.core_modules.common.comparison.registry import CHUNK_SIZE, register_comparison


__golden__ = {
    "aclnn": {
        "aclnnUnique": "aclnn_unique_golden",
        "aclnnUnique2": "aclnn_unique2_golden",
    }
}


def _normalize_signed_zero(array):
    """Copy only arrays containing zeros, leaving dump buffers intact."""
    if isinstance(array, torch.Tensor):
        if array.is_floating_point():
            zero_mask = array == 0
            if torch.any(zero_mask):
                array = array.clone()
                array[zero_mask] = 0
    elif isinstance(array, np.ndarray) and (
        array.dtype.kind == "f" or array.dtype.name == "bfloat16"
    ):
        zero_mask = array == 0
        if np.any(zero_mask):
            array = array.copy()
            array[zero_mask] = 0
    return array


@register_comparison("binary_equal")
class UniqueExactComparison(BinaryComparison):
    """Exact Unique values, with +0 and -0 representing the same value.

    Extend binary_equal inside this Unique-only plugin process: TTK skips pre_compare when inverse/counts
    is disabled. This standard still checks values in that mixed-output case.
    Nonzero values and integer indices/counts retain binary_equal semantics.
    Chunking bounds temporary memory for large capacity-shaped outputs.
    """

    STANDARD_NAME = "unique_exact"

    def compare_impl(self):
        if self.output.dtype != self.golden.dtype or not (
            self.output.dtype.kind == "f" or self.output.dtype.name == "bfloat16"
        ):
            return super().compare_impl()
        indices = []
        diff_count = 0
        for start in range(0, self.output.size, CHUNK_SIZE):
            end = start + CHUNK_SIZE
            output = _normalize_signed_zero(self.output[start:end])
            golden = _normalize_signed_zero(self.golden[start:end])
            diff, _, count = self._numpy_binary_compare(output, golden)
            diff_count += count
            if count and len(indices) < self.MAX_DIFF_OUTPUT:
                indices.extend(
                    (diff + start)[: self.MAX_DIFF_OUTPUT - len(indices)].tolist()
                )
        return self._result(
            np.asarray(indices, dtype=np.int64) if indices else None,
            self.golden.size,
            diff_count,
        )


class UniquePrecisionSpec:
    """Keep the declared dtype and compare selected values exactly.

    Arithmetic tolerance promotes inputs to float64 in TTK, changing the
    dtype-dependent ACLNN route and ordering semantics. The plugin comparison
    extends binary_equal only for equivalent signed zeros; integers are unchanged.
    """

    tolerance = {
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


__spec__ = {
    "aclnnUnique": "UniqueAclnnSpec",
    "aclnnUnique2": "Unique2AclnnSpec",
    "torch.unique": "UniqueTorchEagerSpec",
}


def _as_bool(value):
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _is_aicpu_fallback(tensor):
    """Identify the dtypes using the ACLNN AICPU fallback in this package."""
    return tensor.dtype in (torch.bool, torch.float64)


def _bool_unique(tensor, sorted_flag, return_inverse, return_counts):
    """Reduce bool's two possible values without per-element Python objects."""
    flat = tensor.detach().cpu().numpy().reshape(-1)
    true_count = int(np.count_nonzero(flat))
    false_count = flat.size - true_count
    if not flat.size:
        values = []
    elif not true_count:
        values = [False]
    elif not false_count:
        values = [True]
    else:
        first = False if sorted_flag else bool(flat[0])
        values = [first, not first]
    result = [torch.tensor(values, dtype=torch.bool, device=tensor.device)]
    if return_inverse:
        inverse = np.empty(tensor.shape, dtype=np.int64)
        if len(values) < 2:
            inverse.fill(0)
        else:
            # Write 0/1 indices directly into the required int64 output.
            # Avoid a full-sized cast or temporary boolean comparison array.
            np.not_equal(flat, values[0], out=inverse.reshape(-1))
        result.append(torch.from_numpy(inverse).to(tensor.device))
    if return_counts:
        counts = [true_count if value else false_count for value in values]
        result.append(torch.tensor(counts, dtype=torch.int64, device=tensor.device))
    return result


def _stable_unique(tensor, return_inverse, return_counts):
    """Restore first-occurrence order after vectorized float64 uniqueness."""
    flat = tensor.detach().cpu().numpy().reshape(-1)
    # Common range-bound cases contain one numeric value. NaN != NaN keeps
    # distinct NaNs out of this shortcut; signed zeros share the first value.
    if flat.size and np.all(flat == flat[0]):
        result = [torch.from_numpy(flat[:1].copy()).to(tensor.device)]
        if return_inverse:
            inverse = np.zeros(tensor.shape, dtype=np.int64)
            result.append(torch.from_numpy(inverse).to(tensor.device))
        if return_counts:
            result.append(
                torch.tensor([flat.size], dtype=torch.int64, device=tensor.device)
            )
        return result
    unique = np.unique(
        flat,
        return_index=True,
        return_inverse=return_inverse,
        return_counts=return_counts,
        equal_nan=False,
    )
    first_indices = unique[1]
    order = np.argsort(first_indices)
    # Select the original occurrence, retaining signed zero and NaN payloads.
    result = [torch.from_numpy(flat[first_indices[order]]).to(tensor.device)]
    if return_inverse:
        ranks = np.empty(order.size, dtype=np.int64)
        ranks[order] = np.arange(order.size, dtype=np.int64)
        inverse = ranks[unique[2]].reshape(tensor.shape)
        result.append(torch.from_numpy(inverse).to(tensor.device))
    if return_counts:
        counts = unique[-1][order].astype(np.int64, copy=False)
        result.append(torch.from_numpy(counts).to(tensor.device))
    return result


def _reference_unique(tensor, sorted_flag, return_inverse, return_counts):
    """Use the matching ACLNN route semantics for the CPU golden."""
    if tensor.dtype == torch.bool:
        return _bool_unique(tensor, sorted_flag, return_inverse, return_counts)
    if _is_aicpu_fallback(tensor) and not sorted_flag:
        return _stable_unique(tensor, return_inverse, return_counts)

    # CPU torch.unique lacks parallel sort kernels for these unsigned dtypes.
    # NumPy preserves their full range; this ACLNN route sorts even when sorted=False.
    if tensor.dtype in (torch.uint16, torch.uint32, torch.uint64):
        values, inverse, counts = np.unique(
            tensor.detach().cpu().numpy().reshape(-1),
            return_inverse=True,
            return_counts=True,
        )
        result = [torch.from_numpy(values).to(tensor.device)]
        if return_inverse:
            result.append(
                torch.from_numpy(inverse.astype(np.int64))
                .reshape(tensor.shape)
                .to(tensor.device)
            )
        if return_counts:
            result.append(torch.from_numpy(counts.astype(np.int64)).to(tensor.device))
        return result

    result = torch.unique(
        tensor,
        sorted=sorted_flag,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    if not isinstance(result, tuple):
        return [result]
    result = list(result)
    if (
        (sorted_flag or not _is_aicpu_fallback(tensor))
        and return_inverse
        and tensor.is_floating_point()
    ):
        input_nan = torch.isnan(tensor)
        if torch.any(input_nan):
            # Each NaN is distinct. Torch may permute equal NaN sort keys,
            # whereas AICPU and the stable AiCore sort assign indices in input order.
            nan_indices = torch.nonzero(torch.isnan(result[0]), as_tuple=False).reshape(
                -1
            )
            result[1] = result[1].clone()
            result[1][input_nan] = nan_indices
    return result


def _native_unique(input, sorted, return_inverse, return_counts):
    """Native competitor only: no CPU/NumPy fallback or timed conversion."""
    if input.dtype in (torch.uint16, torch.uint32, torch.uint64):
        raise NotImplementedError(
            "Native torch.unique baseline does not support extended unsigned dtypes"
        )
    if not sorted and input.dtype in (torch.bool, torch.float64):
        raise NotImplementedError(
            "Native torch.unique does not preserve first-occurrence order"
        )
    return torch.unique(
        input, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts
    )


class UniqueTorchCompetitor:
    def __init__(
        self, sorted=True, return_inverse=False, return_counts=False, **kwargs
    ):
        self.flags = sorted, return_inverse, return_counts

    def __call__(self, input, dim=None, **kwargs):
        if dim is not None:
            raise NotImplementedError(
                "This baseline covers global Unique, not UniqueDim"
            )
        return _native_unique(input, *self.flags)


class UniqueAclnnCompetitor:
    def __init__(self, sorted=True, returnInverse=False, **kwargs):
        self.flags = sorted, returnInverse, False

    def __call__(impl, self, valueOut=None, inverseOut=None, **kwargs):
        # The bound signature retains the ACLNN tensor parameter named 'self'.
        result = _native_unique(self, *impl.flags)
        return list(result) if impl.flags[1] else [result, None]


class Unique2AclnnCompetitor:
    def __init__(self, sorted=True, returnInverse=False, returnCounts=False, **kwargs):
        self.flags = sorted, returnInverse, returnCounts

    def __call__(impl, self, valueOut=None, inverseOut=None, countsOut=None, **kwargs):
        result = _native_unique(self, *impl.flags)
        if not impl.flags[1] and not impl.flags[2]:
            return [result, None, None]
        return [
            result[0],
            result[1] if impl.flags[1] else None,
            result[-1] if impl.flags[2] else None,
        ]


class UniqueTorchEagerSpec:
    """Global torch.unique through torch_npu/ACLNN, without graph compilation."""

    tolerance = UniquePrecisionSpec.tolerance
    third_party = {"torch": UniqueTorchCompetitor}

    @staticmethod
    def customize_inputs(input, *args, **kwargs):
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
        if pattern is not None:
            values = torch.tensor(pattern, dtype=input.dtype)
            repeats = (input.numel() + values.numel() - 1) // values.numel()
            input.copy_(values.repeat(repeats)[: input.numel()].reshape(input.shape))

    @staticmethod
    def golden(
        input,
        sorted=True,
        return_inverse=False,
        return_counts=False,
        dim=None,
        **kwargs,
    ):
        if dim is not None:
            raise ValueError("This eager reference covers global Unique, not UniqueDim")
        return _reference_unique(input, sorted, return_inverse, return_counts)

    @staticmethod
    def compare(*outputs_and_goldens, **kwargs):
        if len(outputs_and_goldens) % 2:
            raise ValueError("Expected equal numbers of Torch outputs and goldens")
        count = len(outputs_and_goldens) // 2
        results = []
        for actual, expected in zip(
            outputs_and_goldens[:count], outputs_and_goldens[count:]
        ):
            actual_dtype = str(actual.dtype).removeprefix("torch.")
            expected_dtype = str(expected.dtype).removeprefix("torch.")
            passed = actual_dtype == expected_dtype and actual.shape == expected.shape
            if passed:
                arrays = []
                for tensor in (actual, expected):
                    if isinstance(tensor, torch.Tensor):
                        tensor = tensor.detach().cpu()
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.float()
                        tensor = tensor.numpy()
                    arrays.append(np.asarray(tensor))
                actual, expected = arrays
                passed = bool(np.array_equal(actual, expected, equal_nan=True))
            results.append({"pass": passed, "precision": "100%" if passed else "FAIL"})
        return results


def _fit_output_capacity(values, output_tensor):
    """Match a capacity-shaped output buffer used by the ACLNN wrapper.

    ``countsOut`` keeps the input capacity as its ACL tensor view.  The kernel
    writes the valid prefix and leaves the remaining elements at TTK's
    pure-output initialization value (one).  PyTorch's reference returns only
    the valid unique prefix, so pad it before TTK's byte-wise comparison.
    """
    flat = values.reshape(-1)
    if output_tensor is None:
        return flat
    capacity = int(output_tensor.numel())
    if flat.numel() >= capacity:
        return flat[:capacity]
    padding = torch.ones(
        (capacity - flat.numel(),), dtype=flat.dtype, device=flat.device
    )
    return torch.cat((flat, padding), dim=0)


def aclnn_unique_golden(self, sorted, returnInverse, valueOut, inverseOut, **kwargs):
    """Reference aclnnUnique, marking a disabled inverse slot as None."""
    return_inverse = _as_bool(returnInverse)
    result = _reference_unique(self, _as_bool(sorted), return_inverse, False)
    if return_inverse:
        values, inverse = result
        return [values, inverse]
    return [result[0], None]


def aclnn_unique2_golden(
    self, sorted, returnInverse, returnCounts, valueOut, inverseOut, countsOut, **kwargs
):
    """Reference aclnnUnique2, retaining only outputs enabled by its flags."""
    return_inverse = _as_bool(returnInverse)
    return_counts = _as_bool(returnCounts)
    result = _reference_unique(self, _as_bool(sorted), return_inverse, return_counts)
    values = result[0]
    inverse = result[1] if return_inverse else None
    counts = result[-1] if return_counts else None
    if return_counts and not _is_aicpu_fallback(self):
        counts = _fit_output_capacity(counts, countsOut)
    # Keep ACLNN output positions stable for TTK hooks; disabled API outputs
    # are not written and therefore have no reference value to compare.
    return [values, inverse, counts]


class UniqueAclnnSpec(UniquePrecisionSpec):
    golden = staticmethod(aclnn_unique_golden)
    third_party = {"torch": UniqueAclnnCompetitor}


class Unique2AclnnSpec(UniquePrecisionSpec):
    golden = staticmethod(aclnn_unique2_golden)
    third_party = {"torch": Unique2AclnnCompetitor}
