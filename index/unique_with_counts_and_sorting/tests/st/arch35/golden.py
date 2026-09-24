#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TTK golden for the values-only UniqueWithCountsAndSorting kernel.

This golden belongs to the direct ``UniqueWithCountsAndSorting`` operator.  The
public ``aclnnUnique``/``aclnnUnique2`` API goldens remain in
``index/unique/tests/assets``.
"""

__spec__ = {"unique_with_counts_and_sorting": "UniqueWithCountsAndSortingSpec"}

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - TTK environments may omit PyTorch.
    torch = None


class UniqueWithCountsAndSortingSpec:
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
