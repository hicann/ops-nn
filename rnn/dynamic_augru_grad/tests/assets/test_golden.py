#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from decimal import Decimal, localcontext
import unittest

import numpy as np

from golden import DynamicAugruGradSpec, DynamicAugruGradTorch


def scalar_inputs(dtype):
    def tensor(value, shape):
        return np.full(shape, value, dtype=dtype)

    gate = np.nextafter(dtype(1), dtype(0), dtype=dtype)
    return [
        tensor(0.25, (1, 1, 1)),
        np.array([[0.125, -0.25, 0.5]], dtype=dtype),
        np.array([[-0.25, 0.125, -0.5]], dtype=dtype),
        tensor(0.25, (1, 1, 1)),
        tensor(0, (1, 1, 1)),
        tensor(0.5, (1, 1)),
        tensor(0.25, (1, 1, 1)),
        tensor(0.5, (1, 1, 1)),
        tensor(0.25, (1, 1)),
        tensor(0.25, (1, 1, 1)),
        tensor(0.125, (1, 1, 1)),
        tensor(0.5, (1, 1, 1)),
        tensor(gate, (1, 1, 1)),
        tensor(0.25, (1, 1, 1)),
        np.array([1], dtype=np.int32),
        None,
    ]


def decimal_reference(inputs):
    def value(index):
        return Decimal.from_float(float(inputs[index].flat[0]))

    with localcontext() as context:
        context.prec = 70
        grad = value(7) + (value(8) if inputs[14][0] else Decimal(0))
        hp, z, u, r, n, hn, att = (value(i) for i in (5, 9, 10, 11, 12, 13, 3))
        ghn = grad * (hp - n)
        dn = grad * (1 - u) * (1 - n * n)
        dr = dn * r * (1 - r) * hn
        dz = ghn * (1 - att) * z * (1 - z)
        gi, gh = [dz, dr, dn], [dz, dr, dn * r]
        wi = [Decimal.from_float(float(v)) for v in inputs[1].flat]
        wh = [Decimal.from_float(float(v)) for v in inputs[2].flat]
        return [
            np.array([[float(value(0) * v) for v in gi]]),
            np.array([[float(hp * v) for v in gh]]),
            np.array([float(v) for v in gi]),
            np.array([float(v) for v in gh]),
            np.array([[[float(sum(a * b for a, b in zip(gi, wi)))]]]),
            np.array([[float(grad * u + sum(a * b for a, b in zip(gh, wh)))]]),
            np.array([[float(-ghn * z)]]),
        ]


class GoldenPrecisionTest(unittest.TestCase):
    def test_nonfinite_injection_accepts_input_names_with_underscores(self):
        for name, index in [
            ("weight_input", 1),
            ("weight_hidden", 2),
            ("hidden_new", 13),
        ]:
            inputs = scalar_inputs(np.float32)
            DynamicAugruGradSpec._inject_inf_nan(inputs, f"naninf_{name}_f32")
            self.assertTrue(np.isinf(inputs[index].flat[0]))
            self.assertTrue(np.isfinite(inputs[0]).all())

    def test_promoted_reference_matches_decimal_near_saturation(self):
        for dtype in (np.float16, np.float32):
            for sequence_length in (0, 1):
                with self.subTest(dtype=dtype, sequence_length=sequence_length):
                    inputs = scalar_inputs(dtype)
                    inputs[14][0] = sequence_length
                    actual = DynamicAugruGradSpec.golden(*inputs, golden_mode="Promote")
                    for output, expected in zip(actual, decimal_reference(inputs)):
                        self.assertEqual(output.dtype, np.float64)
                        self.assertEqual(output.shape, expected.shape)
                        np.testing.assert_allclose(output, expected, rtol=2e-15, atol=0)

    def test_ordinary_reference_preserves_interface_dtype(self):
        for dtype in (np.float16, np.float32):
            inputs = scalar_inputs(dtype)
            actual = DynamicAugruGradSpec.golden(*inputs)
            for output, expected in zip(actual, decimal_reference(inputs)):
                self.assertEqual(output.dtype, dtype)
                np.testing.assert_array_equal(output, expected.astype(dtype))

    def test_gate_order_changes_only_weight_and_bias_slots(self):
        inputs = scalar_inputs(np.float32)
        for index in (1, 2):
            inputs[index] = inputs[index][:, [1, 0, 2]]
        rzh = DynamicAugruGradSpec.golden(
            *inputs, gate_order="rzh", golden_mode="Promote"
        )
        zrh = DynamicAugruGradSpec.golden(
            *scalar_inputs(np.float32), golden_mode="Promote"
        )
        for index, (actual, expected) in enumerate(zip(rzh, zrh)):
            if index < 4:
                expected = expected[..., [1, 0, 2]]
            np.testing.assert_array_equal(actual, expected)

    def test_torch_reference_restores_tf32_and_keeps_output_shapes(self):
        import torch

        previous = torch.backends.cuda.matmul.allow_tf32
        inputs = [
            torch.from_numpy(x) if x is not None else None
            for x in scalar_inputs(np.float32)
        ]
        outputs = DynamicAugruGradTorch()(*inputs)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, previous)
        for actual, expected in zip(
            outputs, decimal_reference(scalar_inputs(np.float32))
        ):
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(tuple(actual.shape), expected.shape)


if __name__ == "__main__":
    unittest.main()
