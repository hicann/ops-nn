#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU checks for the local valid-index comparison contract."""

import unittest

import numpy as np

from run_aclnn_cases import compare_output


class OutputComparisonTest(unittest.TestCase):
    def setUp(self):
        # Four outputs across two channels; bytes 0..15 hold all INT32 indices.
        self.expected = np.zeros((1, 2, 4, 32), dtype=np.int8)
        self.expected.reshape(-1)[:16] = np.array([0, 7, 13, 29], dtype="<i4").view(
            np.int8
        )
        self.actual = self.expected.copy()
        self.actual.reshape(-1)[16:] = 0x7F

    def test_tail_is_ignored_in_valid_mode(self):
        compare_output(self.actual, self.expected, 1, 4, "valid")

    def test_original_full_mode_rejects_nonzero_tail(self):
        with self.assertRaises(AssertionError):
            compare_output(self.actual, self.expected, 1, 4, "full")
        compare_output(self.expected, self.expected, 1, 4, "full")

    def test_every_valid_index_byte_is_checked(self):
        for offset in range(16):
            with self.subTest(offset=offset):
                corrupt = self.actual.copy()
                corrupt.reshape(-1)[offset] ^= 1
                with self.assertRaises(AssertionError):
                    compare_output(corrupt, self.expected, 1, 4, "valid")

    def test_three_dimensional_container(self):
        compare_output(self.actual[0], self.expected[0], 1, 4, "valid")

    def test_output_values_still_compare_exact_bits(self):
        expected = np.array([-0.0, 2.0], dtype=np.float32)
        for actual in (
            np.array([0.0, 2.0], dtype=np.float32),
            np.array([-0.0, 3.0], dtype=np.float32),
        ):
            with self.assertRaises(AssertionError):
                compare_output(actual, expected, 0, 2, "valid")

    def test_shape_dtype_and_capacity_remain_checked(self):
        with self.assertRaises(AssertionError):
            compare_output(self.actual.reshape(-1), self.expected, 1, 4, "valid")
        with self.assertRaises(AssertionError):
            compare_output(self.actual.astype(np.uint8), self.expected, 1, 4, "valid")
        with self.assertRaises(AssertionError):
            compare_output(
                np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.int8), 1, 1, "valid"
            )


if __name__ == "__main__":
    unittest.main()
