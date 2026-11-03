# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import unittest

import torch

from golden import _compute


class SgdGoldenTest(unittest.TestCase):
    def test_nesterov_infinite_learning_rate_preserves_update_sign(self):
        # The finite update direction is positive. Multiplying its two terms
        # by infinity separately would instead introduce inf + (-inf) = NaN.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rate, expected in ((math.inf, -math.inf), (-math.inf, math.inf)):
                with self.subTest(dtype=dtype, learning_rate=rate):
                    values = (-1.0, 10.0, rate, 1.0, -0.01, -10.0)
                    inputs = [torch.tensor([value], dtype=dtype) for value in values]
                    outputs = _compute(*inputs, nesterov=True)
                    self.assertEqual(outputs[0].item(), expected)
                    self.assertTrue(torch.isfinite(outputs[1]).all())
                    self.assertEqual(outputs[2].item(), 0.0)

    def test_nesterov_finite_update(self):
        inputs = [torch.tensor([value]) for value in (1.0, 2.0, 0.25, 4.0, 0.5, 1.0)]
        outputs = _compute(*inputs, nesterov=True)
        self.assertEqual([value.item() for value in outputs], [0.0, 4.0, 0.0])


if __name__ == "__main__":
    unittest.main()
