# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call ScatterAddWithSorted through PyTorch on Ascend 910B.

Requires PyTorch and torch_npu. Run from the repository root:
python index/scatter_add_with_sorted/examples/test_torch_scatter_add_with_sorted.py
"""

import torch
import torch_npu


def main():
    torch_npu.npu.set_device(0)
    initial = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    updates = torch.arange(1, 19, dtype=torch.float32).reshape(6, 3)
    rows = torch.tensor([2, 0, 2, 1, 0, 2], dtype=torch.int64)

    # Repeated rows accumulate into nonzero initial values; row 3 stays unchanged.
    expected = initial.clone()
    for position, row in enumerate(rows.tolist()):
        expected[row] += updates[position]

    target = initial.to("npu")
    source = updates.to("npu")
    # Keep one-dimensional storage and strides (1, 0). This broadcast index
    # selects the LinearIndex -> SortAiCore -> ScatterAddWithSorted path on 910B.
    index = rows.to("npu")[:, None].expand(-1, 3)
    target.scatter_add_(0, index, source)
    actual = target.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(actual)
    print("ScatterAddWithSorted example passed.")


if __name__ == "__main__":
    main()
