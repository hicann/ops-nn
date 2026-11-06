# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch


@pytest.mark.parametrize(
    "counts",
    [
        [0] * 14,
        [1] * 14,
        [1, 0] * 7,
        [8, 0, 3, 0, 2, 1] + [0] * 8,
        [4] * 7 + [3] * 7,
        [8] * 10 + [0] * 4,
    ],
)
def test_grouped_bf16_rows_match_ungrouped(counts):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    import cann_ops_nn

    torch.manual_seed(17)
    x = torch.randn(80, 128, device="npu", dtype=torch.bfloat16)
    reference_y, reference_scale = cann_ops_nn.dequant_situ_quant(
        x, beta=1.0, linear_beta=0.0
    )
    grouped_y, grouped_scale = cann_ops_nn.dequant_situ_quant(
        x,
        group_index=torch.tensor(counts, device="npu", dtype=torch.int64),
        beta=1.0,
        linear_beta=0.0,
    )

    active_rows = min(x.shape[0], sum(counts))
    torch.testing.assert_close(
        grouped_y[:active_rows], reference_y[:active_rows], rtol=0, atol=0
    )
    torch.testing.assert_close(
        grouped_scale[:active_rows], reference_scale[:active_rows], rtol=0, atol=0
    )
    assert torch.count_nonzero(grouped_y[active_rows:]).item() == 0
    assert torch.count_nonzero(grouped_scale[active_rows:]).item() == 0
