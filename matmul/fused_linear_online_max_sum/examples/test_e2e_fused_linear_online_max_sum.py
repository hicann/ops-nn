# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call FusedLinearOnlineMaxSum end to end through torch_npu on Ascend 950.

Requires PyTorch and torch_npu (the operator must be included in the active
CANN operator package). Run from the repository root:
python matmul/fused_linear_online_max_sum/examples/test_e2e_fused_linear_online_max_sum.py
"""

import numpy as np
import torch
import torch_npu


def main():
    torch_npu.npu.set_device(0)

    m, k, n = 8, 4, 16
    vocab_start_index = 2
    vocab_end_index = 14

    input_tensor = torch.randn(m, k, dtype=torch.float16)
    weight_tensor = torch.randn(n, k, dtype=torch.float16)
    # target 0 小于 vocab_start_index，target 15 不小于 vocab_end_index，两者均被掩码置零
    target_tensor = torch.tensor([0, 3, 5, 7, 9, 11, 13, 15], dtype=torch.int32)

    (
        logits_max,
        sum_exp_logits,
        predicted_logits,
        target_mask,
        masked_target,
        vocab_parallel_logits,
    ) = torch_npu.npu_fused_linear_online_max_sum(
        input_tensor.npu(),
        weight_tensor.npu(),
        target_tensor.npu(),
        vocab_start_index,
        vocab_end_index,
        return_logits=True,
    )

    print(f"logits_max: {logits_max.cpu()}")
    print(f"sum_exp_logits: {sum_exp_logits.cpu()}")
    print(f"predicted_logits: {predicted_logits.cpu()}")
    print(f"target_mask: {target_mask.cpu()}")
    print(f"masked_target: {masked_target.cpu()}")
    print(f"vocab_parallel_logits shape: {tuple(vocab_parallel_logits.shape)}")

    # CPU float32 参考实现，验证 NPU 输出
    logits = torch.matmul(input_tensor.float(), weight_tensor.float().t())
    ref_max = logits.max(dim=-1).values
    ref_sub = logits - ref_max.unsqueeze(-1)
    ref_sum = ref_sub.exp().sum(dim=-1)
    ref_mask = (target_tensor < vocab_start_index) | (target_tensor >= vocab_end_index)
    ref_masked = (target_tensor - vocab_start_index).masked_fill(ref_mask, 0)
    ref_predicted = (
        ref_sub.gather(1, ref_masked.long().unsqueeze(1))
        .squeeze(1)
        .masked_fill(ref_mask, 0.0)
    )

    torch.testing.assert_close(logits_max.cpu(), ref_max, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(sum_exp_logits.cpu(), ref_sum, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        predicted_logits.cpu(), ref_predicted, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        masked_target.cpu(), ref_masked.to(torch.int32), rtol=0, atol=0
    )
    # target_mask 按 bit 打包为 uint8，每组 8 个 token，首个 token 位于最低位
    ref_mask_byte = np.packbits(ref_mask.numpy(), bitorder="little")
    assert np.array_equal(target_mask.cpu().numpy(), ref_mask_byte), (
        f"target_mask mismatch: {target_mask.cpu().numpy()} vs {ref_mask_byte}"
    )

    print("FusedLinearOnlineMaxSum e2e example passed.")


if __name__ == "__main__":
    main()
