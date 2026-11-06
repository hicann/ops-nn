/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BATCH_NORM_GRAD_V3_RC_PACKED_REGBASE_H
#define BATCH_NORM_GRAD_V3_RC_PACKED_REGBASE_H

#include "batch_norm_grad_v3_common.h"

namespace BatchNormGradV3 {
// Keep the packed arithmetic in the same vector scope style as the other paths.
template <typename DY_TYPE, bool FOLD>
__aicore__ inline void ComputePackedInputs(__ubuf__ DY_TYPE* dyAddr, __ubuf__ DY_TYPE* xAddr, __ubuf__ float* mean,
                                           __ubuf__ float* rstd, __ubuf__ float* beta, __ubuf__ float* gamma,
                                           uint32_t inputStride, uint32_t lanes, uint16_t groups, uint32_t elements)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> dyReg;
        Reg::RegTensor<float> xReg;
        Reg::RegTensor<float> meanReg;
        Reg::RegTensor<float> rstdReg;
        Reg::RegTensor<float> prior;
        Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::LoadAlign(meanReg, mean);
        Reg::LoadAlign(rstdReg, rstd);
        for (uint16_t g = 0; g < groups; ++g) {
            uint32_t remaining = elements - g * lanes;
            uint32_t count = remaining < lanes ? remaining : lanes;
            Reg::MaskReg valid = Reg::UpdateMask<float>(count);
            uint32_t offset = g * VL_FP32;
            LoadOneTensor<DY_TYPE>(dyAddr, dyReg, valid, g * inputStride);
            LoadOneTensor<DY_TYPE>(xAddr, xReg, valid, g * inputStride);
            Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, valid);
            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, valid);
            // dgamma 先归约后乘 rstd（段内不乘，Stage1 归约终点统一乘一次，与 RA Split-R 其余路径一致）
            // Explicitly zero padding before full-vector stores and the lane-wise block reduction.
            Reg::Muls<float, float, Reg::MaskMergeMode::ZEROING>(dyReg, dyReg, 1.0f, valid);
            if constexpr (FOLD) {
                Reg::LoadAlign(prior, beta + offset);
                Reg::Add(dyReg, dyReg, prior, all);
                Reg::LoadAlign(prior, gamma + offset);
                Reg::Add(xReg, xReg, prior, all);
            }
            Reg::StoreAlign(beta + offset, dyReg, all);
            Reg::StoreAlign(gamma + offset, xReg, all);
        }
    }
}

__aicore__ inline void FinalizePackedChannels(__ubuf__ float* output, __ubuf__ float* input, uint32_t channels,
                                              uint16_t rowsPerGroup)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> left;
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign(left, input);
        PackedFoldChannels(left, channels, rowsPerGroup);
        uint32_t count = channels;
        Reg::MaskReg valid = Reg::UpdateMask<float>(count);
        Reg::StoreAlign(output, left, valid);
    }
}
} // namespace BatchNormGradV3
#endif // BATCH_NORM_GRAD_V3_RC_PACKED_REGBASE_H
