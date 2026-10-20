/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IN_TRAINING_REDUCE_V2_PRECISION_H_
#define IN_TRAINING_REDUCE_V2_PRECISION_H_

#include "in_training_reduce_v2_common.h"

namespace INTrainingReduceV2Ops {
// Keep the rounded sum and its low part separate until the final output. TwoSum
// does not feed the correction back into the main sum, preserving Inf/NaN propagation.
__simd_callee__ inline void AccumulateWithError(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& value,
                                                MaskReg& mask)
{
    RegTensor<float> total;
    RegTensor<float> back;
    RegTensor<float> error;
    Add(total, hi, value, mask);
    Sub(back, total, hi, mask);
    Sub(error, total, back, mask);
    Sub(error, hi, error, mask);
    Sub(back, value, back, mask);
    Add(error, error, back, mask);
    Add(lo, lo, error, mask);
    AscendC::Reg::Move(hi, total, mask);
}

__simd_callee__ inline void ClearNonFiniteCorrection(RegTensor<float>& lo, MaskReg& mask)
{
    RegTensor<float> zero;
    MaskReg finite;
    Duplicate(zero, 0.0f, mask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LE>(finite, lo, 3.4028234663852886e38F, mask);
    AscendC::Reg::Select(lo, lo, zero, finite);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(finite, lo, -3.4028234663852886e38F, mask);
    AscendC::Reg::Select(lo, lo, zero, finite);
}

// Explicit pairwise reduction also carries the rounding error introduced by
// horizontal addition. All gather indices stay within the 64-lane register.
__simd_callee__ inline void ReduceWithError(RegTensor<float>& hi, RegTensor<float>& lo, MaskReg& full)
{
    RegTensor<uint32_t> even;
    RegTensor<uint32_t> odd;
    RegTensor<float> rhsHi;
    RegTensor<float> rhsLo;
    AscendC::Reg::Arange((RegTensor<int32_t>&)even, static_cast<int32_t>(0));
    ShiftLefts(even, even, static_cast<int16_t>(1), full);
    Duplicate(odd, static_cast<uint32_t>(VL_FP32 - 1), full);
    AscendC::Reg::And(even, even, odd, full);
    Adds(odd, even, static_cast<uint32_t>(1), full);
#pragma unroll
    for (uint16_t level = 0; level < 6; ++level) {
        uint32_t active = VL_FP32 >> (level + 1);
        MaskReg mask = UpdateMask<float>(active);
        AscendC::Reg::Gather(rhsHi, hi, odd);
        AscendC::Reg::Gather(rhsLo, lo, odd);
        AscendC::Reg::Gather(hi, hi, even);
        AscendC::Reg::Gather(lo, lo, even);
        Add(lo, lo, rhsLo, mask);
        AccumulateWithError(hi, lo, rhsHi, mask);
    }
    ClearNonFiniteCorrection(lo, full);
}
} // namespace INTrainingReduceV2Ops
#endif // IN_TRAINING_REDUCE_V2_PRECISION_H_
