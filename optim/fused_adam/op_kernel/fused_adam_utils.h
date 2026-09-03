/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_adam_copy_adapter.h
 * \brief unified fused_adam regbase kernel for FP32/FP16/BF16
 */

#include "kernel_operator.h"

namespace FusedAdam {
using namespace AscendC;

__simd_vf__ inline void WeightDecayFP32(__ubuf__ float* dstAddr, __ubuf__ float* paramAddr, __ubuf__ float* gradAddr,
                                        float weightDecay, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> paramReg;
    Reg::RegTensor<float> gradReg;
    Reg::RegTensor<float> dstReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(paramReg, paramAddr + i * oneRepeatSize);
        Reg::LoadAlign(gradReg, gradAddr + i * oneRepeatSize);
        Reg::Muls(paramReg, paramReg, weightDecay, mask);
        Reg::Add(dstReg, gradReg, paramReg, mask);
        Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}

__simd_vf__ inline void MaximizeAndWeightDecayFP32(__ubuf__ float* dstAddr, __ubuf__ float* paramAddr,
                                                   __ubuf__ float* gradAddr, float weightDecay, uint32_t count,
                                                   uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> paramReg;
    Reg::RegTensor<float> gradReg;
    Reg::RegTensor<float> dstReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(paramReg, paramAddr + i * oneRepeatSize);
        Reg::LoadAlign(gradReg, gradAddr + i * oneRepeatSize);
        Reg::Neg(gradReg, gradReg, mask);
        Reg::Muls(paramReg, paramReg, weightDecay, mask);
        Reg::Add(dstReg, gradReg, paramReg, mask);
        Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}

__simd_vf__ inline void UpdateMVMaxVFP32(__ubuf__ float* mAddr, __ubuf__ float* vAddr, __ubuf__ float* maxvAddr,
                                         __ubuf__ float* gradAddr, float beta1, float beta2, float cbeta1, float cbeta2,
                                         uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> mReg;
    Reg::RegTensor<float> vReg;
    Reg::RegTensor<float> maxvReg;
    Reg::RegTensor<float> gradReg;
    Reg::RegTensor<float> squareReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(gradReg, gradAddr + i * oneRepeatSize);
        Reg::Mul(squareReg, gradReg, gradReg, mask);
        Reg::LoadAlign(mReg, mAddr + i * oneRepeatSize);
        Reg::Muls(mReg, mReg, beta1, mask);
        Reg::LoadAlign(vReg, vAddr + i * oneRepeatSize);
        Reg::Muls(vReg, vReg, beta2, mask);
        Reg::LoadAlign(maxvReg, maxvAddr + i * oneRepeatSize);
        Reg::Axpy(mReg, gradReg, cbeta1, mask);
        Reg::StoreAlign(mAddr + i * oneRepeatSize, mReg, mask);
        Reg::Axpy(vReg, squareReg, cbeta2, mask);
        Reg::StoreAlign(vAddr + i * oneRepeatSize, vReg, mask);
        Reg::Max(maxvReg, vReg, maxvReg, mask);
        Reg::StoreAlign(maxvAddr + i * oneRepeatSize, maxvReg, mask);
    }
}

__simd_vf__ inline void UpdateMVFP32(__ubuf__ float* mAddr, __ubuf__ float* vAddr, __ubuf__ float* gradAddr,
                                     float beta1, float beta2, float cbeta1, float cbeta2, uint32_t count,
                                     uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> mReg;
    Reg::RegTensor<float> vReg;
    Reg::RegTensor<float> gradReg;
    Reg::RegTensor<float> squareReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(gradReg, gradAddr + i * oneRepeatSize);
        Reg::Mul(squareReg, gradReg, gradReg, mask);
        Reg::LoadAlign(mReg, mAddr + i * oneRepeatSize);
        Reg::Muls(mReg, mReg, beta1, mask);
        Reg::LoadAlign(vReg, vAddr + i * oneRepeatSize);
        Reg::Muls(vReg, vReg, beta2, mask);
        Reg::Axpy(mReg, gradReg, cbeta1, mask);
        Reg::StoreAlign(mAddr + i * oneRepeatSize, mReg, mask);
        Reg::Axpy(vReg, squareReg, cbeta2, mask);
        Reg::StoreAlign(vAddr + i * oneRepeatSize, vReg, mask);
    }
}

__simd_vf__ inline void UpdateParamFP32(__ubuf__ float* paramAddr, __ubuf__ float* mAddr, __ubuf__ float* vAddr,
                                        float negStepSizeDivBC1, float sqrtbc2, float eps, uint32_t count,
                                        uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    Reg::RegTensor<float> paramReg;
    Reg::RegTensor<float> mReg;
    Reg::RegTensor<float> vReg;
    Reg::RegTensor<float> tReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(paramReg, paramAddr + i * oneRepeatSize);
        Reg::LoadAlign(mReg, mAddr + i * oneRepeatSize);
        Reg::LoadAlign(vReg, vAddr + i * oneRepeatSize);
        Reg::Sqrt(vReg, vReg, mask);
        Reg::Duplicate(tReg, sqrtbc2, mask);
        Reg::Div(vReg, vReg, tReg, mask);
        Reg::Adds(vReg, vReg, eps, mask);
        Reg::Div(mReg, mReg, vReg, mask);
        Reg::Axpy(paramReg, mReg, negStepSizeDivBC1, mask);
        Reg::StoreAlign(paramAddr + i * oneRepeatSize, paramReg, mask);
    }
}

} // namespace FusedAdam
