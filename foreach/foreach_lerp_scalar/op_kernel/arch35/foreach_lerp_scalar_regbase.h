/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file foreach_lerp_scalar_regbase.h
 * \brief
 */
#ifndef FOREACH_LERP_SCALAR_REGBASE_H
#define FOREACH_LERP_SCALAR_REGBASE_H

#include "../foreach_utils/arch35/foreach_regbase_binary.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/load_store_utils.h"

namespace ForeachLerpScalar {
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;
constexpr int32_t VL_SIZE = platform::GetVRegSize();
constexpr float FLOAT_NUM_NEG = -0.5f;
constexpr float FLOAT_NUM_POS = 0.5f;
constexpr float FLOAT_NUM_ONE = 1.0f;
template <typename T, typename Tiling>
class ForeachLerpScalarRegbase : public ForeachRegbaseBinary<T, Tiling, ForeachLerpScalarRegbase<T, Tiling>>
{
public:
    using Base = ForeachRegbaseBinary<T, Tiling, ForeachLerpScalarRegbase<T, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachLerpScalarRegbase() : Base(*this){};
    __aicore__ inline void Init(
        GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR weight, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData,
        TPipe* tPipe)
    {
        Base::Init(tensor1, tensor2, outputs, workspace, tilingData, tPipe);
        inScalarGM_.SetGlobalBuffer((__gm__ float*)weight, 1);
        weightVal_ = float(inScalarGM_.GetValue(0));
    }

    template <bool needExchange>
    __aicore__ inline void DoCompute(
        LocalTensor<T> tensorOneLocal, LocalTensor<T> tensorTwoLocal, LocalTensor<T> outLocal, int64_t dataCount)
    {
        // tensor1 + weight * (tensor2 - tensor1)
        // tensor2 + (tensor2 - tensor1) * (weight - 1)
        __local_mem__ T* tensorOneUbAddr = (__ubuf__ T*)tensorOneLocal.GetPhyAddr();
        __local_mem__ T* tensorTwoUbAddr = (__ubuf__ T*)tensorTwoLocal.GetPhyAddr();
        __local_mem__ T* outUbAddr = (__ubuf__ T*)outLocal.GetPhyAddr();

        uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
        uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
        uint32_t sreg = (uint32_t)dataCount;
        float weightVal = weightVal_ - FLOAT_NUM_ONE;

        __VEC_SCOPE__
        {
            MaskReg maskReg;
            RegTensor<float> inRegToFloat;
            RegTensor<float> tensorOneRegToFloat;
            RegTensor<float> tensorTwoRegToFloat;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                ops::LoadOneTensorForDtypeT<T>(tensorOneUbAddr, tensorOneRegToFloat, maskReg, i * dataCountPerLoop);
                ops::LoadOneTensorForDtypeT<T>(tensorTwoUbAddr, tensorTwoRegToFloat, maskReg, i * dataCountPerLoop);
                if constexpr (needExchange) {
                    Sub(tensorTwoRegToFloat, tensorTwoRegToFloat, tensorOneRegToFloat, maskReg);
                    Axpy(tensorOneRegToFloat, tensorTwoRegToFloat, weightVal_, maskReg);
                    ops::StoreOneTensorForDtypeT<T>(outUbAddr, tensorOneRegToFloat, maskReg, i * dataCountPerLoop);
                } else {
                    Sub(tensorOneRegToFloat, tensorTwoRegToFloat, tensorOneRegToFloat, maskReg);
                    Axpy(tensorTwoRegToFloat, tensorOneRegToFloat, weightVal, maskReg);
                    ops::StoreOneTensorForDtypeT<T>(outUbAddr, tensorTwoRegToFloat, maskReg, i * dataCountPerLoop);
                }
            }
        }
    }

    __aicore__ inline void Compute(
        LocalTensor<T> tensorOneLocal, LocalTensor<T> tensorTwoLocal, LocalTensor<T> outLocal, int64_t dataCount)
    {
        if (weightVal_ < FLOAT_NUM_POS && weightVal_ > FLOAT_NUM_NEG) {
            DoCompute<true>(tensorOneLocal, tensorTwoLocal, outLocal, dataCount);
        } else {
            DoCompute<false>(tensorOneLocal, tensorTwoLocal, outLocal, dataCount);
        }
    }

private:
    GlobalTensor<float> inScalarGM_;
    float weightVal_ = 0;
};
} // namespace ForeachLerpScalar

#endif // FOREACH_LERP_SCALAR_REGBASE_H