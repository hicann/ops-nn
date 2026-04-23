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
 * \file foreach_div_scalar_regbase.h
 * \brief
 */
#ifndef FOREACH_DIV_SCALAR_REGBASE_H
#define FOREACH_DIV_SCALAR_REGBASE_H

#include "../foreach_utils/arch35/foreach_regbase_unary.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/load_store_utils.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace ForeachDivScalar {
using namespace AscendC;
constexpr int32_t VL_SIZE = platform::GetVRegSize();
template <typename T, typename ScalarT, typename Tiling>
class ForeachDivScalarRegbase : public ForeachRegbaseUnary<T, Tiling, ForeachDivScalarRegbase<T, ScalarT, Tiling>>
{
public:
    using Base = ForeachRegbaseUnary<T, Tiling, ForeachDivScalarRegbase<T, ScalarT, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachDivScalarRegbase() : Base(*this){};
    __aicore__ inline void Init(
        GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData,
        TPipe* tPipe)
    {
        Base::Init(tensor1, outputs, workspace, tilingData, tPipe);
        inScalarGM_.SetGlobalBuffer((__gm__ ScalarT*)tensor2);
    }

    __aicore__ inline void Compute(LocalTensor<T> inLocal, LocalTensor<T> outLocal, int64_t dataCount)
    {
        __local_mem__ T* tensorOneUbAddr = (__ubuf__ T*)inLocal.GetPhyAddr();
        __local_mem__ T* outUbAddr = (__ubuf__ T*)outLocal.GetPhyAddr();

        uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
        uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
        uint32_t sreg = (uint32_t)dataCount;
        float scalarValue = 0.0f;
        float invScalarVal = 0.0f;
        scalarValue = float(inScalarGM_.GetValue(0));
        if (scalarValue == 0.0f) {
            float reciprocalZero = 1.0f / scalarValue;
            if (reciprocalZero < 0) {
                invScalarVal = -INFINITY;
            } else {
                invScalarVal = INFINITY;
            }
        } else if (scalarValue == INFINITY) {
            invScalarVal = 0.0f;
        } else if (scalarValue == -INFINITY) {
            invScalarVal = -0.0f;
        } else {
            invScalarVal = 1.0f / scalarValue;
            if constexpr (IsSameType<ScalarT, float>::value) {
                invScalarVal = invScalarVal * (2.0f - scalarValue * invScalarVal);
            }
        }
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg maskReg;
            MicroAPI::RegTensor<float> tensorOneRegToFloat;
            MicroAPI::RegTensor<float> outReg;
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = MicroAPI::UpdateMask<float>(sreg);
                ops::LoadOneTensorForDtypeT<T>(tensorOneUbAddr, tensorOneRegToFloat, maskReg, i * dataCountPerLoop);
                MicroAPI::Muls(outReg, tensorOneRegToFloat, invScalarVal, maskReg);
                ops::StoreOneTensorForDtypeT<T>(outUbAddr, outReg, maskReg, i * dataCountPerLoop);
            }
        }
    }

private:
    GlobalTensor<ScalarT> inScalarGM_;
};
} // namespace ForeachDivScalar

#endif // FOREACH_DIV_SCALAR_REGBASE_H