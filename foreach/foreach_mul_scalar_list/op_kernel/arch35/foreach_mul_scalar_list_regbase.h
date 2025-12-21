/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_mul_scalar_list_regbase.h
 * \brief
 */
#ifndef FOREACH_MUL_SCALAR_LIST_REGBASE_H
#define FOREACH_MUL_SCALAR_LIST_REGBASE_H

#include "../../foreach_utils/arch35/foreach_regbase_unary_scalar_list.h"
#include "../../inc/platform.h"
#include "../../inc/load_store_utils.h"

namespace ForeachMulScalarList {
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;
constexpr int32_t VL_SIZE = platform::GetVRegSize();
template <typename T, typename ScalarT, typename Tiling>
class ForeachMulScalarListRegbase
    : public ForeachRegbaseUnaryScalarList<T, Tiling, ForeachMulScalarListRegbase<T, ScalarT, Tiling>>
{
public:
    using Base = ForeachRegbaseUnaryScalarList<T, Tiling, ForeachMulScalarListRegbase<T, ScalarT, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachMulScalarListRegbase() : Base(*this){};
    __aicore__ inline void Init(
        GM_ADDR tensor1, GM_ADDR scalars, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(tensor1, outputs, workspace, tilingData, tPipe);
        inScalarGM_.SetGlobalBuffer((__gm__ ScalarT*)scalars);
    }

    __aicore__ inline void Compute(
        LocalTensor<T> tensorLocal, LocalTensor<T> outLocal, int64_t tensorIndex, int64_t dataCount)
    {
        __local_mem__ T* inUbAddr = (__ubuf__ T*)tensorLocal.GetPhyAddr();
        __local_mem__ T* outUbAddr = (__ubuf__ T*)outLocal.GetPhyAddr();

        using scalarCalcType = typename Conditional<AscendC::IsSameType<ScalarT, int32_t>::value, int32_t, float>::type;
        scalarCalcType scaleVal = scalarCalcType(inScalarGM_.GetValue(tensorIndex));

        uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
        uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
        uint32_t sreg = (uint32_t)dataCount;

        __VEC_SCOPE__
        {
            MaskReg maskReg;
            if constexpr (
                IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value || IsSameType<T, float>::value) {
                RegTensor<float> inRegToFloat;
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
                    Muls(inRegToFloat, inRegToFloat, scaleVal, maskReg);
                    ops::StoreOneTensorForDtypeT<T>(outUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
                }
            } else {
                RegTensor<T> inReg;
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    DataCopy(inReg, inUbAddr + i * dataCountPerLoop);
                    Muls(inReg, inReg, scaleVal, maskReg);
                    DataCopy(outUbAddr + i * dataCountPerLoop, inReg, maskReg);
                }
            }
        }
    }

private:
    GlobalTensor<ScalarT> inScalarGM_;
};
} // namespace ForeachMulScalarList

#endif // FOREACH_MUL_SCALAR_LIST_REGBASE_H