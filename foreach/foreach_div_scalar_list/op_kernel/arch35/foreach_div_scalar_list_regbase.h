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
 * \file foreach_div_scalar_list_regbase.h
 * \brief
 */
#ifndef FOREACH_DIV_SCALAR_LIST_REGBASE_H
#define FOREACH_DIV_SCALAR_LIST_REGBASE_H

#include "../../foreach_utils/arch35/foreach_regbase_unary_scalar_list.h"
#include "../../inc/platform.h"
#include "kernel_operator.h"
#include "../../inc/load_store_utils.h"

namespace ForeachDivScalarList {
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;
constexpr int32_t VL_SIZE = platform::GetVRegSize();
template <typename T, typename Tiling>
class ForeachDivScalarListRegbase
    : public ForeachRegbaseUnaryScalarList<T, Tiling, ForeachDivScalarListRegbase<T, Tiling>> {
public:
    using Base = ForeachRegbaseUnaryScalarList<T, Tiling, ForeachDivScalarListRegbase<T, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachDivScalarListRegbase() : Base(*this){};
    __aicore__ inline void Init(
        GM_ADDR tensor1, GM_ADDR scalars, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(tensor1, outputs, workspace, tilingData, tPipe);
        inScalarGM_.SetGlobalBuffer((__gm__ float*)scalars);
    }

    __aicore__ inline void Compute(
        LocalTensor<T> tensorLocal, LocalTensor<T> outLocal, int64_t tensorIndex, int64_t dataCount)
    {
        __local_mem__ T* inUbAddr = (__ubuf__ T*)tensorLocal.GetPhyAddr();
        __local_mem__ T* outUbAddr = (__ubuf__ T*)outLocal.GetPhyAddr();

        float scaleVal = float(inScalarGM_.GetValue(tensorIndex));

        uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
        uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
        uint32_t sreg = (uint32_t)dataCount;

        __VEC_SCOPE__
        {
            MaskReg maskReg;
            RegTensor<float> inRegToFloat;
            RegTensor<float> scaleValReg;
            RegTensor<float> outReg;
            Duplicate(scaleValReg, scaleVal);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
                Div(outReg, inRegToFloat, scaleValReg, maskReg);
                ops::StoreOneTensorForDtypeT<T>(outUbAddr, outReg, maskReg, i * dataCountPerLoop);
            }
        }
    }

private:
    GlobalTensor<float> inScalarGM_;
};
} // namespace ForeachDivScalarList

#endif // FOREACH_DIV_SCALAR_LIST_REGBASE_H