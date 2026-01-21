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
 * \file foreach_addcmul_scalar_regbase.h
 * \brief
 */
#ifndef FOREACH_ADDCMUL_SCALAR_REGBASE_H
#define FOREACH_ADDCMUL_SCALAR_REGBASE_H

#include "../foreach_utils/arch35/foreach_regbase_ternary.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/load_store_utils.h"

namespace ForeachAddcmulScalar {
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;

constexpr int32_t VL_SIZE = platform::GetVRegSize();

template <typename T, typename ScalarT, typename Tiling>
class ForeachAddcmulScalarRegbase
    : public ForeachRegbaseTernary<T, Tiling, ForeachAddcmulScalarRegbase<T, ScalarT, Tiling>> {
public:
    using Base = ForeachRegbaseTernary<T, Tiling, ForeachAddcmulScalarRegbase<T, ScalarT, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachAddcmulScalarRegbase() : Base(*this){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace,
        const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(inputs, tensor1, tensor2, outputs, workspace, tilingData, tPipe);
        inScalarGM_.SetGlobalBuffer((__gm__ ScalarT*)scalar, 1);
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            scalarVal_ = float(inScalarGM_.GetValue(0));
        } else {
            scalarVal_ = T(inScalarGM_.GetValue(0));
        }
    }

    __aicore__ inline void Compute(
        LocalTensor<T> inLocal, LocalTensor<T> tensorOneLocal, LocalTensor<T> tensorTwoLocal, LocalTensor<T> outLocal,
        int64_t dataCount)
    {
        // input + scalar * tensor1 * tensor2
        __local_mem__ T* inUbAddr = (__ubuf__ T*)inLocal.GetPhyAddr();
        __local_mem__ T* tensorOneUbAddr = (__ubuf__ T*)tensorOneLocal.GetPhyAddr();
        __local_mem__ T* tensorTwoUbAddr = (__ubuf__ T*)tensorTwoLocal.GetPhyAddr();
        __local_mem__ T* outUbAddr = (__ubuf__ T*)outLocal.GetPhyAddr();

        uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
        uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
        uint32_t sreg = (uint32_t)dataCount;
        using scalarCalcType = typename Conditional<AscendC::IsSameType<T, int32_t>::value, int32_t, float>::type;
        scalarCalcType scalarVal = scalarCalcType(scalarVal_);

        __VEC_SCOPE__
        {
            MaskReg maskReg;
            if constexpr (
                IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value || IsSameType<T, float>::value) {
                RegTensor<float> inRegToFloat;
                RegTensor<float> tensorOneRegToFloat;
                RegTensor<float> tensorTwoRegToFloat;
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
                    ops::LoadOneTensorForDtypeT<T>(tensorOneUbAddr, tensorOneRegToFloat, maskReg, i * dataCountPerLoop);
                    ops::LoadOneTensorForDtypeT<T>(tensorTwoUbAddr, tensorTwoRegToFloat, maskReg, i * dataCountPerLoop);

                    Mul(tensorOneRegToFloat, tensorOneRegToFloat, tensorTwoRegToFloat, maskReg);
                    Axpy(inRegToFloat, tensorOneRegToFloat, scalarVal, maskReg);

                    ops::StoreOneTensorForDtypeT<T>(outUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
                }
            } else {
                RegTensor<T> inReg;
                RegTensor<T> tensorOneReg;
                RegTensor<T> tensorTwoReg;
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    DataCopy(inReg, inUbAddr + i * dataCountPerLoop);
                    DataCopy(tensorOneReg, tensorOneUbAddr + i * dataCountPerLoop);
                    DataCopy(tensorTwoReg, tensorTwoUbAddr + i * dataCountPerLoop);
                    Mul(tensorOneReg, tensorOneReg, tensorTwoReg, maskReg);
                    Muls(tensorOneReg, tensorOneReg, scalarVal, maskReg);
                    Add(inReg, inReg, tensorOneReg, maskReg);
                    DataCopy(outUbAddr + i * dataCountPerLoop, inReg, maskReg);
                }
            }
        }
    }

private:
    GlobalTensor<ScalarT> inScalarGM_;
    ScalarT scalarVal_ = 0;
};
} // namespace ForeachAddcmulScalar

#endif // FOREACH_ADDCMUL_SCALAR_REGBASE_H