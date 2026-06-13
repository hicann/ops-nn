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
 * \file foreach_addc_list_regbase.h
 * \brief Shared Reg-VF kernel for foreach addc*_list ops with a per-tensor scalar list:
 *        y[t][i] = x1[t][i] + scalars[t] * (x2[t][i] OP x3[t][i]), OP = mul / div.
 *        addcmul_list and addcdiv_list differ only by OP, selected at compile time via AddcOp.
 */
#ifndef FOREACH_ADDC_LIST_REGBASE_H
#define FOREACH_ADDC_LIST_REGBASE_H

#include "../foreach_utils/arch35/foreach_regbase_ternary_scalar_list.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/load_store_utils.h"

namespace ForeachAddcList {
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;

enum class AddcOp { MUL, DIV };

constexpr int32_t VL_SIZE = platform::GetVRegSize();

template <typename T, typename ScalarT, typename Tiling, AddcOp OP>
class ForeachAddcListRegbase
    : public ForeachRegbaseTernaryScalarList<T, Tiling, ForeachAddcListRegbase<T, ScalarT, Tiling, OP>> {
public:
    using Base = ForeachRegbaseTernaryScalarList<T, Tiling, ForeachAddcListRegbase<T, ScalarT, Tiling, OP>>;
    using Base::Process;
    __aicore__ inline ForeachAddcListRegbase() : Base(*this){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR scalars, GM_ADDR outputs, GM_ADDR workspace,
        const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(inputs, tensor1, tensor2, outputs, workspace, tilingData, tPipe);
        inScalarPtr_ = reinterpret_cast<__gm__ ScalarT*>(scalars);
    }

    // fp16/bf16/fp32 path: load + promote to fp32, y = in + scalar*(t1 OP t2), store back as T.
    __aicore__ inline void ComputeFloatPath(
        __local_mem__ T* inUbAddr, __local_mem__ T* tensorOneUbAddr, __local_mem__ T* tensorTwoUbAddr,
        __local_mem__ T* outUbAddr, float scalarVal, uint16_t repeatTimes, uint32_t sreg, uint32_t dataCountPerLoop)
    {
        __VEC_SCOPE__
        {
            MaskReg maskReg;
            RegTensor<float> inRegToFloat;
            RegTensor<float> tensorOneRegToFloat;
            RegTensor<float> tensorTwoRegToFloat;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
                ops::LoadOneTensorForDtypeT<T>(tensorOneUbAddr, tensorOneRegToFloat, maskReg, i * dataCountPerLoop);
                ops::LoadOneTensorForDtypeT<T>(tensorTwoUbAddr, tensorTwoRegToFloat, maskReg, i * dataCountPerLoop);
                if constexpr (OP == AddcOp::MUL) {
                    Mul(tensorOneRegToFloat, tensorOneRegToFloat, tensorTwoRegToFloat, maskReg);
                } else {
                    Div(tensorOneRegToFloat, tensorOneRegToFloat, tensorTwoRegToFloat, maskReg);
                }
                Axpy(inRegToFloat, tensorOneRegToFloat, scalarVal, maskReg);
                ops::StoreOneTensorForDtypeT<T>(outUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
            }
        }
    }

    // int32 path: compute in int RegTensor, y = in + scalar*(t1 OP t2).
    __aicore__ inline void ComputeIntPath(
        __local_mem__ T* inUbAddr, __local_mem__ T* tensorOneUbAddr, __local_mem__ T* tensorTwoUbAddr,
        __local_mem__ T* outUbAddr, int32_t scalarVal, uint16_t repeatTimes, uint32_t sreg, uint32_t dataCountPerLoop)
    {
        __VEC_SCOPE__
        {
            MaskReg maskReg;
            RegTensor<T> inReg;
            RegTensor<T> tensorOneReg;
            RegTensor<T> tensorTwoReg;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                DataCopy(inReg, inUbAddr + i * dataCountPerLoop);
                DataCopy(tensorOneReg, tensorOneUbAddr + i * dataCountPerLoop);
                DataCopy(tensorTwoReg, tensorTwoUbAddr + i * dataCountPerLoop);
                if constexpr (OP == AddcOp::MUL) {
                    Mul(tensorOneReg, tensorOneReg, tensorTwoReg, maskReg);
                } else {
                    Div(tensorOneReg, tensorOneReg, tensorTwoReg, maskReg);
                }
                Muls(tensorOneReg, tensorOneReg, scalarVal, maskReg);
                Add(inReg, inReg, tensorOneReg, maskReg);
                DataCopy(outUbAddr + i * dataCountPerLoop, inReg, maskReg);
            }
        }
    }

    __aicore__ inline void Compute(
        LocalTensor<T> inLocal, LocalTensor<T> tensorOneLocal, LocalTensor<T> tensorTwoLocal, LocalTensor<T> outLocal,
        int64_t tensorIndex, int64_t dataCount)
    {
        // y = input + scalars[tensorIndex] * (tensor1 OP tensor2)
        __local_mem__ T* inUbAddr = (__ubuf__ T*)inLocal.GetPhyAddr();
        __local_mem__ T* tensorOneUbAddr = (__ubuf__ T*)tensorOneLocal.GetPhyAddr();
        __local_mem__ T* tensorTwoUbAddr = (__ubuf__ T*)tensorTwoLocal.GetPhyAddr();
        __local_mem__ T* outUbAddr = (__ubuf__ T*)outLocal.GetPhyAddr();
        uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
        uint16_t repeatTimes = CeilDivision(dataCount, static_cast<int64_t>(dataCountPerLoop));
        uint32_t sreg = (uint32_t)dataCount;
        // scalars 原型 dtype 与 x 一致；在标量域转成计算类型。regbase 标量域不支持 bf16 转换，
        // bf16→fp32 用位移手动完成（bf16 即 fp32 的高 16 位）。
        if constexpr (AscendC::IsSameType<T, int32_t>::value) {
            ComputeIntPath(inUbAddr, tensorOneUbAddr, tensorTwoUbAddr, outUbAddr,
                static_cast<int32_t>(inScalarPtr_[tensorIndex]), repeatTimes, sreg, dataCountPerLoop);
        } else {
            float scalarVal;
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                uint32_t bits =
                    static_cast<uint32_t>(*reinterpret_cast<__gm__ uint16_t*>(inScalarPtr_ + tensorIndex)) << 16;
                *reinterpret_cast<uint32_t*>(&scalarVal) = bits;
            } else {
                scalarVal = static_cast<float>(inScalarPtr_[tensorIndex]);
            }
            ComputeFloatPath(inUbAddr, tensorOneUbAddr, tensorTwoUbAddr, outUbAddr, scalarVal, repeatTimes, sreg,
                dataCountPerLoop);
        }
    }

private:
    __gm__ ScalarT* inScalarPtr_ = nullptr;
};
} // namespace ForeachAddcList

#endif // FOREACH_ADDC_LIST_REGBASE_H
