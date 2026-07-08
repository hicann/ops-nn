/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GLU_GRAD_COMMON_ARCH35_H
#define GLU_GRAD_COMMON_ARCH35_H

#include "kernel_operator.h"

#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

namespace GluGrad {
namespace Common {

using namespace AscendC;

constexpr static int32_t NUM_ONE = 1;
constexpr static int32_t BUFFER_NUM = 2;
constexpr static int64_t BUFFER_SIZE = 4 * 1024;
constexpr static int32_t BLOCK_BYTES = 32;

template <typename T>
__aicore__ inline void SetGlobalBuffers(
    GlobalTensor<T>& gradOutGm, GlobalTensor<T>& selfGm, GlobalTensor<T>& outGm,
    GM_ADDR gradOut, GM_ADDR self, GM_ADDR out)
{
    gradOutGm.SetGlobalBuffer((__gm__ T*)gradOut);
    selfGm.SetGlobalBuffer((__gm__ T*)self);
    outGm.SetGlobalBuffer((__gm__ T*)out);
}

#ifdef __CCE_AICORE__

template <typename T>
__aicore__ inline void ComputeGluGradCore(
    AscendC::MicroAPI::RegTensor<T>& vregA,
    AscendC::MicroAPI::RegTensor<T>& vregB,
    AscendC::MicroAPI::RegTensor<T>& vregGrad,
    AscendC::MicroAPI::RegTensor<T>& vregOutputA,
    AscendC::MicroAPI::RegTensor<T>& vregOutputB,
    __local_mem__ T* outALocalPtr,
    __local_mem__ T* outBLocalPtr,
    uint16_t loopIdx,
    uint32_t vlSize,
    AscendC::MicroAPI::MaskReg& preg0,
    AscendC::MicroAPI::MaskReg& maskAll8,
    AscendC::MicroAPI::RegTensor<float>& vregOne)
{
    AscendC::MicroAPI::RegTensor<float> vregSigmoidB;
    AscendC::MicroAPI::RegTensor<float> vregGradA;
    AscendC::MicroAPI::RegTensor<float> vregTemp;
    AscendC::MicroAPI::RegTensor<float> vregSub;
    AscendC::MicroAPI::RegTensor<float> vregGradB;

    static constexpr AscendC::MicroAPI::DivSpecificMode highPrecisionDivMode = {
        AscendC::MicroAPI::MaskMergeMode::ZEROING, true};

    if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        AscendC::MicroAPI::RegTensor<float> vregAF;
        AscendC::MicroAPI::RegTensor<float> vregBF;
        AscendC::MicroAPI::RegTensor<float> vregGradF;

        constexpr static AscendC::MicroAPI::CastTrait castToFloatEven = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        constexpr static AscendC::MicroAPI::CastTrait castToFloatOdd = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        constexpr static AscendC::MicroAPI::CastTrait castFromFloatEven = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        constexpr static AscendC::MicroAPI::CastTrait castFromFloatOdd = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        AscendC::MicroAPI::RegTensor<T> vregOutAT;
        AscendC::MicroAPI::RegTensor<T> vregOutBT;

        AscendC::MicroAPI::Cast<float, T, castToFloatEven>(vregAF, vregA, maskAll8);
        AscendC::MicroAPI::Cast<float, T, castToFloatEven>(vregBF, vregB, maskAll8);
        AscendC::MicroAPI::Cast<float, T, castToFloatEven>(vregGradF, vregGrad, maskAll8);

        AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregBF, static_cast<float>(-1), preg0);
        AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregSub, vregTemp, preg0);
        AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregSub, static_cast<float>(1), preg0);
        AscendC::MicroAPI::Div<float, &highPrecisionDivMode>(
            vregSigmoidB, vregOne, vregTemp, preg0);

        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregGradA, vregGradF, vregSigmoidB, preg0);

        AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregOne, vregSigmoidB, preg0);
        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregTemp, vregSigmoidB, preg0);
        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregTemp, vregAF, preg0);
        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregGradB, vregTemp, vregGradF, preg0);

        AscendC::MicroAPI::Cast<T, float, castFromFloatEven>(vregOutputA, vregGradA, maskAll8);
        AscendC::MicroAPI::Cast<T, float, castFromFloatEven>(vregOutputB, vregGradB, maskAll8);

        AscendC::MicroAPI::Cast<float, T, castToFloatOdd>(vregAF, vregA, maskAll8);
        AscendC::MicroAPI::Cast<float, T, castToFloatOdd>(vregBF, vregB, maskAll8);
        AscendC::MicroAPI::Cast<float, T, castToFloatOdd>(vregGradF, vregGrad, maskAll8);

        AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregBF, static_cast<float>(-1), preg0);
        AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregSub, vregTemp, preg0);
        AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregSub, static_cast<float>(1), preg0);
        AscendC::MicroAPI::Div<float, &highPrecisionDivMode>(
            vregSigmoidB, vregOne, vregTemp, preg0);

        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregGradA, vregGradF, vregSigmoidB, preg0);

        AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregOne, vregSigmoidB, preg0);
        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregTemp, vregSigmoidB, preg0);
        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregTemp, vregAF, preg0);
        AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregGradB, vregTemp, vregGradF, preg0);

        AscendC::MicroAPI::Cast<T, float, castFromFloatOdd>(vregOutAT, vregGradA, maskAll8);
        AscendC::MicroAPI::Cast<T, float, castFromFloatOdd>(vregOutBT, vregGradB, maskAll8);

        AscendC::MicroAPI::Add<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregOutputA, vregOutputA, vregOutAT, preg0);
        AscendC::MicroAPI::Add<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregOutputB, vregOutputB, vregOutBT, preg0);

        AscendC::MicroAPI::DataCopy(outALocalPtr + loopIdx * vlSize, vregOutputA, preg0);
        AscendC::MicroAPI::DataCopy(outBLocalPtr + loopIdx * vlSize, vregOutputB, preg0);
    } else {
        AscendC::MicroAPI::Muls<T, T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregB, static_cast<T>(-1), preg0);
        AscendC::MicroAPI::Exp<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregSub, vregTemp, preg0);
        AscendC::MicroAPI::Adds<T, T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregSub, static_cast<T>(1), preg0);
        AscendC::MicroAPI::Div<T, &highPrecisionDivMode>(
            vregSigmoidB, vregOne, vregTemp, preg0);

        AscendC::MicroAPI::Mul<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregGradA, vregGrad, vregSigmoidB, preg0);

        AscendC::MicroAPI::Sub<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregOne, vregSigmoidB, preg0);
        AscendC::MicroAPI::Mul<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregTemp, vregSigmoidB, preg0);
        AscendC::MicroAPI::Mul<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregTemp, vregTemp, vregA, preg0);
        AscendC::MicroAPI::Mul<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregGradB, vregTemp, vregGrad, preg0);

        AscendC::MicroAPI::DataCopy(outALocalPtr + loopIdx * vlSize, vregGradA, preg0);
        AscendC::MicroAPI::DataCopy(outBLocalPtr + loopIdx * vlSize, vregGradB, preg0);
    }
}

template <typename T>
__aicore__ inline void ComputeGluGradImpl(
    __local_mem__ T* aLocalPtr,
    __local_mem__ T* bLocalPtr,
    __local_mem__ T* gradLocalPtr,
    __local_mem__ T* outALocalPtr,
    __local_mem__ T* outBLocalPtr,
    const int64_t& count)
{
    using namespace Ops::Base;
    constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
    uint32_t dtypeSize = sizeof(T);
    uint32_t vl = VECTOR_LENGTH / dtypeSize;
    uint16_t loopNum = CeilDivision(count, vl);
    uint32_t vlSize = vl;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregOne;
        AscendC::MicroAPI::RegTensor<T> vregA;
        AscendC::MicroAPI::RegTensor<T> vregB;
        AscendC::MicroAPI::RegTensor<T> vregGrad;
        AscendC::MicroAPI::RegTensor<T> vregOutputA;
        AscendC::MicroAPI::RegTensor<T> vregOutputB;

        AscendC::MicroAPI::MaskReg preg0;
        uint32_t size = count;
        preg0 = AscendC::MicroAPI::CreateMask<T>();
        AscendC::MicroAPI::Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(
            vregOne, static_cast<float>(1), preg0);
        AscendC::MicroAPI::MaskReg maskAll8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            preg0 = AscendC::MicroAPI::UpdateMask<T>(size);
            AscendC::MicroAPI::DataCopy(vregA, (__ubuf__ T*)(aLocalPtr + loopIdx * vlSize));
            AscendC::MicroAPI::DataCopy(vregB, (__ubuf__ T*)(bLocalPtr + loopIdx * vlSize));
            AscendC::MicroAPI::DataCopy(vregGrad, (__ubuf__ T*)(gradLocalPtr + loopIdx * vlSize));

            ComputeGluGradCore<T>(vregA, vregB, vregGrad, vregOutputA, vregOutputB,
                outALocalPtr, outBLocalPtr, loopIdx, vlSize, preg0, maskAll8, vregOne);
        }
    }
}

#endif

} // namespace Common
} // namespace GluGrad

#endif // GLU_GRAD_COMMON_ARCH35_H

