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
 * \file avg_pool_v2_grad_base.h
 * \brief
 */

#ifndef AVG_POOL_V2_GRAD_BASE_H_
#define AVG_POOL_V2_GRAD_BASE_H_

namespace AvgPoolV2Grad {
using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t DOUBLE = 2;
constexpr uint32_t HELP_BUFFER = 1024;
constexpr uint32_t HELP_BUFFER_T3 = 2048;

constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;
constexpr uint32_t INDEX_FOUR = 4;

using computeType = float;

constexpr AscendC::MicroAPI::CastTrait castTraitT1ComputeType = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitI64I32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr AscendC::MicroAPI::CastTrait castTraitU32U16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitI32F32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};

template <typename T, const MicroAPI::RegTrait& Trait, const uint32_t COUNT_PAD>
__aicore__ inline void ComputeDivisor(
    MicroAPI::RegTensor<int32_t>& divisorW, MicroAPI::RegTensor<T, Trait>& outWStart,
    MicroAPI::RegTensor<T, Trait>& zeroConstRegT, int32_t wOutput, uint16_t padW, uint16_t padRightW, uint16_t kW,
    uint32_t count)
{
    uint32_t numT = count;
    AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T, Trait>(numT);
    AscendC::MicroAPI::RegTensor<T, Trait> wStartReg;
    AscendC::MicroAPI::RegTensor<T, Trait> wEndReg;
    AscendC::MicroAPI::RegTensor<T, Trait> divisorWT;

    AscendC::MicroAPI::Adds(wStartReg, outWStart, static_cast<T>(-padW), maskT);
    AscendC::MicroAPI::Adds(wEndReg, wStartReg, static_cast<T>(kW), maskT);
    AscendC::MicroAPI::Mins(wEndReg, wEndReg, static_cast<T>(wOutput + padRightW), maskT);

    if constexpr (COUNT_PAD == 0) {
        AscendC::MicroAPI::Max(wStartReg, wStartReg, zeroConstRegT, maskT);
        AscendC::MicroAPI::Mins(wEndReg, wEndReg, wOutput, maskT);
    }

    AscendC::MicroAPI::Sub(divisorWT, wEndReg, wStartReg, maskT);
    divisorW = (AscendC::MicroAPI::RegTensor<int32_t>&)divisorWT.reg[0];
}

template <
    typename T, const MicroAPI::RegTrait& Trait, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE,
    const uint32_t COUNT_PAD>
__aicore__ inline void GenDivisor(
    MicroAPI::RegTensor<int32_t>& divisorReg, MicroAPI::RegTensor<T, Trait>& outWStart,
    MicroAPI::RegTensor<T, Trait>& outHStart, MicroAPI::RegTensor<T, Trait>& zeroConstRegT, int32_t hOutput,
    int32_t wOutput, uint16_t padH, uint16_t padW, uint16_t padDownH, uint16_t padRightW, uint16_t kH, uint16_t kW,
    int32_t divisorOverride, uint32_t count)
{
    if constexpr (HAS_DIVISOR == 1) {
        AscendC::MicroAPI::Duplicate(divisorReg, divisorOverride);
    } else if constexpr (IS_CHECK_RANGE == 1) {
        AscendC::MicroAPI::RegTensor<int32_t> divisorW;
        AscendC::MicroAPI::RegTensor<int32_t> divisorH;
        uint32_t numI32 = count;
        AscendC::MicroAPI::MaskReg maskI32 = AscendC::MicroAPI::UpdateMask<int32_t>(numI32);
        ComputeDivisor<T, Trait, COUNT_PAD>(divisorW, outWStart, zeroConstRegT, wOutput, padW, padRightW, kW, count);
        ComputeDivisor<T, Trait, COUNT_PAD>(divisorH, outHStart, zeroConstRegT, hOutput, padH, padDownH, kH, count);
        AscendC::MicroAPI::Mul(divisorReg, divisorW, divisorH, maskI32);
    } else {
        AscendC::MicroAPI::Duplicate(divisorReg, int32_t(kH * kW));
    }
}

template <typename T, const AscendC::MicroAPI::RegTrait& Trait = AscendC::MicroAPI::RegTraitNumOne>
__aicore__ inline void GenGatterIndex2D(MicroAPI::RegTensor<T, Trait>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T, Trait> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T, Trait> tmpReg;
    AscendC::MicroAPI::RegTensor<T, Trait> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(num1D));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg, static_cast<T>(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, static_cast<T>(rate1D), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T, const AscendC::MicroAPI::RegTrait& Trait = AscendC::MicroAPI::RegTraitNumOne>
__aicore__ inline void GenGatterIndex3D(
    MicroAPI::RegTensor<T, Trait>& indexReg, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T, Trait> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T, Trait> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T, Trait> tmpReg;
    AscendC::MicroAPI::RegTensor<T, Trait> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(num2D));
    AscendC::MicroAPI::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg2, static_cast<T>(num2D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, static_cast<T>(rate3D), preg);

    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(num1D));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg, static_cast<T>(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, static_cast<T>(rate1D), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg2, preg);
}

__aicore__ inline int64_t PStart(int64_t index, int64_t pad, int64_t kernel, int64_t stride)
{
    return (index + pad < kernel) ? 0 : ops::FloorDiv(index + pad - kernel, stride) + 1;
}
__aicore__ inline int64_t PEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize)
{
    int64_t tmp = ops::FloorDiv(index + pad, stride) + 1;
    return tmp < pooledSize ? tmp : pooledSize;
}

__aicore__ inline void FilterMask(
    MicroAPI::MaskReg& preg, MicroAPI::RegTensor<int32_t>& hIndexReg, MicroAPI::RegTensor<int32_t>& wIndexReg,
    MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& wMaxReg,
    MicroAPI::RegTensor<int32_t>& hMaxReg)
{
    AscendC::MicroAPI::MaskReg gtMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, hIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, hMaxReg, hIndexReg, gtMask);

    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);
    AscendC::MicroAPI::MaskAnd(preg, preg, gtMask, allMask);
}

template <typename T>
__aicore__ inline void GradientAcc(
    __local_mem__ computeType* yAddr, MicroAPI::RegTensor<computeType>& gradReg,
    MicroAPI::RegTensor<T>& scatterIndexReg, MicroAPI::RegTensor<int32_t>& divisorReg, MicroAPI::MaskReg& pregRes)
{
    AscendC::MicroAPI::RegTensor<computeType> scatterAccResReg;
    AscendC::MicroAPI::RegTensor<computeType> divisorCastReg;
    AscendC::MicroAPI::RegTensor<computeType> divisorResReg;
    AscendC::MicroAPI::DataCopyGather(
        scatterAccResReg, yAddr, (AscendC::MicroAPI::RegTensor<uint32_t>&)scatterIndexReg, pregRes);
    AscendC::MicroAPI::Cast<computeType, int32_t, castTraitI32F32>(divisorCastReg, divisorReg, pregRes);
    AscendC::MicroAPI::Div(divisorResReg, gradReg, divisorCastReg, pregRes);
    AscendC::MicroAPI::Add(scatterAccResReg, scatterAccResReg, divisorResReg, pregRes);
    AscendC::MicroAPI::DataCopyScatter(
        yAddr, scatterAccResReg, (AscendC::MicroAPI::RegTensor<uint32_t>&)scatterIndexReg, pregRes);
}

template <typename T1>
__aicore__ inline void GetConCurrentInput(
    MicroAPI::RegTensor<computeType>& gradReg, __local_mem__ T1* gradAddr,
    MicroAPI::RegTensor<uint32_t>& parallelRegIndex, MicroAPI::MaskReg& pregT1)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::MicroAPI::RegTensor<T1> gradRegT1;
        AscendC::MicroAPI::RegTensor<uint16_t> parallelRegIndexU16;
        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Cast<uint16_t, uint32_t, castTraitU32U16>(parallelRegIndexU16, parallelRegIndex, allMaskU32);
        AscendC::MicroAPI::Pack(parallelRegIndexU16, (AscendC::MicroAPI::RegTensor<int32_t>&)parallelRegIndexU16);
        AscendC::MicroAPI::DataCopyGather(gradRegT1, gradAddr, parallelRegIndexU16, pregT1);
        AscendC::MicroAPI::UnPack(
            (AscendC::MicroAPI::RegTensor<uint32_t>&)gradRegT1, (AscendC::MicroAPI::RegTensor<uint16_t>&)gradRegT1);
        AscendC::MicroAPI::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::MicroAPI::DataCopyGather(gradReg, gradAddr, parallelRegIndex, pregT1);
    }
}
} // namespace AvgPoolV2Grad
#endif // AVG_POOL_V2_GRAD_BASE_H_
