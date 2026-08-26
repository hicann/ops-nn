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

constexpr AscendC::Reg::CastTrait castTraitT1ComputeType = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitI64I32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr AscendC::Reg::CastTrait castTraitU32U16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait castTraitI32F32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_RINT};

template <typename T, const Reg::RegTrait& Trait, const uint32_t COUNT_PAD>
__aicore__ inline void ComputeDivisor(Reg::RegTensor<int32_t>& divisorW, Reg::RegTensor<T, Trait>& outWStart,
                                      Reg::RegTensor<T, Trait>& zeroConstRegT, int32_t wOutput, uint16_t padW,
                                      uint16_t padRightW, uint16_t kW, uint32_t count)
{
    uint32_t numT = count;
    AscendC::Reg::MaskReg maskT = AscendC::Reg::UpdateMask<T, Trait>(numT);
    AscendC::Reg::RegTensor<T, Trait> wStartReg;
    AscendC::Reg::RegTensor<T, Trait> wEndReg;
    AscendC::Reg::RegTensor<T, Trait> divisorWT;

    AscendC::Reg::Adds(wStartReg, outWStart, static_cast<T>(-padW), maskT);
    AscendC::Reg::Adds(wEndReg, wStartReg, static_cast<T>(kW), maskT);
    AscendC::Reg::Mins(wEndReg, wEndReg, static_cast<T>(wOutput + padRightW), maskT);

    if constexpr (COUNT_PAD == 0) {
        AscendC::Reg::Max(wStartReg, wStartReg, zeroConstRegT, maskT);
        AscendC::Reg::Mins(wEndReg, wEndReg, wOutput, maskT);
    }

    AscendC::Reg::Sub(divisorWT, wEndReg, wStartReg, maskT);
    divisorW = (AscendC::Reg::RegTensor<int32_t>&)divisorWT.reg[0];
}

template <typename T, const Reg::RegTrait& Trait, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE,
          const uint32_t COUNT_PAD>
__aicore__ inline void GenDivisor(Reg::RegTensor<int32_t>& divisorReg, Reg::RegTensor<T, Trait>& outWStart,
                                  Reg::RegTensor<T, Trait>& outHStart, Reg::RegTensor<T, Trait>& zeroConstRegT,
                                  int32_t hOutput, int32_t wOutput, uint16_t padH, uint16_t padW, uint16_t padDownH,
                                  uint16_t padRightW, uint16_t kH, uint16_t kW, int32_t divisorOverride, uint32_t count)
{
    if constexpr (HAS_DIVISOR == 1) {
        AscendC::Reg::Duplicate(divisorReg, divisorOverride);
    } else if constexpr (IS_CHECK_RANGE == 1) {
        AscendC::Reg::RegTensor<int32_t> divisorW;
        AscendC::Reg::RegTensor<int32_t> divisorH;
        uint32_t numI32 = count;
        AscendC::Reg::MaskReg maskI32 = AscendC::Reg::UpdateMask<int32_t>(numI32);
        ComputeDivisor<T, Trait, COUNT_PAD>(divisorW, outWStart, zeroConstRegT, wOutput, padW, padRightW, kW, count);
        ComputeDivisor<T, Trait, COUNT_PAD>(divisorH, outHStart, zeroConstRegT, hOutput, padH, padDownH, kH, count);
        AscendC::Reg::Mul(divisorReg, divisorW, divisorH, maskI32);
    } else {
        AscendC::Reg::Duplicate(divisorReg, int32_t(kH * kW));
    }
}

template <typename T, const AscendC::Reg::RegTrait& Trait = AscendC::Reg::RegTraitNumOne>
__aicore__ inline void GenGatterIndex2D(Reg::RegTensor<T, Trait>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T, Trait> segmentScalarReg;
    AscendC::Reg::RegTensor<T, Trait> tmpReg;
    AscendC::Reg::RegTensor<T, Trait> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL, Trait>();
    AscendC::Reg::Duplicate(constReg, static_cast<T>(num1D));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg, static_cast<T>(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, static_cast<T>(rate1D), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T, const AscendC::Reg::RegTrait& Trait = AscendC::Reg::RegTraitNumOne>
__aicore__ inline void GenGatterIndex3D(Reg::RegTensor<T, Trait>& indexReg, T rate3D, T num2D, T rate2D, T num1D,
                                        T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T, Trait> segmentScalarReg;
    AscendC::Reg::RegTensor<T, Trait> segmentScalarReg2;
    AscendC::Reg::RegTensor<T, Trait> tmpReg;
    AscendC::Reg::RegTensor<T, Trait> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL, Trait>();
    AscendC::Reg::Duplicate(constReg, static_cast<T>(num2D));
    AscendC::Reg::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg2, static_cast<T>(num2D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, static_cast<T>(rate3D), preg);

    AscendC::Reg::Duplicate(constReg, static_cast<T>(num1D));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg, static_cast<T>(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, static_cast<T>(rate1D), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg2, preg);
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

__aicore__ inline void FilterMask(Reg::MaskReg& preg, Reg::RegTensor<int32_t>& hIndexReg,
                                  Reg::RegTensor<int32_t>& wIndexReg, Reg::RegTensor<int32_t>& zeroConstReg,
                                  Reg::RegTensor<int32_t>& wMaxReg, Reg::RegTensor<int32_t>& hMaxReg)
{
    AscendC::Reg::MaskReg gtMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(gtMask, hIndexReg, zeroConstReg, gtMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(gtMask, hMaxReg, hIndexReg, gtMask);

    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);
    AscendC::Reg::And(preg, preg, gtMask, allMask);
}

__aicore__ inline void FilterMaskForMergeW(Reg::MaskReg& preg, Reg::RegTensor<int32_t>& wIndexReg,
                                           Reg::RegTensor<int32_t>& zeroConstReg, Reg::RegTensor<int32_t>& wMaxReg)
{
    AscendC::Reg::MaskReg gtMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);
    AscendC::Reg::And(preg, preg, gtMask, allMask);
}

template <typename T>
__aicore__ inline void GradientAcc(__ubuf__ computeType* yAddr, Reg::RegTensor<computeType>& gradReg,
                                   Reg::RegTensor<T>& scatterIndexReg, Reg::RegTensor<int32_t>& divisorReg,
                                   Reg::MaskReg& pregRes)
{
    AscendC::Reg::RegTensor<computeType> scatterAccResReg;
    AscendC::Reg::RegTensor<computeType> divisorCastReg;
    AscendC::Reg::RegTensor<computeType> divisorResReg;
    AscendC::Reg::Gather(scatterAccResReg, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)scatterIndexReg, pregRes);
    AscendC::Reg::Cast<computeType, int32_t, castTraitI32F32>(divisorCastReg, divisorReg, pregRes);
    AscendC::Reg::Div(divisorResReg, gradReg, divisorCastReg, pregRes);
    AscendC::Reg::Add(scatterAccResReg, scatterAccResReg, divisorResReg, pregRes);
    AscendC::Reg::Scatter(yAddr, scatterAccResReg, (AscendC::Reg::RegTensor<uint32_t>&)scatterIndexReg, pregRes);
}

template <typename T1>
__aicore__ inline void GetConCurrentInput(Reg::RegTensor<computeType>& gradReg, __ubuf__ T1* gradAddr,
                                          Reg::RegTensor<uint32_t>& parallelRegIndex, Reg::MaskReg& pregT1)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::Reg::RegTensor<T1> gradRegT1;
        AscendC::Reg::RegTensor<uint16_t> parallelRegIndexU16;
        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Cast<uint16_t, uint32_t, castTraitU32U16>(parallelRegIndexU16, parallelRegIndex, allMaskU32);
        AscendC::Reg::Pack(parallelRegIndexU16, (AscendC::Reg::RegTensor<int32_t>&)parallelRegIndexU16);
        AscendC::Reg::Gather(gradRegT1, gradAddr, parallelRegIndexU16, pregT1);
        AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint32_t>&)gradRegT1,
                             (AscendC::Reg::RegTensor<uint16_t>&)gradRegT1);
        AscendC::Reg::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::Reg::Gather(gradReg, gradAddr, parallelRegIndex, pregT1);
    }
}
} // namespace AvgPoolV2Grad
#endif // AVG_POOL_V2_GRAD_BASE_H_
