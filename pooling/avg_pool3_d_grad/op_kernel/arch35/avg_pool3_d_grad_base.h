/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_base.h
 * \brief 3D average pooling backward common MicroAPI helpers (arch35).
 *        Modeled on avg_pool_v2_grad_base.h, extended for the D axis.
 */

#ifndef AVG_POOL3_D_GRAD_BASE_H_
#define AVG_POOL3_D_GRAD_BASE_H_

#include "kernel_operator.h"

namespace AvgPool3DGrad {
using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t DOUBLE = 2;
constexpr uint32_t HELP_BUFFER = 1024;
constexpr uint32_t HELP_BUFFER_T3 = 2048;
constexpr uint32_t HELP_BUFFER_T3_NDHWC = 9216;

constexpr uint32_t V_REG_SIZE = 256;

constexpr uint32_t ZERO = 0;
constexpr uint32_t DIGIT_ONE = 1;
constexpr uint32_t DIGIT_TWO = 2;
constexpr uint32_t DIGIT_THREE = 3;
constexpr uint32_t DIGIT_FOUR = 4;
constexpr uint32_t DIGIT_FIVE = 5;

constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;
constexpr uint32_t INDEX_FOUR = 4;
constexpr uint32_t INDEX_FIVE = 5;
constexpr uint32_t INDEX_SIX = 6;
constexpr uint32_t INDEX_SEVEN = 7;
constexpr uint32_t INDEX_EIGHT = 8;
constexpr uint32_t INDEX_NINE = 9;
constexpr uint32_t INDEX_TEN = 10;
constexpr uint32_t INDEX_ELEVEN = 11;

using computeType = float;

constexpr AscendC::MicroAPI::CastTrait castTraitT1ComputeType = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait castTraitI64I32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

constexpr AscendC::MicroAPI::CastTrait castTraitU32U16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait castTraitI32F32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::DivSpecificMode divHighPrecisionMode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true,
                                                                     DivAlgo::PRECISION_0ULP_FTZ_TRUE};

// 3D divisor for a single axis (D/H/W), mirroring v2 ComputeDivisor.
template <typename T, const MicroAPI::RegTrait& Trait, const uint32_t COUNT_PAD>
__aicore__ inline void ComputeDivisor1D(MicroAPI::RegTensor<int32_t>& divisorAxis,
                                        MicroAPI::RegTensor<T, Trait>& outStart,
                                        MicroAPI::RegTensor<T, Trait>& zeroConstRegT, int32_t outputSize,
                                        uint16_t padLeft, uint16_t padRight, uint16_t kSize, uint32_t count)
{
    uint32_t numT = count;
    AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T, Trait>(numT);
    AscendC::MicroAPI::RegTensor<T, Trait> startReg;
    AscendC::MicroAPI::RegTensor<T, Trait> endReg;
    AscendC::MicroAPI::RegTensor<T, Trait> divisorAxisT;

    AscendC::MicroAPI::Adds(startReg, outStart, static_cast<T>(-padLeft), maskT);
    AscendC::MicroAPI::Adds(endReg, startReg, static_cast<T>(kSize), maskT);
    AscendC::MicroAPI::Mins(endReg, endReg, static_cast<T>(outputSize + padRight), maskT);

    if constexpr (COUNT_PAD == 0) {
        AscendC::MicroAPI::Max(startReg, startReg, zeroConstRegT, maskT);
        AscendC::MicroAPI::Mins(endReg, endReg, outputSize, maskT);
    }

    AscendC::MicroAPI::Sub(divisorAxisT, endReg, startReg, maskT);
    divisorAxis = (AscendC::MicroAPI::RegTensor<int32_t>&)divisorAxisT.reg[0];
}

// 3D divisor: D*H*W pool size, optionally dynamic per element (COUNT_PAD / IS_CHECK_RANGE).
template <typename T, const MicroAPI::RegTrait& Trait, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE,
          const uint32_t COUNT_PAD>
__aicore__ inline void GenDivisor3D(MicroAPI::RegTensor<int32_t>& divisorReg, MicroAPI::RegTensor<T, Trait>& outDStart,
                                    MicroAPI::RegTensor<T, Trait>& outHStart, MicroAPI::RegTensor<T, Trait>& outWStart,
                                    MicroAPI::RegTensor<T, Trait>& zeroConstRegT, int32_t dOutput, int32_t hOutput,
                                    int32_t wOutput, uint16_t padD, uint16_t padH, uint16_t padW, uint16_t padBackD,
                                    uint16_t padDownH, uint16_t padRightW, uint16_t kD, uint16_t kH, uint16_t kW,
                                    int32_t divisorOverride, uint32_t count)
{
    if constexpr (HAS_DIVISOR == 1) {
        AscendC::MicroAPI::Duplicate(divisorReg, divisorOverride);
    } else if constexpr (IS_CHECK_RANGE == 1) {
        AscendC::MicroAPI::RegTensor<int32_t> divisorD;
        AscendC::MicroAPI::RegTensor<int32_t> divisorH;
        AscendC::MicroAPI::RegTensor<int32_t> divisorW;
        uint32_t numI32 = count;
        AscendC::MicroAPI::MaskReg maskI32 = AscendC::MicroAPI::UpdateMask<int32_t>(numI32);
        ComputeDivisor1D<T, Trait, COUNT_PAD>(divisorD, outDStart, zeroConstRegT, dOutput, padD, padBackD, kD, count);
        ComputeDivisor1D<T, Trait, COUNT_PAD>(divisorH, outHStart, zeroConstRegT, hOutput, padH, padDownH, kH, count);
        ComputeDivisor1D<T, Trait, COUNT_PAD>(divisorW, outWStart, zeroConstRegT, wOutput, padW, padRightW, kW, count);
        AscendC::MicroAPI::Mul(divisorReg, divisorD, divisorH, maskI32);
        AscendC::MicroAPI::Mul(divisorReg, divisorReg, divisorW, maskI32);
    } else {
        AscendC::MicroAPI::Duplicate(divisorReg, int32_t(kD * kH * kW));
    }
}

// Expand a linear index into a 2D/3D/4D grid; returns the gather index vector.
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
__aicore__ inline void GenGatterIndex3D(MicroAPI::RegTensor<T, Trait>& indexReg, T rate3D, T num2D, T rate2D, T num1D,
                                        T rate1D = 1)
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
    return (index + pad < kernel) ? 0 : ((index + pad - kernel) / stride) + 1;
}
__aicore__ inline int64_t PEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize)
{
    int64_t tmp = (index + pad) / stride + 1;
    return tmp < pooledSize ? tmp : pooledSize;
}

// 3D range filter for d/h/w indices (IS_CHECK_RANGE).
__aicore__ inline void FilterMask3D(MicroAPI::MaskReg& preg, MicroAPI::RegTensor<int32_t>& dIndexReg,
                                    MicroAPI::RegTensor<int32_t>& hIndexReg, MicroAPI::RegTensor<int32_t>& wIndexReg,
                                    MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& dMaxReg,
                                    MicroAPI::RegTensor<int32_t>& hMaxReg, MicroAPI::RegTensor<int32_t>& wMaxReg)
{
    AscendC::MicroAPI::MaskReg gtMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, dIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, dMaxReg, dIndexReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, hIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, hMaxReg, hIndexReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);
    AscendC::MicroAPI::MaskAnd(preg, preg, gtMask, allMask);
}

// 3D gradient scatter accumulate: gradReg / divisor adds into yAddr at scatterIndexReg.
template <typename T>
__aicore__ inline void GradientAcc(__local_mem__ computeType* yAddr, MicroAPI::RegTensor<computeType>& gradReg,
                                   MicroAPI::RegTensor<T>& scatterIndexReg, MicroAPI::RegTensor<int32_t>& divisorReg,
                                   MicroAPI::MaskReg& pregRes)
{
    AscendC::MicroAPI::RegTensor<computeType> scatterAccResReg;
    AscendC::MicroAPI::RegTensor<computeType> divisorCastReg;
    AscendC::MicroAPI::RegTensor<computeType> divisorResReg;
    AscendC::MicroAPI::DataCopyGather(scatterAccResReg, yAddr, (AscendC::MicroAPI::RegTensor<uint32_t>&)scatterIndexReg,
                                      pregRes);
    AscendC::MicroAPI::Cast<computeType, int32_t, castTraitI32F32>(divisorCastReg, divisorReg, pregRes);

    AscendC::MicroAPI::Div<computeType, &divHighPrecisionMode>(divisorResReg, gradReg, divisorCastReg, pregRes);
    AscendC::MicroAPI::Add(scatterAccResReg, scatterAccResReg, divisorResReg, pregRes);
    AscendC::MicroAPI::DataCopyScatter(yAddr, scatterAccResReg,
                                       (AscendC::MicroAPI::RegTensor<uint32_t>&)scatterIndexReg, pregRes);
}

template <typename T1>
__aicore__ inline void GetConCurrentInput(MicroAPI::RegTensor<computeType>& gradReg, __local_mem__ T1* gradAddr,
                                          MicroAPI::RegTensor<uint32_t>& parallelRegIndex, MicroAPI::MaskReg& pregT1)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::MicroAPI::RegTensor<T1> gradRegT1;
        AscendC::MicroAPI::RegTensor<uint16_t> parallelRegIndexU16;
        AscendC::MicroAPI::MaskReg
            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Cast<uint16_t, uint32_t, castTraitU32U16>(parallelRegIndexU16, parallelRegIndex, allMaskU32);
        AscendC::MicroAPI::Pack(parallelRegIndexU16, (AscendC::MicroAPI::RegTensor<int32_t>&)parallelRegIndexU16);
        AscendC::MicroAPI::DataCopyGather(gradRegT1, gradAddr, parallelRegIndexU16, pregT1);
        AscendC::MicroAPI::UnPack((AscendC::MicroAPI::RegTensor<uint32_t>&)gradRegT1,
                                  (AscendC::MicroAPI::RegTensor<uint16_t>&)gradRegT1);
        AscendC::MicroAPI::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::MicroAPI::DataCopyGather(gradReg, gradAddr, parallelRegIndex, pregT1);
    }
}

// --- Shared coordinate helpers (used by both NCDHW and NDHWC kernels) ---

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void ComputeOutRegStart(MicroAPI::RegTensor<T, Trait>& outRegStart,
                                          MicroAPI::RegTensor<T, Trait>& initialRegIndex, T axisGradOffset, T stride)
{
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
    AscendC::MicroAPI::Adds(outRegStart, initialRegIndex, axisGradOffset, allMask);
    AscendC::MicroAPI::Muls(outRegStart, outRegStart, stride, allMask);
}

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void GenInitial1DIndices(MicroAPI::RegTensor<T, Trait>& indexReg, int64_t colGenRate)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, static_cast<T>(colGenRate), preg);
}

template <typename T>
__aicore__ inline void GenInitial2DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(fullBatchColNum));
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, static_cast<T>(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, static_cast<T>(colGenRate), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rowGenRate * colNumAligned), preg);
    AscendC::MicroAPI::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void Gen2DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, static_cast<T>(rowGenRate * colNumAligned), preg);
}

template <typename T>
__aicore__ inline void GenInitial3DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum, int64_t fullBatchRowNum,
                                           int64_t rowNumCount)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg2;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(fullBatchColNum * fullBatchRowNum));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, static_cast<T>(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rowNumCount * colNumAligned), preg);
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(fullBatchColNum));
    AscendC::MicroAPI::Div(segmentScalarReg2, segmentIncReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg2, segmentScalarReg2, static_cast<T>(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::MicroAPI::Muls(segmentIncReg2, segmentIncReg2, colGenRate, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, static_cast<T>(rowGenRate * colNumAligned), preg);
    AscendC::MicroAPI::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned,
                                     int64_t fullBatchRowNum, int64_t rowNumCount)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg2;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(1 * fullBatchRowNum));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, static_cast<T>(1 * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rowNumCount * colNumAligned), preg);

    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, static_cast<T>(rowGenRate * colNumAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void ComputeOutDHWIndex(MicroAPI::RegTensor<int32_t>& dIndexReg,
                                          MicroAPI::RegTensor<int32_t>& hIndexReg,
                                          MicroAPI::RegTensor<int32_t>& wIndexReg,
                                          MicroAPI::RegTensor<T, Trait>& outDStart,
                                          MicroAPI::RegTensor<T, Trait>& outHStart,
                                          MicroAPI::RegTensor<T, Trait>& outWStart, T curDIndex, T curHIndex,
                                          T curWIndex, uint16_t padD, uint16_t padH, uint16_t padW, uint32_t count)
{
    AscendC::MicroAPI::RegTensor<T, Trait> dIndexRegTwo;
    AscendC::MicroAPI::RegTensor<T, Trait> hIndexRegTwo;
    AscendC::MicroAPI::RegTensor<T, Trait> wIndexRegTwo;
    uint32_t numT = count;
    AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T, Trait>(numT);
    AscendC::MicroAPI::Adds(dIndexRegTwo, outDStart, static_cast<T>(-curDIndex - padD), maskT);
    AscendC::MicroAPI::Adds(hIndexRegTwo, outHStart, static_cast<T>(-curHIndex - padH), maskT);
    AscendC::MicroAPI::Adds(wIndexRegTwo, outWStart, static_cast<T>(-curWIndex - padW), maskT);
    dIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)dIndexRegTwo.reg[0];
    hIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)hIndexRegTwo.reg[0];
    wIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)wIndexRegTwo.reg[0];
}

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void ComputeOutWIndex(MicroAPI::RegTensor<int32_t>& wIndexReg,
                                        MicroAPI::RegTensor<T, Trait>& outWStart, T curWIndex, uint16_t padW,
                                        uint32_t count)
{
    AscendC::MicroAPI::RegTensor<T, Trait> wIndexRegTwo;
    uint32_t numT = count;
    AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T, Trait>(numT);
    AscendC::MicroAPI::Adds(wIndexRegTwo, outWStart, static_cast<T>(-curWIndex - padW), maskT);
    wIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)wIndexRegTwo.reg[0];
}

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void ComputeOutWHIndex(MicroAPI::RegTensor<int32_t>& wIndexReg,
                                         MicroAPI::RegTensor<int32_t>& hIndexReg,
                                         MicroAPI::RegTensor<T, Trait>& outWStart,
                                         MicroAPI::RegTensor<T, Trait>& outHStart, T curWIndex, T curHIndex,
                                         uint16_t padH, uint16_t padW, uint32_t count)
{
    AscendC::MicroAPI::RegTensor<T, Trait> wIndexRegTwo;
    AscendC::MicroAPI::RegTensor<T, Trait> hIndexRegTwo;
    uint32_t numT = count;
    AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T, Trait>(numT);
    AscendC::MicroAPI::Adds(wIndexRegTwo, outWStart, static_cast<T>(-curWIndex - padW), maskT);
    AscendC::MicroAPI::Adds(hIndexRegTwo, outHStart, static_cast<T>(-curHIndex - padH), maskT);
    wIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)wIndexRegTwo.reg[0];
    hIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)hIndexRegTwo.reg[0];
}

__aicore__ inline void FilterMaskForHwParallel(MicroAPI::MaskReg& preg, MicroAPI::RegTensor<int32_t>& hIndexReg,
                                               MicroAPI::RegTensor<int32_t>& wIndexReg,
                                               MicroAPI::RegTensor<int32_t>& zeroConstReg,
                                               MicroAPI::RegTensor<int32_t>& wMaxReg,
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

__aicore__ inline void FilterMaskForWParallel(MicroAPI::MaskReg& preg, MicroAPI::RegTensor<int32_t>& wIndexReg,
                                              MicroAPI::RegTensor<int32_t>& zeroConstReg,
                                              MicroAPI::RegTensor<int32_t>& wMaxReg)
{
    AscendC::MicroAPI::MaskReg gtMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);
    AscendC::MicroAPI::MaskAnd(preg, preg, gtMask, allMask);
}

__aicore__ inline void GenIndicesToUb(__local_mem__ uint32_t* helpAddr, int64_t wProBatchSize, int64_t hProBatchSize,
                                      int64_t wGradAligned, int64_t wFullBatchCount, int64_t hFullBatchCount,
                                      int64_t hGradActual)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::MicroAPI::MaskReg
            allMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, wProBatchSize, hProBatchSize,
                            wGradAligned, wFullBatchCount, hFullBatchCount, hGradActual);
        AscendC::MicroAPI::DataCopy(helpAddr, initial3DRegIndex, allMask);

        Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, hProBatchSize, wGradAligned,
                      hFullBatchCount, hGradActual);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initial3DRegIndexOne, allMask);

        GenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, hGradActual,
                            wGradAligned, wFullBatchCount);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndex, allMask);

        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, hGradActual, wGradAligned);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndexOne,
                                    allMask);
    }
}

template <typename T3, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void GenIndicesToUbForT3(__local_mem__ T3* helpAddrT3, T3 whFullBatchCount, T3 wFullBatchCount,
                                           T3 wProBatchSize, T3 hProBatchSize, T3 hFullBatchCount)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegWIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdxOne;
        AscendC::MicroAPI::RegTensor<T3, Trait> initial2DRegWIdx;

        GenGatterIndex3D<T3, Trait>(initial3DRegWIdx, 0, whFullBatchCount, 0, wFullBatchCount, wProBatchSize);
        GenGatterIndex3D<T3, Trait>(initial3DRegHIdx, 0, whFullBatchCount, hProBatchSize, wFullBatchCount, 0);
        GenGatterIndex2D<T3, Trait>(initial3DRegHIdxOne, 0, hFullBatchCount, hProBatchSize);
        GenGatterIndex2D<T3, Trait>(initial2DRegWIdx, 0, wFullBatchCount, wProBatchSize);

        AscendC::MicroAPI::MaskReg
            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        AscendC::MicroAPI::DataCopy(helpAddrT3, initial3DRegWIdx, allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3), initial3DRegHIdx, allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3), initial3DRegHIdxOne,
                                    allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3), initial2DRegWIdx,
                                    allMaskT3);
    }
}

} // namespace AvgPool3DGrad
#endif // AVG_POOL3_D_GRAD_BASE_H_
