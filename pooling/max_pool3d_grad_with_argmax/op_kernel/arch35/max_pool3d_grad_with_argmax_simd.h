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
 * \file max_pool_grad_with_argmax_simd.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_SIMD_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_SIMD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "max_pool3d_grad_with_argmax_struct.h"
#include "../pool_3d_common/arch35/pool_3d_common.h"

using namespace AscendC;
using Pool3D::FastDivImpl;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t DOUBLE = 2;

constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;
constexpr uint32_t INDEX_FOUR = 4;
constexpr uint32_t INDEX_FIVE = 5;
constexpr uint32_t INDEX_SIX = 6;
constexpr uint32_t INDEX_SEVEN = 7;
using computeType = float;

constexpr uint32_t VER_NORMAL = 0;
constexpr uint32_t VER_V3 = 1;

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

__aicore__ inline constexpr uint32_t GetUbBlockSize() { return 32U; }

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310 || __NPU_ARCH == 5102
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
template <typename T2>
__aicore__ inline MicroAPI::MaskReg GenT2Mask(uint32_t& maskCount)
{
    MicroAPI::MaskReg reg;
    if constexpr (std::is_same<T2, int64_t>::value) {
        reg = AscendC::MicroAPI::UpdateMask<T2, AscendC::MicroAPI::RegTraitNumTwo>(maskCount);
    } else {
        reg = AscendC::MicroAPI::UpdateMask<T2>(maskCount);
    }
    return reg;
}

template <typename T>
__aicore__ inline void GradientAcc(
    __local_mem__ computeType* yAddr, MicroAPI::RegTensor<computeType>& gradReg, MicroAPI::RegTensor<T>& argmaxReg,
    MicroAPI::MaskReg& pregArgmax)
{
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    AscendC::MicroAPI::RegTensor<computeType> scatterAccResReg;
    AscendC::MicroAPI::DataCopyGather(
        scatterAccResReg, yAddr, (AscendC::MicroAPI::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
    AscendC::MicroAPI::Add(scatterAccResReg, scatterAccResReg, gradReg, pregArgmax);
    AscendC::MicroAPI::DataCopyScatter(
        yAddr, scatterAccResReg, (AscendC::MicroAPI::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
}

template <typename T1, typename T2>
__aicore__ inline void GetConCurrentInput(
    MicroAPI::RegTensor<int32_t>& argmaxReg, MicroAPI::RegTensor<computeType>& gradReg, __local_mem__ T1* gradAddr,
    __local_mem__ T2* argmaxAddr, MicroAPI::RegTensor<uint32_t>& parallelRegIndex, MicroAPI::MaskReg& pregT1,
    MicroAPI::MaskReg& pregT2)
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

    if constexpr (std::is_same<T2, int32_t>::value) {
        AscendC::MicroAPI::DataCopyGather(argmaxReg, argmaxAddr, parallelRegIndex, pregT2);
    } else if constexpr (std::is_same<T2, int64_t>::value) {
        AscendC::MicroAPI::RegTensor<T2, AscendC::MicroAPI::RegTraitNumTwo> argmaxRegTwo;
        AscendC::MicroAPI::DataCopyGather(argmaxRegTwo, argmaxAddr, parallelRegIndex, pregT2);
        argmaxReg = (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxRegTwo.reg[0];
    }
}

namespace MaxPool3DGradWithArgmaxNCDHWNameSpace {

template <const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNcdhwFastDiv(
    MicroAPI::RegTensor<int32_t>& argmaxReg, MicroAPI::RegTensor<uint32_t>& dTmpReg,
    MicroAPI::RegTensor<uint32_t>& hTmpReg, MicroAPI::RegTensor<uint32_t>& wTmpReg,
    MicroAPI::RegTensor<uint32_t>& magicHWReg, int16_t shiftHW, MicroAPI::RegTensor<uint32_t>& magicWReg,
    int16_t shiftW, int32_t hwOutputAligned,
    int32_t wOutputAligned, int32_t wOutput, int32_t hwOutput, int32_t baseOffset,
    int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual,
    MicroAPI::RegTensor<uint32_t>& magicHighReg, int16_t shiftHigh)
{
    MicroAPI::MaskReg allMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> remU32;

    FastDivImpl(dTmpReg, (MicroAPI::RegTensor<uint32_t>&)argmaxReg, magicHWReg, shiftHW, allMask);
    MicroAPI::Muls(remU32, dTmpReg, uint32_t(hwOutput), allMask);
    MicroAPI::Sub(remU32, (MicroAPI::RegTensor<uint32_t>&)argmaxReg, remU32, allMask);

    FastDivImpl(hTmpReg, remU32, magicWReg, shiftW, allMask);
    MicroAPI::Muls(wTmpReg, hTmpReg, uint32_t(wOutput), allMask);
    MicroAPI::Sub(wTmpReg, remU32, wTmpReg, allMask);

    MicroAPI::Muls(argmaxReg, (MicroAPI::RegTensor<int32_t>&)hTmpReg, int32_t(wOutputAligned), allMask);
    MicroAPI::Add(argmaxReg, argmaxReg, (MicroAPI::RegTensor<int32_t>&)wTmpReg, allMask);
    MicroAPI::RegTensor<int32_t> dhwTmpIndexReg;
    MicroAPI::Muls(dhwTmpIndexReg, (MicroAPI::RegTensor<int32_t>&)dTmpReg, int32_t(hwOutputAligned), allMask);
    MicroAPI::Add(argmaxReg, argmaxReg, dhwTmpIndexReg, allMask);
    MicroAPI::Adds(argmaxReg, argmaxReg, baseOffset, allMask);

    if constexpr (IS_MUL_NC == 1) {
        MicroAPI::RegTensor<int32_t> highIncRegI32;
        MicroAPI::Arange(highIncRegI32, 0);
        MicroAPI::RegTensor<uint32_t> highIncReg;
        FastDivImpl(
            (MicroAPI::RegTensor<uint32_t>&)highIncReg, (MicroAPI::RegTensor<uint32_t>&)highIncRegI32, magicHighReg,
            shiftHigh, allMask);
        MicroAPI::Muls(highIncRegI32, (MicroAPI::RegTensor<int32_t>&)highIncReg, highOutputPlaneActual, allMask);
        MicroAPI::Add(argmaxReg, argmaxReg, highIncRegI32, allMask);
    }
}

__aicore__ inline int64_t PStart(int64_t index, int64_t pad, int64_t kernel, int64_t dilation, int64_t stride)
{
    if (stride == 0) {
        return 0;
    }
    return (index + pad < (kernel - 1) * dilation + 1) ? 0 : (index + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
};
__aicore__ inline int64_t PEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize)
{
    if (stride == 0) {
        return 0;
    }
    return (index + pad) / stride + 1 < pooledSize ? (index + pad) / stride + 1 : pooledSize;
};

__aicore__ inline void FilterMask3D(
    MicroAPI::MaskReg& preg, MicroAPI::RegTensor<uint32_t>& dTmpReg, MicroAPI::RegTensor<uint32_t>& hTmpReg,
    MicroAPI::RegTensor<uint32_t>& wTmpReg, MicroAPI::RegTensor<int32_t>& dLowerReg,
    MicroAPI::RegTensor<int32_t>& hLowerReg, MicroAPI::RegTensor<int32_t>& wLowerReg,
    MicroAPI::RegTensor<int32_t>& dUpperReg, MicroAPI::RegTensor<int32_t>& hUpperReg,
    MicroAPI::RegTensor<int32_t>& wUpperReg)
{
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg hMask;
    AscendC::MicroAPI::MaskReg wMask;
    AscendC::MicroAPI::MaskReg dMask;
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(hMask, (AscendC::MicroAPI::RegTensor<int32_t>&)hTmpReg, hLowerReg, allMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(wMask, (AscendC::MicroAPI::RegTensor<int32_t>&)wTmpReg, wLowerReg, allMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(dMask, (AscendC::MicroAPI::RegTensor<int32_t>&)dTmpReg, dLowerReg, allMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(hMask, hUpperReg, (AscendC::MicroAPI::RegTensor<int32_t>&)hTmpReg, hMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(wMask, wUpperReg, (AscendC::MicroAPI::RegTensor<int32_t>&)wTmpReg, wMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(dMask, dUpperReg, (AscendC::MicroAPI::RegTensor<int32_t>&)dTmpReg, dMask);
    AscendC::MicroAPI::MaskAnd(hMask, hMask, wMask, allMask);
    AscendC::MicroAPI::MaskAnd(dMask, dMask, hMask, allMask);
    AscendC::MicroAPI::MaskAnd(preg, preg, dMask, allMask);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNchwFastDiv(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    MicroAPI::RegTensor<uint32_t>& parallelRegIndex, uint32_t argmaxMaskCount,
    MicroAPI::RegTensor<uint32_t>& magicHWReg, int16_t shiftHW, MicroAPI::RegTensor<uint32_t>& magicWReg,
    int16_t shiftW, int32_t hwOutputAligned, int32_t wOutputAligned, int32_t wOutput, int32_t hwOutput,
    int32_t baseOffset,
    MicroAPI::RegTensor<int32_t>& dLowerReg, MicroAPI::RegTensor<int32_t>& hLowerReg,
    MicroAPI::RegTensor<int32_t>& wLowerReg, MicroAPI::RegTensor<int32_t>& dUpperReg,
    MicroAPI::RegTensor<int32_t>& hUpperReg, MicroAPI::RegTensor<int32_t>& wUpperReg)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<int32_t> argmaxReg;
    AscendC::MicroAPI::RegTensor<uint32_t> dTmpReg;
    AscendC::MicroAPI::RegTensor<uint32_t> hTmpReg;
    AscendC::MicroAPI::RegTensor<uint32_t> wTmpReg;

    MicroAPI::RegTensor<uint32_t> dummyMagicHighReg;
    int16_t dummyShiftHigh = 0;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2>(maskT2);

    GetConCurrentInput<T1, T2>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, pregT1, pregT2);
    IndexConvNcdhwFastDiv<0>(
        argmaxReg, dTmpReg, hTmpReg, wTmpReg, magicHWReg, shiftHW, magicWReg, shiftW,
        hwOutputAligned, wOutputAligned, wOutput, hwOutput, baseOffset, 0, 0,
        dummyMagicHighReg, dummyShiftHigh);
    if constexpr (std::is_same<T2, int32_t>::value) {
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask3D(pregT2, dTmpReg, hTmpReg, wTmpReg, dLowerReg, hLowerReg, wLowerReg,
                         dUpperReg, hUpperReg, wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregT2);
    } else {
        uint32_t argmaxMask = argmaxMaskCount;
        AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask3D(pregArgmax, dTmpReg, hTmpReg, wTmpReg, dLowerReg, hLowerReg, wLowerReg,
                         dUpperReg, hUpperReg, wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
    }
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNcdhwFastDiv(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    MicroAPI::RegTensor<uint32_t>& parallelRegIndex, uint32_t argmaxMaskCount,
    MicroAPI::RegTensor<uint32_t>& magicHWReg, int16_t shiftHW, MicroAPI::RegTensor<uint32_t>& magicWReg,
    int16_t shiftW, int32_t hwOutputAligned, int32_t wOutputAligned, int32_t wOutput, int32_t hwOutput,
    int32_t baseOffset,
    MicroAPI::RegTensor<int32_t>& dLowerReg, MicroAPI::RegTensor<int32_t>& hLowerReg,
    MicroAPI::RegTensor<int32_t>& wLowerReg, MicroAPI::RegTensor<int32_t>& dUpperReg,
    MicroAPI::RegTensor<int32_t>& hUpperReg, MicroAPI::RegTensor<int32_t>& wUpperReg)
{
    DoSingleNCNchwFastDiv<T1, T2, IS_CHECK_RANGE>(
        yAddr, gradAddr, argmaxAddr, parallelRegIndex, argmaxMaskCount, magicHWReg, shiftHW, magicWReg, shiftW,
        hwOutputAligned, wOutputAligned, wOutput, hwOutput, baseOffset,
        dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg, wUpperReg);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulNCNcdhwFastDiv(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    MicroAPI::RegTensor<uint32_t>& parallelRegIndex, uint32_t argmaxMaskCount,
    MicroAPI::RegTensor<uint32_t>& magicHWReg, int16_t shiftHW, MicroAPI::RegTensor<uint32_t>& magicWReg,
    int16_t shiftW, int32_t hwOutputAligned, int32_t wOutputAligned, int32_t wOutput, int32_t hwOutput,
    int32_t baseOffset,
    MicroAPI::RegTensor<int32_t>& dLowerReg, MicroAPI::RegTensor<int32_t>& hLowerReg,
    MicroAPI::RegTensor<int32_t>& wLowerReg, MicroAPI::RegTensor<int32_t>& dUpperReg,
    MicroAPI::RegTensor<int32_t>& hUpperReg, MicroAPI::RegTensor<int32_t>& wUpperReg,
    int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual,
    MicroAPI::RegTensor<uint32_t>& magicHighReg, int16_t shiftHigh)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<int32_t> argmaxReg;
    AscendC::MicroAPI::RegTensor<uint32_t> dTmpReg;
    AscendC::MicroAPI::RegTensor<uint32_t> hTmpReg;
    AscendC::MicroAPI::RegTensor<uint32_t> wTmpReg;
    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2>(maskT2);
    GetConCurrentInput<T1, T2>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, pregT1, pregT2);

    IndexConvNcdhwFastDiv<1>(
        argmaxReg, dTmpReg, hTmpReg, wTmpReg, magicHWReg, shiftHW, magicWReg, shiftW,
        hwOutputAligned, wOutputAligned, wOutput, hwOutput, baseOffset,
        highOutputPlaneActual, highArgmaxPlaneActual, magicHighReg, shiftHigh);

    if constexpr (std::is_same<T2, int32_t>::value) {
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask3D(pregT2, dTmpReg, hTmpReg, wTmpReg, dLowerReg, hLowerReg, wLowerReg,
                         dUpperReg, hUpperReg, wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregT2);
    } else {
        uint32_t argmaxMask = argmaxMaskCount;
        AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask3D(pregArgmax, dTmpReg, hTmpReg, wTmpReg, dLowerReg, hLowerReg, wLowerReg,
                         dUpperReg, hUpperReg, wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
    }
}

template <typename T>
__aicore__ inline void GenInitial1DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(colGenRate), preg);
}

template <typename T>
__aicore__ inline void GenInitial2DHighIndices(
    MicroAPI::RegTensor<T>& indexReg, int64_t highStride, int64_t colGenRate, int64_t fullBatchColNum)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum));
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(highStride), preg);
    AscendC::MicroAPI::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void Gen2DHighIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t highStride)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(highStride), preg);
}

template <typename T>
__aicore__ inline void GenInitial2DIndices(
    MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchColNum)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum));
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNumAligned), preg);
    AscendC::MicroAPI::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}
template <typename T>
__aicore__ inline void DhwGenInitial2DIndices(
    MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchColNum)
{
    GenInitial2DIndices<T>(indexReg, colGenRate, rowGenRate, colNumAligned, fullBatchColNum);
}

template <typename T>
__aicore__ inline void Gen2DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rowGenRate * colNumAligned), preg);
}

template <typename T>
__aicore__ inline void DhwGen2DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    Gen2DIndexOne<T>(indexReg, rowGenRate, colNumAligned);
}

template <typename T>
__aicore__ inline void GenInitial3DIndices(
    MicroAPI::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate, int64_t colGenRate, int64_t fullBatchRowNum,
    int64_t rowNumCount, int64_t fullBatchColNum, int64_t colNumAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg2;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum));
    AscendC::MicroAPI::Div(segmentScalarReg2, segmentIncReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg2, segmentScalarReg2, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::MicroAPI::Muls(segmentIncReg2, segmentIncReg2, colGenRate, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, T(rowGenRate * colNumAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOne(
    MicroAPI::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchRowNum, int64_t rowNumCount)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg2;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(1 * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);

    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, T(rowGenRate * colNumAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial3DHighIndices(
    MicroAPI::RegTensor<T>& indexReg, int64_t highStride, int64_t colGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchColNum, int64_t fullBatchRowNum)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::RegTensor<T> highReg;
    AscendC::MicroAPI::RegTensor<T> hwReg;
    AscendC::MicroAPI::RegTensor<T> hReg;
    AscendC::MicroAPI::RegTensor<T> wReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;
    const uint64_t wStride = colGenRate;

    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum));
    AscendC::MicroAPI::Div(highReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, highReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(hwReg, indexReg, tmpReg, preg);

    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum));
    AscendC::MicroAPI::Div(hReg, hwReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, hReg, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(wReg, hwReg, tmpReg, preg);

    AscendC::MicroAPI::RegTensor<T> highPartReg;
    AscendC::MicroAPI::RegTensor<T> hPartReg;
    AscendC::MicroAPI::RegTensor<T> wPartReg;
    AscendC::MicroAPI::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::MicroAPI::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::MicroAPI::Muls(wPartReg, wReg, T(wStride), preg);

    AscendC::MicroAPI::Add(indexReg, highPartReg, hPartReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DHighIndexOne(
    MicroAPI::RegTensor<T>& indexReg, int64_t highStride, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchRowNum)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::RegTensor<T> highReg;
    AscendC::MicroAPI::RegTensor<T> hReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;

    AscendC::MicroAPI::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::MicroAPI::Div(highReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, highReg, T(1 * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(hReg, indexReg, tmpReg, preg);

    AscendC::MicroAPI::RegTensor<T> highPartReg;
    AscendC::MicroAPI::RegTensor<T> hPartReg;
    AscendC::MicroAPI::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::MicroAPI::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::MicroAPI::Add(indexReg, highPartReg, hPartReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial4DIndices(
    MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchColNum, int64_t fullBatchRowNum, int64_t fullBatchDepthNum, int64_t depthStride,
    int64_t highStride)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::RegTensor<T> highReg;
    AscendC::MicroAPI::RegTensor<T> dReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::RegTensor<T> dhwReg;
    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum * fullBatchDepthNum));
    AscendC::MicroAPI::Div(highReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(dReg, highReg, T(fullBatchColNum * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::MicroAPI::Sub(dhwReg, indexReg, dReg, preg);

    AscendC::MicroAPI::RegTensor<T> hwReg;
    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchRowNum * fullBatchColNum));
    AscendC::MicroAPI::Div(dReg, dhwReg, constReg, preg);
    AscendC::MicroAPI::Muls(hwReg, dReg, T(fullBatchRowNum * fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(hwReg, dhwReg, hwReg, preg);

    AscendC::MicroAPI::RegTensor<T> hReg;
    AscendC::MicroAPI::RegTensor<T> wReg;

    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum));
    AscendC::MicroAPI::Div(hReg, hwReg, constReg, preg);
    AscendC::MicroAPI::Muls(wReg, hReg, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(wReg, hwReg, wReg, preg);

    // 组装offset
    AscendC::MicroAPI::RegTensor<T> highPartReg;
    AscendC::MicroAPI::RegTensor<T> dPartReg;
    AscendC::MicroAPI::RegTensor<T> hPartReg;
    AscendC::MicroAPI::RegTensor<T> wPartReg;

    AscendC::MicroAPI::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::MicroAPI::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::MicroAPI::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::MicroAPI::Muls(wPartReg, wReg, T(colGenRate), preg);
    AscendC::MicroAPI::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, hPartReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen4DIndexOne(
    MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned, int64_t fullBatchRowNum,
    int64_t fullBatchDepthNum, int64_t depthStride, int64_t highStride)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::RegTensor<T> highReg;
    AscendC::MicroAPI::RegTensor<T> dReg;
    AscendC::MicroAPI::RegTensor<T> dhwReg;
    AscendC::MicroAPI::RegTensor<T> constReg;

    AscendC::MicroAPI::Duplicate(constReg, T(1 * fullBatchRowNum * fullBatchDepthNum));
    AscendC::MicroAPI::Div(highReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(dReg, highReg, T(1 * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::MicroAPI::Sub(dhwReg, indexReg, dReg, preg);

    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> hReg;

    AscendC::MicroAPI::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::MicroAPI::Div(dReg, dhwReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, dReg, T(1 * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(hReg, dhwReg, tmpReg, preg);
    AscendC::MicroAPI::RegTensor<T> highPartReg;
    AscendC::MicroAPI::RegTensor<T> dPartReg;
    AscendC::MicroAPI::RegTensor<T> hPartReg;

    AscendC::MicroAPI::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::MicroAPI::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::MicroAPI::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::MicroAPI::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, hPartReg, preg);
}

struct DivMagic {
    uint32_t magic;
    int16_t shift;
};

__aicore__ inline DivMagic PrecomputeDiv(uint32_t divisor)
{
    DivMagic dm;
    uint32_t m = 0, s = 0;
    GetUintDivMagicAndShift<uint32_t>(m, s, divisor);
    dm.magic = m;
    dm.shift = static_cast<int16_t>(s);
    return dm;
}

__aicore__ inline void FastDivInt32(
    MicroAPI::RegTensor<int32_t>& res, MicroAPI::RegTensor<int32_t>& src, const DivMagic& dm)
{
    MicroAPI::RegTensor<uint32_t> tmp;
    MicroAPI::RegTensor<uint32_t> magicReg;
    MicroAPI::Duplicate(magicReg, dm.magic);
    MicroAPI::MaskReg allMask = MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    FastDivImpl(tmp, (MicroAPI::RegTensor<uint32_t>&)src, magicReg, dm.shift, allMask);
    res = (MicroAPI::RegTensor<int32_t>&)tmp;
}

template <typename T>
__aicore__ inline void GenInitial2DHighIndicesFast(
    MicroAPI::RegTensor<T>& indexReg, int64_t highStride, int64_t colGenRate, int64_t fullBatchColNum,
    const DivMagic& divW)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    FastDivInt32(segmentScalarReg, indexReg, divW);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(highStride), preg);
    AscendC::MicroAPI::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial2DIndicesFast(
    MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchColNum, const DivMagic& divW)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    FastDivInt32(segmentScalarReg, indexReg, divW);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNumAligned), preg);
    AscendC::MicroAPI::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void DhwGenInitial2DIndicesFast(
    MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchColNum, const DivMagic& divW)
{
    GenInitial2DIndicesFast<T>(indexReg, colGenRate, rowGenRate, colNumAligned, fullBatchColNum, divW);
}

template <typename T>
__aicore__ inline void GenInitial3DIndicesFast(
    MicroAPI::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate, int64_t colGenRate, int64_t fullBatchRowNum,
    int64_t rowNumCount, int64_t fullBatchColNum, int64_t colNumAligned,
    const DivMagic& divWH, const DivMagic& divW)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg2;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    FastDivInt32(segmentScalarReg, indexReg, divWH);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);

    FastDivInt32(segmentScalarReg2, segmentIncReg, divW);
    AscendC::MicroAPI::Muls(segmentIncReg2, segmentScalarReg2, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::MicroAPI::Muls(segmentIncReg2, segmentIncReg2, colGenRate, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, T(rowGenRate * colNumAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOneFast(
    MicroAPI::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchRowNum, int64_t rowNumCount, const DivMagic& divH)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    FastDivInt32(segmentScalarReg, indexReg, divH);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(1 * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, T(rowGenRate * colNumAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial3DHighIndicesFast(
    MicroAPI::RegTensor<T>& indexReg, int64_t highStride, int64_t colGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchColNum, int64_t fullBatchRowNum,
    const DivMagic& divWH, const DivMagic& divW)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> highReg;
    AscendC::MicroAPI::RegTensor<T> hwReg;
    AscendC::MicroAPI::RegTensor<T> hReg;
    AscendC::MicroAPI::RegTensor<T> wReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;
    const uint64_t wStride = colGenRate;

    FastDivInt32(highReg, indexReg, divWH);
    AscendC::MicroAPI::Muls(tmpReg, highReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(hwReg, indexReg, tmpReg, preg);

    FastDivInt32(hReg, hwReg, divW);
    AscendC::MicroAPI::Muls(tmpReg, hReg, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(wReg, hwReg, tmpReg, preg);

    AscendC::MicroAPI::RegTensor<T> highPartReg;
    AscendC::MicroAPI::RegTensor<T> hPartReg;
    AscendC::MicroAPI::RegTensor<T> wPartReg;
    AscendC::MicroAPI::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::MicroAPI::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::MicroAPI::Muls(wPartReg, wReg, T(wStride), preg);

    AscendC::MicroAPI::Add(indexReg, highPartReg, hPartReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DHighIndexOneFast(
    MicroAPI::RegTensor<T>& indexReg, int64_t highStride, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchRowNum, const DivMagic& divH)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> highReg;
    AscendC::MicroAPI::RegTensor<T> hReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;

    FastDivInt32(highReg, indexReg, divH);
    AscendC::MicroAPI::Muls(tmpReg, highReg, T(1 * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(hReg, indexReg, tmpReg, preg);

    AscendC::MicroAPI::RegTensor<T> highPartReg;
    AscendC::MicroAPI::RegTensor<T> hPartReg;
    AscendC::MicroAPI::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::MicroAPI::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::MicroAPI::Add(indexReg, highPartReg, hPartReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial4DIndicesFast(
    MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate, int64_t colNumAligned,
    int64_t fullBatchColNum, int64_t fullBatchRowNum, int64_t fullBatchDepthNum, int64_t depthStride,
    int64_t highStride,
    const DivMagic& divDHW, const DivMagic& divHW, const DivMagic& divW)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::RegTensor<T> highReg;
    AscendC::MicroAPI::RegTensor<T> dReg;
    AscendC::MicroAPI::RegTensor<T> dhwReg;
    AscendC::MicroAPI::RegTensor<T> hwReg;
    AscendC::MicroAPI::RegTensor<T> hReg;
    AscendC::MicroAPI::RegTensor<T> wReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;

    FastDivInt32(highReg, indexReg, divDHW);
    AscendC::MicroAPI::Muls(dReg, highReg, T(fullBatchColNum * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::MicroAPI::Sub(dhwReg, indexReg, dReg, preg);

    FastDivInt32(dReg, dhwReg, divHW);
    AscendC::MicroAPI::Muls(hwReg, dReg, T(fullBatchRowNum * fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(hwReg, dhwReg, hwReg, preg);

    FastDivInt32(hReg, hwReg, divW);
    AscendC::MicroAPI::Muls(wReg, hReg, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(wReg, hwReg, wReg, preg);

    AscendC::MicroAPI::RegTensor<T> highPartReg;
    AscendC::MicroAPI::RegTensor<T> dPartReg;
    AscendC::MicroAPI::RegTensor<T> hPartReg;
    AscendC::MicroAPI::RegTensor<T> wPartReg;

    AscendC::MicroAPI::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::MicroAPI::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::MicroAPI::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::MicroAPI::Muls(wPartReg, wReg, T(colGenRate), preg);
    AscendC::MicroAPI::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, hPartReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen4DIndexOneFast(
    MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned, int64_t fullBatchRowNum,
    int64_t fullBatchDepthNum, int64_t depthStride, int64_t highStride,
    const DivMagic& divHD, const DivMagic& divH)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::RegTensor<T> highReg;
    AscendC::MicroAPI::RegTensor<T> dReg;
    AscendC::MicroAPI::RegTensor<T> dhwReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> hReg;

    FastDivInt32(highReg, indexReg, divHD);
    AscendC::MicroAPI::Muls(dReg, highReg, T(1 * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::MicroAPI::Sub(dhwReg, indexReg, dReg, preg);

    FastDivInt32(dReg, dhwReg, divH);
    AscendC::MicroAPI::Muls(tmpReg, dReg, T(1 * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(hReg, dhwReg, tmpReg, preg);

    AscendC::MicroAPI::RegTensor<T> highPartReg;
    AscendC::MicroAPI::RegTensor<T> dPartReg;
    AscendC::MicroAPI::RegTensor<T> hPartReg;

    AscendC::MicroAPI::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::MicroAPI::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::MicroAPI::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::MicroAPI::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, hPartReg, preg);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
class MaxPool3DGradWithArgmaxNCDHWKernel {
public:
    __aicore__ inline MaxPool3DGradWithArgmaxNCDHWKernel(void){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
        const MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData& tilingData);
    __aicore__ inline void ParseTilingData(
        const MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void singleLineProcessVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void multipleLineProcessVF2(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void multipleLineHwProcessVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void multipleLineDhwProcessVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void CopyOut();

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> argmaxQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;

    GlobalTensor<T1> gradGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;

    uint32_t blockIdx_ = 0;

    int64_t dArgmax_ = 1;
    int64_t hArgmax_ = 1;
    int64_t wArgmax_ = 1;

    int64_t dOutput_ = 1;
    int64_t hOutput_ = 1;
    int64_t wOutput_ = 1;

    int64_t kernelD_ = 1;
    int64_t kernelH_ = 1;
    int64_t kernelW_ = 1;

    int64_t strideD_ = 1;
    int64_t strideH_ = 1;
    int64_t strideW_ = 1;

    int64_t padD_ = 0;
    int64_t padH_ = 0;
    int64_t padW_ = 0;

    int64_t dilationD_ = 1;
    int64_t dilationH_ = 1;
    int64_t dilationW_ = 1;

    int64_t highAxisInner_ = 1;
    int64_t highAxisTail_ = 1;
    int64_t highAxisOuter_ = 1;
    int64_t highAxisActual_ = 1;

    int64_t dOutputInner_ = 1;
    int64_t dOutputTail_ = 1;
    int64_t dOutputOuter_ = 1;
    int64_t dOutputActual_ = 1;

    int64_t hOutputInner_ = 1;
    int64_t hOutputTail_ = 1;
    int64_t hOutputOuter_ = 1;
    int64_t hOutputActual_ = 1;

    int64_t wOutputInner_ = 1;
    int64_t wOutputTail_ = 1;
    int64_t wOutputOuter_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t wOutputAligned_ = 1;

    int64_t normalCoreProcessNum_ = 1;
    int64_t tailCoreProcessNum_ = 1;
    int64_t curCoreProcessNum_ = 1;
    int64_t usedCoreNum_ = 1;

    int64_t outputBufferSize_ = 1;
    int64_t gradBufferSize_ = 1;
    int64_t argmaxBufferSize_ = 1;

    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;

    int64_t hArgmaxActual_ = 0;
    int64_t dArgmaxActual_ = 0;
    int64_t wArgmaxActual_ = 0;
    int64_t wArgmaxAligned_ = 0;

    int64_t highAxisArgmaxOffset_ = 0;
    int64_t hAxisArgmaxOffset_ = 0;
    int64_t dAxisArgmaxOffset_ = 0;
    int64_t wAxisArgmaxOffset_ = 0;

    int64_t argmaxPlaneSize_ = 1;

    int64_t dProBatchSize_ = 1;
    int64_t hProBatchSize_ = 1;
    int64_t wProBatchSize_ = 1;
    int64_t curDProBatchSize_ = 1;
    int64_t curHProBatchSize_ = 1;
    int64_t curWProBatchSize_ = 1;
    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();

    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK =
        BLOCK_SIZE / sizeof(T1) >= BLOCK_SIZE / sizeof(T2) ? BLOCK_SIZE / sizeof(T1) : BLOCK_SIZE / sizeof(T2);
    constexpr static int64_t VREG_LENGTH_DATA_NUM_T2 = platform::GetVRegSize() / sizeof(T2);
};

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, IS_CHECK_RANGE>::ParseTilingData(
    const MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData& tilingData)
{
    dArgmax_ = tilingData.dArgmax;
    hArgmax_ = tilingData.hArgmax;
    wArgmax_ = tilingData.wArgmax;

    dOutput_ = tilingData.dOutput;
    hOutput_ = tilingData.hOutput;
    wOutput_ = tilingData.wOutput;

    kernelD_ = tilingData.dKernel;
    kernelH_ = tilingData.hKernel;
    kernelW_ = tilingData.wKernel;

    strideD_ = tilingData.dStride;
    strideH_ = tilingData.hStride;
    strideW_ = tilingData.wStride;

    padD_ = tilingData.padD;
    padH_ = tilingData.padH;
    padW_ = tilingData.padW;

    dilationD_ = tilingData.dilationD;
    dilationH_ = tilingData.dilationH;
    dilationW_ = tilingData.dilationW;

    highAxisInner_ = tilingData.highAxisInner;
    highAxisTail_ = tilingData.highAxisTail;
    highAxisOuter_ = tilingData.highAxisOuter;

    dOutputInner_ = tilingData.dOutputInner;
    dOutputTail_ = tilingData.dOutputTail;
    dOutputOuter_ = tilingData.dOutputOuter;

    hOutputInner_ = tilingData.hOutputInner;
    hOutputTail_ = tilingData.hOutputTail;
    hOutputOuter_ = tilingData.hOutputOuter;

    wOutputInner_ = tilingData.wOutputInner;
    wOutputTail_ = tilingData.wOutputTail;
    wOutputOuter_ = tilingData.wOutputOuter;

    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
    usedCoreNum_ = tilingData.usedCoreNum;

    outputBufferSize_ = tilingData.outputBufferSize;
    gradBufferSize_ = tilingData.gradBufferSize;
    argmaxBufferSize_ = tilingData.argmaxBufferSize;

    dProBatchSize_ = tilingData.dProBatchSize;
    hProBatchSize_ = tilingData.hProBatchSize;
    wProBatchSize_ = tilingData.wProBatchSize;
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, IS_CHECK_RANGE>::Init(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
    const MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData& tilingData)
{
    ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();
    argmaxPlaneSize_ = dArgmax_ * hArgmax_ * wArgmax_;
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    curCoreProcessNum_ = (blockIdx_ + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;
    gradGm_.SetGlobalBuffer((__gm__ T1*)grad);
    argmaxGm_.SetGlobalBuffer((__gm__ T2*)argmax);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);

    pipe_ = pipeIn;
    pipe_.InitBuffer(outputQue_, BUFFER_NUM, outputBufferSize_);
    pipe_.InitBuffer(gradQue_, BUFFER_NUM, gradBufferSize_);
    pipe_.InitBuffer(argmaxQue_, BUFFER_NUM, argmaxBufferSize_);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, IS_CHECK_RANGE>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;
    highAxisIndex_ = baseBlockIdx / (dOutputOuter_ * hOutputOuter_ * wOutputOuter_);
    highAxisActual_ = highAxisIndex_ == (highAxisOuter_ - 1) ? highAxisTail_ : highAxisInner_;
    int64_t tempTail = baseBlockIdx - (dOutputOuter_ * hOutputOuter_ * wOutputOuter_) * highAxisIndex_;
    dAxisIndex_ = tempTail / (hOutputOuter_ * wOutputOuter_);
    dOutputActual_ = dAxisIndex_ == (dOutputOuter_ - 1) ? dOutputTail_ : dOutputInner_;
    int64_t tempTail2 = tempTail - (hOutputOuter_ * wOutputOuter_) * dAxisIndex_;
    hAxisIndex_ = tempTail2 / wOutputOuter_;
    hOutputActual_ = hAxisIndex_ == (hOutputOuter_ - 1) ? hOutputTail_ : hOutputInner_;
    wAxisIndex_ = tempTail2 - wOutputOuter_ * hAxisIndex_;
    wOutputActual_ = wAxisIndex_ == (wOutputOuter_ - 1) ? wOutputTail_ : wOutputInner_;

    wOutputAligned_ =
        (wOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK * MAX_DATA_NUM_IN_ONE_BLOCK;
    int64_t dArgmaxActualStart = PStart(dAxisIndex_ * dOutputInner_, padD_, kernelD_, dilationD_, strideD_);
    int64_t dArgmaxActualEnd = PEnd(dAxisIndex_ * dOutputInner_ + dOutputActual_ - 1, padD_, strideD_, dArgmax_);
    int64_t hArgmaxActualStart = PStart(hAxisIndex_ * hOutputInner_, padH_, kernelH_, dilationH_, strideH_);
    int64_t hArgmaxActualEnd = PEnd(hAxisIndex_ * hOutputInner_ + hOutputActual_ - 1, padH_, strideH_, hArgmax_);
    int64_t wArgmaxActualStart = PStart(wAxisIndex_ * wOutputInner_, padW_, kernelW_, dilationW_, strideW_);
    int64_t wArgmaxActualEnd = PEnd(wAxisIndex_ * wOutputInner_ + wOutputActual_ - 1, padW_, strideW_, wArgmax_);
    wArgmaxActual_ = wArgmaxActualEnd - wArgmaxActualStart;
    wArgmaxAligned_ =
        (wArgmaxActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK * MAX_DATA_NUM_IN_ONE_BLOCK;
    hArgmaxActual_ = hArgmaxActualEnd - hArgmaxActualStart;
    dArgmaxActual_ = dArgmaxActualEnd - dArgmaxActualStart;

    curDProBatchSize_ = dProBatchSize_ > dArgmaxActual_ ? dArgmaxActual_ : dProBatchSize_;
    curHProBatchSize_ = hProBatchSize_ > hArgmaxActual_ ? hArgmaxActual_ : hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_ > wArgmaxActual_ ? wArgmaxActual_ : wProBatchSize_;
    highAxisArgmaxOffset_ = highAxisIndex_ * highAxisInner_ * argmaxPlaneSize_;
    dAxisArgmaxOffset_ = dArgmaxActualStart * hArgmax_ * wArgmax_;
    hAxisArgmaxOffset_ = hArgmaxActualStart * wArgmax_;
    wAxisArgmaxOffset_ = wArgmaxActualStart;
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, IS_CHECK_RANGE>::CopyIn()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();
    int64_t planeHW = hArgmax_ * wArgmax_;
    int64_t argmaxGmOffset = highAxisArgmaxOffset_ + dAxisArgmaxOffset_ + hAxisArgmaxOffset_ + wAxisArgmaxOffset_;
    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = dArgmaxActual_;
    loopModeParamsT1.loop2Size = highAxisActual_;
    loopModeParamsT1.loop1SrcStride = planeHW * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = argmaxPlaneSize_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = hArgmaxActual_ * wArgmaxAligned_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = dArgmaxActual_ * hArgmaxActual_ * wArgmaxAligned_ * sizeof(T1);

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(hArgmaxActual_), static_cast<uint32_t>(wArgmaxActual_ * sizeof(T1)),
        static_cast<uint32_t>((wArgmax_ - wArgmaxActual_) * sizeof(T1)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)};

    DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyOutParamT1, paramsT1);

    DataCopyPadExtParams<T2> paramsT2 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT2;
    loopModeParamsT2.loop1Size = dArgmaxActual_;
    loopModeParamsT2.loop2Size = highAxisActual_;
    loopModeParamsT2.loop1SrcStride = planeHW * sizeof(T2);
    loopModeParamsT2.loop2SrcStride = argmaxPlaneSize_ * sizeof(T2);
    loopModeParamsT2.loop1DstStride = hArgmaxActual_ * wArgmaxAligned_ * sizeof(T2);
    loopModeParamsT2.loop2DstStride = dArgmaxActual_ * hArgmaxActual_ * wArgmaxAligned_ * sizeof(T2);

    uint32_t dstStrideT2 = (wArgmaxAligned_ - wArgmaxActual_) * sizeof(T2) / BLOCK_SIZE;
    SetLoopModePara(loopModeParamsT2, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT2 = {
        static_cast<uint16_t>(hArgmaxActual_), static_cast<uint32_t>(wArgmaxActual_ * sizeof(T2)),
        static_cast<uint32_t>((wArgmax_ - wArgmaxActual_) * sizeof(T2)), static_cast<uint32_t>(dstStrideT2),
        static_cast<uint32_t>(0)};

    DataCopyPad(argmaxLocal, argmaxGm_[argmaxGmOffset], copyOutParamT2, paramsT2);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
    argmaxQue_.EnQue(argmaxLocal);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, IS_CHECK_RANGE>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();
    int64_t outputPlaneSize = hOutput_ * wOutput_;
    int64_t outputPlaneDHW = dOutput_ * outputPlaneSize;
    int64_t ncBase = highAxisIndex_ * highAxisInner_ * outputPlaneDHW;
    int64_t dBase = dAxisIndex_ * dOutputInner_ * outputPlaneSize;
    int64_t hBase = hAxisIndex_ * hOutputInner_ * wOutput_;
    int64_t wBase = wAxisIndex_ * wOutputInner_;
    int64_t outputGmOffset = ncBase + dBase + hBase + wBase;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = dOutputActual_;
    loopModeParamsT1.loop2Size = highAxisActual_;
    loopModeParamsT1.loop1SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = dOutputActual_ * hOutputActual_ * wOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = outputPlaneSize * sizeof(T1);
    loopModeParamsT1.loop2DstStride = outputPlaneDHW * sizeof(T1);

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(hOutputActual_), static_cast<uint32_t>(wOutputActual_ * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>((wOutput_ - wOutputActual_) * sizeof(T1)),
        static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}
} // namespace MaxPool3DGradWithArgmaxNCDHWNameSpace
#endif // MAX_POOL_GRAD_WITH_ARGMAX_SIMD_H_