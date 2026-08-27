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
 * \file max_pool3d_grad_small_kernel_scatter.h
 * \brief
 */

#ifndef MAX_POOL3D_GRAD_SMALL_KERNEL_SCATTER_H
#define MAX_POOL3D_GRAD_SMALL_KERNEL_SCATTER_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../../pool_3d_common/arch35/pool_3d_common.h"

using namespace AscendC;
using Pool3D::FastDivImpl;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t DOUBLE = 2;
constexpr uint32_t HELP_BUFFER = 5120;

constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;
constexpr uint32_t INDEX_FOUR = 4;
constexpr uint32_t INDEX_FIVE = 5;
constexpr uint32_t INDEX_SIX = 6;
constexpr uint32_t INDEX_SEVEN = 7;
using computeType = float;

constexpr uint32_t VER_NORMAL = 0;
constexpr uint32_t VER_V3 = 1;

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
__aicore__ inline Reg::MaskReg GenT2Mask(uint32_t& maskCount)
{
    Reg::MaskReg reg;
    if constexpr (std::is_same<T2, int64_t>::value) {
        reg = AscendC::Reg::UpdateMask<T2, AscendC::Reg::RegTraitNumTwo>(maskCount);
    } else {
        reg = AscendC::Reg::UpdateMask<T2>(maskCount);
    }
    return reg;
}

template <typename T>
__aicore__ inline void GradientAcc(__local_mem__ computeType* yAddr, Reg::RegTensor<computeType>& gradReg,
                                   Reg::RegTensor<T>& argmaxReg, Reg::MaskReg& pregArgmax)
{
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    AscendC::Reg::RegTensor<computeType> scatterAccResReg;
    AscendC::Reg::DataCopyGather(scatterAccResReg, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
    AscendC::Reg::Add(scatterAccResReg, scatterAccResReg, gradReg, pregArgmax);
    AscendC::Reg::DataCopyScatter(yAddr, scatterAccResReg, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
}

template <typename T1, typename T2>
__aicore__ inline void GetConCurrentInput(Reg::RegTensor<int32_t>& argmaxReg, Reg::RegTensor<computeType>& gradReg,
                                          __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
                                          Reg::RegTensor<uint32_t>& parallelRegIndex,
                                          Reg::RegTensor<uint32_t>& parallelRegGrad, Reg::MaskReg& pregT1,
                                          Reg::MaskReg& pregT2)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::Reg::RegTensor<T1> gradRegT1;
        AscendC::Reg::RegTensor<uint16_t> parallelRegGradU16;
        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Cast<uint16_t, uint32_t, castTraitU32U16>(parallelRegGradU16, parallelRegGrad, allMaskU32);
        AscendC::Reg::Pack(parallelRegGradU16, (AscendC::Reg::RegTensor<int32_t>&)parallelRegGradU16);
        AscendC::Reg::DataCopyGather(gradRegT1, gradAddr, parallelRegGradU16, pregT1);
        AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint32_t>&)gradRegT1,
                             (AscendC::Reg::RegTensor<uint16_t>&)gradRegT1);
        AscendC::Reg::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::Reg::DataCopyGather(gradReg, gradAddr, parallelRegGrad, pregT1);
    }

    if constexpr (std::is_same<T2, int32_t>::value) {
        AscendC::Reg::DataCopyGather(argmaxReg, argmaxAddr, parallelRegIndex, pregT2);
    } else if constexpr (std::is_same<T2, int64_t>::value) {
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> argmaxRegTwo;
        AscendC::Reg::DataCopyGather(argmaxRegTwo, argmaxAddr, parallelRegIndex, pregT2);
        argmaxReg = (AscendC::Reg::RegTensor<int32_t>&)argmaxRegTwo.reg[0];
    }
}

namespace MaxPool3DSmallKernelNameSpace {
template <const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNcdhwFastDiv(Reg::RegTensor<int32_t>& argmaxReg, Reg::RegTensor<uint32_t>& dTmpReg,
                                             Reg::RegTensor<uint32_t>& hTmpReg, Reg::RegTensor<uint32_t>& wTmpReg,
                                             Reg::RegTensor<uint32_t>& magicHWReg, int16_t shiftHW,
                                             Reg::RegTensor<uint32_t>& magicWReg, int16_t shiftW,
                                             int32_t hwOutputAligned, int32_t wOutputAligned, int32_t wOutput,
                                             int32_t hwOutput, int32_t baseOffset, int32_t highOutputPlaneActual,
                                             int32_t highArgmaxPlaneActual, Reg::RegTensor<uint32_t>& magicHighReg,
                                             int16_t shiftHigh)
{
    Reg::MaskReg allMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> remU32;

    FastDivImpl(dTmpReg, (Reg::RegTensor<uint32_t>&)argmaxReg, magicHWReg, shiftHW, allMask);
    Reg::Muls(remU32, dTmpReg, uint32_t(hwOutput), allMask);
    Reg::Sub(remU32, (Reg::RegTensor<uint32_t>&)argmaxReg, remU32, allMask);

    FastDivImpl(hTmpReg, remU32, magicWReg, shiftW, allMask);
    Reg::Muls(wTmpReg, hTmpReg, uint32_t(wOutput), allMask);
    Reg::Sub(wTmpReg, remU32, wTmpReg, allMask);

    Reg::Muls(argmaxReg, (Reg::RegTensor<int32_t>&)hTmpReg, int32_t(wOutputAligned), allMask);
    Reg::Add(argmaxReg, argmaxReg, (Reg::RegTensor<int32_t>&)wTmpReg, allMask);
    Reg::RegTensor<int32_t> dhwTmpIndexReg;
    Reg::Muls(dhwTmpIndexReg, (Reg::RegTensor<int32_t>&)dTmpReg, int32_t(hwOutputAligned), allMask);
    Reg::Add(argmaxReg, argmaxReg, dhwTmpIndexReg, allMask);
    Reg::Adds(argmaxReg, argmaxReg, baseOffset, allMask);

    if constexpr (IS_MUL_NC == 1) {
        Reg::RegTensor<int32_t> highIncRegI32;
        Reg::Arange(highIncRegI32, 0);
        Reg::RegTensor<uint32_t> highIncReg;
        FastDivImpl((Reg::RegTensor<uint32_t>&)highIncReg, (Reg::RegTensor<uint32_t>&)highIncRegI32, magicHighReg,
                    shiftHigh, allMask);
        Reg::Muls(highIncRegI32, (Reg::RegTensor<int32_t>&)highIncReg, highOutputPlaneActual, allMask);
        Reg::Add(argmaxReg, argmaxReg, highIncRegI32, allMask);
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

__aicore__ inline void FilterMask3D(Reg::MaskReg& preg, Reg::RegTensor<uint32_t>& dTmpReg,
                                    Reg::RegTensor<uint32_t>& hTmpReg, Reg::RegTensor<uint32_t>& wTmpReg,
                                    Reg::RegTensor<int32_t>& dLowerReg, Reg::RegTensor<int32_t>& hLowerReg,
                                    Reg::RegTensor<int32_t>& wLowerReg, Reg::RegTensor<int32_t>& dUpperReg,
                                    Reg::RegTensor<int32_t>& hUpperReg, Reg::RegTensor<int32_t>& wUpperReg)
{
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg hMask;
    AscendC::Reg::MaskReg wMask;
    AscendC::Reg::MaskReg dMask;
    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(hMask, (AscendC::Reg::RegTensor<int32_t>&)hTmpReg, hLowerReg, allMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(wMask, (AscendC::Reg::RegTensor<int32_t>&)wTmpReg, wLowerReg, allMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(dMask, (AscendC::Reg::RegTensor<int32_t>&)dTmpReg, dLowerReg, allMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(hMask, hUpperReg, (AscendC::Reg::RegTensor<int32_t>&)hTmpReg, hMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(wMask, wUpperReg, (AscendC::Reg::RegTensor<int32_t>&)wTmpReg, wMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(dMask, dUpperReg, (AscendC::Reg::RegTensor<int32_t>&)dTmpReg, dMask);
    AscendC::Reg::MaskAnd(hMask, hMask, wMask, allMask);
    AscendC::Reg::MaskAnd(dMask, dMask, hMask, allMask);
    AscendC::Reg::MaskAnd(preg, preg, dMask, allMask);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNchwFastDiv(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    Reg::RegTensor<uint32_t>& parallelRegIndex, Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
    Reg::RegTensor<uint32_t>& magicHWReg, int16_t shiftHW, Reg::RegTensor<uint32_t>& magicWReg, int16_t shiftW,
    int32_t hwOutputAligned, int32_t wOutputAligned, int32_t wOutput, int32_t hwOutput, int32_t baseOffset,
    Reg::RegTensor<int32_t>& dLowerReg, Reg::RegTensor<int32_t>& hLowerReg, Reg::RegTensor<int32_t>& wLowerReg,
    Reg::RegTensor<int32_t>& dUpperReg, Reg::RegTensor<int32_t>& hUpperReg, Reg::RegTensor<int32_t>& wUpperReg)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> argmaxReg;
    AscendC::Reg::RegTensor<uint32_t> dTmpReg;
    AscendC::Reg::RegTensor<uint32_t> hTmpReg;
    AscendC::Reg::RegTensor<uint32_t> wTmpReg;

    Reg::RegTensor<uint32_t> dummyMagicHighReg;
    int16_t dummyShiftHigh = 0;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<T2>(maskT2);

    GetConCurrentInput<T1, T2>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1,
                               pregT2);
    IndexConvNcdhwFastDiv<0>(argmaxReg, dTmpReg, hTmpReg, wTmpReg, magicHWReg, shiftHW, magicWReg, shiftW,
                             hwOutputAligned, wOutputAligned, wOutput, hwOutput, baseOffset, 0, 0, dummyMagicHighReg,
                             dummyShiftHigh);
    if constexpr (std::is_same<T2, int32_t>::value) {
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask3D(pregT2, dTmpReg, hTmpReg, wTmpReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                         wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregT2);
    } else {
        uint32_t argmaxMask = argmaxMaskCount;
        AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask3D(pregArgmax, dTmpReg, hTmpReg, wTmpReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                         wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
    }
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNcdhwFastDiv(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    Reg::RegTensor<uint32_t>& parallelRegIndex, Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
    Reg::RegTensor<uint32_t>& magicHWReg, int16_t shiftHW, Reg::RegTensor<uint32_t>& magicWReg, int16_t shiftW,
    int32_t hwOutputAligned, int32_t wOutputAligned, int32_t wOutput, int32_t hwOutput, int32_t baseOffset,
    Reg::RegTensor<int32_t>& dLowerReg, Reg::RegTensor<int32_t>& hLowerReg, Reg::RegTensor<int32_t>& wLowerReg,
    Reg::RegTensor<int32_t>& dUpperReg, Reg::RegTensor<int32_t>& hUpperReg, Reg::RegTensor<int32_t>& wUpperReg)
{
    DoSingleNCNchwFastDiv<T1, T2, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad,
                                                  argmaxMaskCount, magicHWReg, shiftHW, magicWReg, shiftW,
                                                  hwOutputAligned, wOutputAligned, wOutput, hwOutput, baseOffset,
                                                  dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg, wUpperReg);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulNCNcdhwFastDiv(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    Reg::RegTensor<uint32_t>& parallelRegIndex, Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
    Reg::RegTensor<uint32_t>& magicHWReg, int16_t shiftHW, Reg::RegTensor<uint32_t>& magicWReg, int16_t shiftW,
    int32_t hwOutputAligned, int32_t wOutputAligned, int32_t wOutput, int32_t hwOutput, int32_t baseOffset,
    Reg::RegTensor<int32_t>& dLowerReg, Reg::RegTensor<int32_t>& hLowerReg, Reg::RegTensor<int32_t>& wLowerReg,
    Reg::RegTensor<int32_t>& dUpperReg, Reg::RegTensor<int32_t>& hUpperReg, Reg::RegTensor<int32_t>& wUpperReg,
    int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual, Reg::RegTensor<uint32_t>& magicHighReg,
    int16_t shiftHigh)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> argmaxReg;
    AscendC::Reg::RegTensor<uint32_t> dTmpReg;
    AscendC::Reg::RegTensor<uint32_t> hTmpReg;
    AscendC::Reg::RegTensor<uint32_t> wTmpReg;
    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<T2>(maskT2);
    GetConCurrentInput<T1, T2>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1,
                               pregT2);

    IndexConvNcdhwFastDiv<1>(argmaxReg, dTmpReg, hTmpReg, wTmpReg, magicHWReg, shiftHW, magicWReg, shiftW,
                             hwOutputAligned, wOutputAligned, wOutput, hwOutput, baseOffset, highOutputPlaneActual,
                             highArgmaxPlaneActual, magicHighReg, shiftHigh);

    if constexpr (std::is_same<T2, int32_t>::value) {
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask3D(pregT2, dTmpReg, hTmpReg, wTmpReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                         wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregT2);
    } else {
        uint32_t argmaxMask = argmaxMaskCount;
        AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask3D(pregArgmax, dTmpReg, hTmpReg, wTmpReg, dLowerReg, hLowerReg, wLowerReg, dUpperReg, hUpperReg,
                         wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
    }
}

template <typename T>
__aicore__ inline void GenInitial1DIndices(Reg::RegTensor<T>& indexReg, int64_t colGenRate)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, T(colGenRate), preg);
}

template <typename T>
__aicore__ inline void GenInitial2DHighIndices(Reg::RegTensor<T>& indexReg, int64_t highStride, int64_t colGenRate,
                                               int64_t fullBatchColNum)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(highStride), preg);
    AscendC::Reg::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void Gen2DHighIndexOne(Reg::RegTensor<T>& indexReg, int64_t highStride)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, T(highStride), preg);
}

template <typename T>
__aicore__ inline void GenInitial2DIndices(Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}
template <typename T>
__aicore__ inline void DhwGenInitial2DIndices(Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                              int64_t colNumAligned, int64_t fullBatchColNum)
{
    GenInitial2DIndices<T>(indexReg, colGenRate, rowGenRate, colNumAligned, fullBatchColNum);
}

template <typename T>
__aicore__ inline void Gen2DIndexOne(Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, T(rowGenRate * colNumAligned), preg);
}

template <typename T>
__aicore__ inline void DhwGen2DIndexOne(Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    Gen2DIndexOne<T>(indexReg, rowGenRate, colNumAligned);
}

template <typename T>
__aicore__ inline void GenInitial3DIndices(Reg::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate,
                                           int64_t colGenRate, int64_t fullBatchRowNum, int64_t rowNumCount,
                                           int64_t fullBatchColNum, int64_t colNumAligned)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::Div(segmentScalarReg2, segmentIncReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg2, segmentScalarReg2, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::Reg::Muls(segmentIncReg2, segmentIncReg2, colGenRate, preg);

    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOne(Reg::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate,
                                     int64_t colNumAligned, int64_t fullBatchRowNum, int64_t rowNumCount)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);

    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial3DHighIndices(Reg::RegTensor<T>& indexReg, int64_t highStride, int64_t colGenRate,
                                               int64_t rowGenRate, int64_t colNumAligned, int64_t fullBatchColNum,
                                               int64_t fullBatchRowNum)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> hwReg;
    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::RegTensor<T> wReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;
    const uint64_t wStride = colGenRate;

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum));
    AscendC::Reg::Div(highReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, highReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::Reg::Sub(hwReg, indexReg, tmpReg, preg);

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::Div(hReg, hwReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, hReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(wReg, hwReg, tmpReg, preg);

    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;
    AscendC::Reg::RegTensor<T> wPartReg;
    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::Reg::Muls(wPartReg, wReg, T(wStride), preg);

    AscendC::Reg::Add(indexReg, highPartReg, hPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DHighIndexOne(Reg::RegTensor<T>& indexReg, int64_t highStride, int64_t rowGenRate,
                                         int64_t colNumAligned, int64_t fullBatchRowNum)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;

    AscendC::Reg::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::Reg::Div(highReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, highReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(hReg, indexReg, tmpReg, preg);

    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;
    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::Reg::Add(indexReg, highPartReg, hPartReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial4DIndices(Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum, int64_t fullBatchRowNum,
                                           int64_t fullBatchDepthNum, int64_t depthStride, int64_t highStride)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> dReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::RegTensor<T> dhwReg;
    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum * fullBatchDepthNum));
    AscendC::Reg::Div(highReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(dReg, highReg, T(fullBatchColNum * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::Reg::Sub(dhwReg, indexReg, dReg, preg);

    AscendC::Reg::RegTensor<T> hwReg;
    AscendC::Reg::Duplicate(constReg, T(fullBatchRowNum * fullBatchColNum));
    AscendC::Reg::Div(dReg, dhwReg, constReg, preg);
    AscendC::Reg::Muls(hwReg, dReg, T(fullBatchRowNum * fullBatchColNum), preg);
    AscendC::Reg::Sub(hwReg, dhwReg, hwReg, preg);

    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::RegTensor<T> wReg;

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::Div(hReg, hwReg, constReg, preg);
    AscendC::Reg::Muls(wReg, hReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(wReg, hwReg, wReg, preg);

    // 组装offset
    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> dPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;
    AscendC::Reg::RegTensor<T> wPartReg;

    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Muls(wPartReg, wReg, T(colGenRate), preg);
    AscendC::Reg::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, hPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen4DIndexOne(Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned,
                                     int64_t fullBatchRowNum, int64_t fullBatchDepthNum, int64_t depthStride,
                                     int64_t highStride)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> dReg;
    AscendC::Reg::RegTensor<T> dhwReg;
    AscendC::Reg::RegTensor<T> constReg;

    AscendC::Reg::Duplicate(constReg, T(1 * fullBatchRowNum * fullBatchDepthNum));
    AscendC::Reg::Div(highReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(dReg, highReg, T(1 * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::Reg::Sub(dhwReg, indexReg, dReg, preg);

    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> hReg;

    AscendC::Reg::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::Reg::Div(dReg, dhwReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, dReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(hReg, dhwReg, tmpReg, preg);
    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> dPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;

    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, hPartReg, preg);
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

__aicore__ inline void FastDivInt32(Reg::RegTensor<int32_t>& res, Reg::RegTensor<int32_t>& src, const DivMagic& dm)
{
    Reg::RegTensor<uint32_t> tmp;
    Reg::RegTensor<uint32_t> magicReg;
    Reg::Duplicate(magicReg, dm.magic);
    Reg::MaskReg allMask = Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    FastDivImpl(tmp, (Reg::RegTensor<uint32_t>&)src, magicReg, dm.shift, allMask);
    res = (Reg::RegTensor<int32_t>&)tmp;
}

template <typename T>
__aicore__ inline void GenInitial2DHighIndicesFast(Reg::RegTensor<T>& indexReg, int64_t highStride, int64_t colGenRate,
                                                   int64_t fullBatchColNum, const DivMagic& divW)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    FastDivInt32(segmentScalarReg, indexReg, divW);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(highStride), preg);
    AscendC::Reg::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial2DIndicesFast(Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                               int64_t colNumAligned, int64_t fullBatchColNum, const DivMagic& divW)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    FastDivInt32(segmentScalarReg, indexReg, divW);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void DhwGenInitial2DIndicesFast(Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                                  int64_t colNumAligned, int64_t fullBatchColNum, const DivMagic& divW)
{
    GenInitial2DIndicesFast<T>(indexReg, colGenRate, rowGenRate, colNumAligned, fullBatchColNum, divW);
}

template <typename T>
__aicore__ inline void GenInitial3DIndicesFast(Reg::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate,
                                               int64_t colGenRate, int64_t fullBatchRowNum, int64_t rowNumCount,
                                               int64_t fullBatchColNum, int64_t colNumAligned, const DivMagic& divWH,
                                               const DivMagic& divW)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    FastDivInt32(segmentScalarReg, indexReg, divWH);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);

    FastDivInt32(segmentScalarReg2, segmentIncReg, divW);
    AscendC::Reg::Muls(segmentIncReg2, segmentScalarReg2, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::Reg::Muls(segmentIncReg2, segmentIncReg2, colGenRate, preg);

    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOneFast(Reg::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate,
                                         int64_t colNumAligned, int64_t fullBatchRowNum, int64_t rowNumCount,
                                         const DivMagic& divH)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    FastDivInt32(segmentScalarReg, indexReg, divH);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(dGenRate * rowNumCount * colNumAligned), preg);
    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial3DHighIndicesFast(Reg::RegTensor<T>& indexReg, int64_t highStride, int64_t colGenRate,
                                                   int64_t rowGenRate, int64_t colNumAligned, int64_t fullBatchColNum,
                                                   int64_t fullBatchRowNum, const DivMagic& divWH, const DivMagic& divW)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> hwReg;
    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::RegTensor<T> wReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;
    const uint64_t wStride = colGenRate;

    FastDivInt32(highReg, indexReg, divWH);
    AscendC::Reg::Muls(tmpReg, highReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::Reg::Sub(hwReg, indexReg, tmpReg, preg);

    FastDivInt32(hReg, hwReg, divW);
    AscendC::Reg::Muls(tmpReg, hReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(wReg, hwReg, tmpReg, preg);

    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;
    AscendC::Reg::RegTensor<T> wPartReg;
    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::Reg::Muls(wPartReg, wReg, T(wStride), preg);

    AscendC::Reg::Add(indexReg, highPartReg, hPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DHighIndexOneFast(Reg::RegTensor<T>& indexReg, int64_t highStride, int64_t rowGenRate,
                                             int64_t colNumAligned, int64_t fullBatchRowNum, const DivMagic& divH)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    const uint64_t hStride = rowGenRate * colNumAligned;

    FastDivInt32(highReg, indexReg, divH);
    AscendC::Reg::Muls(tmpReg, highReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(hReg, indexReg, tmpReg, preg);

    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;
    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(hStride), preg);
    AscendC::Reg::Add(indexReg, highPartReg, hPartReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial4DIndicesFast(Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                               int64_t colNumAligned, int64_t fullBatchColNum, int64_t fullBatchRowNum,
                                               int64_t fullBatchDepthNum, int64_t depthStride, int64_t highStride,
                                               const DivMagic& divDHW, const DivMagic& divHW, const DivMagic& divW)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> dReg;
    AscendC::Reg::RegTensor<T> dhwReg;
    AscendC::Reg::RegTensor<T> hwReg;
    AscendC::Reg::RegTensor<T> hReg;
    AscendC::Reg::RegTensor<T> wReg;
    AscendC::Reg::RegTensor<T> tmpReg;

    FastDivInt32(highReg, indexReg, divDHW);
    AscendC::Reg::Muls(dReg, highReg, T(fullBatchColNum * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::Reg::Sub(dhwReg, indexReg, dReg, preg);

    FastDivInt32(dReg, dhwReg, divHW);
    AscendC::Reg::Muls(hwReg, dReg, T(fullBatchRowNum * fullBatchColNum), preg);
    AscendC::Reg::Sub(hwReg, dhwReg, hwReg, preg);

    FastDivInt32(hReg, hwReg, divW);
    AscendC::Reg::Muls(wReg, hReg, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(wReg, hwReg, wReg, preg);

    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> dPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;
    AscendC::Reg::RegTensor<T> wPartReg;

    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Muls(wPartReg, wReg, T(colGenRate), preg);
    AscendC::Reg::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, hPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, wPartReg, preg);
}

template <typename T>
__aicore__ inline void Gen4DIndexOneFast(Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned,
                                         int64_t fullBatchRowNum, int64_t fullBatchDepthNum, int64_t depthStride,
                                         int64_t highStride, const DivMagic& divHD, const DivMagic& divH)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<T> highReg;
    AscendC::Reg::RegTensor<T> dReg;
    AscendC::Reg::RegTensor<T> dhwReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> hReg;

    FastDivInt32(highReg, indexReg, divHD);
    AscendC::Reg::Muls(dReg, highReg, T(1 * fullBatchRowNum * fullBatchDepthNum), preg);
    AscendC::Reg::Sub(dhwReg, indexReg, dReg, preg);

    FastDivInt32(dReg, dhwReg, divH);
    AscendC::Reg::Muls(tmpReg, dReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(hReg, dhwReg, tmpReg, preg);

    AscendC::Reg::RegTensor<T> highPartReg;
    AscendC::Reg::RegTensor<T> dPartReg;
    AscendC::Reg::RegTensor<T> hPartReg;

    AscendC::Reg::Muls(highPartReg, highReg, T(highStride), preg);
    AscendC::Reg::Muls(dPartReg, dReg, T(depthStride), preg);
    AscendC::Reg::Muls(hPartReg, hReg, T(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Add(indexReg, highPartReg, dPartReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, hPartReg, preg);
}
} // namespace MaxPool3DSmallKernelNameSpace
#endif // MAX_POOL3D_GRAD_SMALL_KERNEL_SCATTER_H
