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

using namespace AscendC;
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
template <typename T2, typename T3>
__aicore__ inline Reg::MaskReg GenT2Mask(uint32_t& maskCount)
{
    Reg::MaskReg reg;
    if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int64_t>::value) {
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
    AscendC::Reg::RegTensor<computeType> scatterAccResReg;
    AscendC::Reg::DataCopyGather(scatterAccResReg, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
    AscendC::Reg::Add(scatterAccResReg, scatterAccResReg, gradReg, pregArgmax);
    AscendC::Reg::DataCopyScatter(yAddr, scatterAccResReg, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void GetConCurrentInput(Reg::RegTensor<T3>& argmaxReg, Reg::RegTensor<computeType>& gradReg,
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

    if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int32_t>::value) {
        AscendC::Reg::DataCopyGather(argmaxReg, argmaxAddr, parallelRegIndex, pregT2);
    } else if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int64_t>::value) {
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> argmaxRegTwo;
        AscendC::Reg::DataCopyGather(argmaxRegTwo, argmaxAddr, parallelRegIndex, pregT2);
        argmaxReg = (AscendC::Reg::RegTensor<T3>&)argmaxRegTwo.reg[0];
    } else if constexpr (std::is_same<T3, int64_t>::value && std::is_same<T2, int64_t>::value) {
        AscendC::Reg::DataCopyGather(argmaxReg, argmaxAddr, parallelRegIndex, pregT2);
    }
}

namespace MaxPool3DSmallKernelNameSpace {
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

template <typename T, const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNcdhw(Reg::RegTensor<T>& argmaxReg, Reg::RegTensor<int32_t>& dIndexReg,
                                      Reg::RegTensor<int32_t>& hIndexReg, Reg::RegTensor<int32_t>& wIndexReg,
                                      Reg::RegTensor<T>& hwOutputConstReg, Reg::RegTensor<T>& wOutputConstReg,
                                      int64_t curDIndex, int64_t curHIndex, int64_t curWIndex, int32_t hOutputActual,
                                      int32_t wOutputAligned, int32_t highOutputOffset, int32_t highOutputPlaneActual,
                                      int32_t highArgmaxPlaneActual)
{
    AscendC::Reg::RegTensor<T> dTmpIndexReg;
    AscendC::Reg::RegTensor<T> hTmpIndexReg;
    AscendC::Reg::RegTensor<T> wTmpIndexReg;
    AscendC::Reg::RegTensor<int32_t> dhwTmpIndexReg;
    AscendC::Reg::RegTensor<T> remReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Div(dTmpIndexReg, argmaxReg, hwOutputConstReg, allMask);
    AscendC::Reg::Mul(remReg, dTmpIndexReg, hwOutputConstReg, allMask);
    AscendC::Reg::Sub(remReg, argmaxReg, remReg, allMask);
    AscendC::Reg::Div(hTmpIndexReg, remReg, wOutputConstReg, allMask);
    AscendC::Reg::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMask);
    AscendC::Reg::Sub(wTmpIndexReg, remReg, wTmpIndexReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::Reg::Adds(tmpReg, dTmpIndexReg, T(-curDIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(dIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)dIndexReg, (AscendC::Reg::RegTensor<int64_t>&)dIndexReg);

        AscendC::Reg::Adds(tmpReg, hTmpIndexReg, T(-curHIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(hIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)hIndexReg, (AscendC::Reg::RegTensor<int64_t>&)hIndexReg);

        AscendC::Reg::Adds(tmpReg, wTmpIndexReg, T(-curWIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(wIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)wIndexReg, (AscendC::Reg::RegTensor<int64_t>&)wIndexReg);
    } else {
        AscendC::Reg::Adds(dIndexReg, dTmpIndexReg, T(-curDIndex), allMask);
        AscendC::Reg::Adds(hIndexReg, hTmpIndexReg, T(-curHIndex), allMask);
        AscendC::Reg::Adds(wIndexReg, wTmpIndexReg, T(-curWIndex), allMask);
    }

    int32_t hwOutputAligned = hOutputActual * wOutputAligned;
    AscendC::Reg::Muls((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, hIndexReg, T(wOutputAligned), allMaskU32);

    AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                      wIndexReg, allMaskU32);

    AscendC::Reg::Adds((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                       highOutputOffset, allMaskU32);

    AscendC::Reg::Muls(dhwTmpIndexReg, dIndexReg, T(hwOutputAligned), allMaskU32);

    AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                      dhwTmpIndexReg, allMaskU32);

    if constexpr (IS_MUL_NC == 1) {
        AscendC::Reg::RegTensor<int32_t> highIncReg;
        AscendC::Reg::Arange(highIncReg, 0);
        AscendC::Reg::RegTensor<int32_t> constReg;
        AscendC::Reg::Duplicate(constReg, highArgmaxPlaneActual);
        AscendC::Reg::Div(highIncReg, highIncReg, constReg, allMaskU32);
        AscendC::Reg::Muls(highIncReg, highIncReg, highOutputPlaneActual, allMaskU32);
        AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                          highIncReg, allMaskU32);
    }
}

template <typename T, const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNchw(Reg::RegTensor<T>& argmaxReg, Reg::RegTensor<int32_t>& dIndexReg,
                                     Reg::RegTensor<int32_t>& hIndexReg, Reg::RegTensor<int32_t>& wIndexReg,
                                     Reg::RegTensor<T>& hwOutputConstReg, Reg::RegTensor<T>& wOutputConstReg,
                                     int64_t curDIndex, int64_t curHIndex, int64_t curWIndex, int32_t hOutputActual,
                                     int32_t wOutputAligned, int32_t highOutputOffset, int32_t highOutputPlaneActual,
                                     int32_t highArgmaxPlaneActual)
{
    AscendC::Reg::RegTensor<T> dTmpIndexReg;
    AscendC::Reg::RegTensor<T> hTmpIndexReg;
    AscendC::Reg::RegTensor<T> wTmpIndexReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Div(dTmpIndexReg, argmaxReg, hwOutputConstReg, allMask);

    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::Reg::Adds(tmpReg, dTmpIndexReg, T(-curDIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(dIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)dIndexReg, (AscendC::Reg::RegTensor<int64_t>&)dIndexReg);
    } else {
        AscendC::Reg::Adds(dIndexReg, dTmpIndexReg, T(-curDIndex), allMask);
    }

    AscendC::Reg::Mul(hTmpIndexReg, dTmpIndexReg, hwOutputConstReg, allMask);

    AscendC::Reg::Sub(dTmpIndexReg, argmaxReg, hTmpIndexReg, allMask);

    AscendC::Reg::Div(hTmpIndexReg, dTmpIndexReg, wOutputConstReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::Reg::Adds(tmpReg, hTmpIndexReg, T(-curHIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(hIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)hIndexReg, (AscendC::Reg::RegTensor<int64_t>&)hIndexReg);
    } else {
        AscendC::Reg::Adds(hIndexReg, hTmpIndexReg, T(-curHIndex), allMask);
    }

    AscendC::Reg::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMask);
    AscendC::Reg::Sub(wTmpIndexReg, dTmpIndexReg, wTmpIndexReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::Reg::Adds(tmpReg, wTmpIndexReg, T(-curWIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(wIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)wIndexReg, (AscendC::Reg::RegTensor<int64_t>&)wIndexReg);
    } else {
        AscendC::Reg::Adds(wIndexReg, wTmpIndexReg, T(-curWIndex), allMask);
    }

    AscendC::Reg::Muls((AscendC::Reg::RegTensor<int32_t>&)dTmpIndexReg, dIndexReg, T(wOutputAligned * hOutputActual),
                       allMaskU32);

    AscendC::Reg::Muls((AscendC::Reg::RegTensor<int32_t>&)hTmpIndexReg, hIndexReg, T(wOutputAligned), allMaskU32);
    AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)dTmpIndexReg,
                      hTmpIndexReg, allMaskU32);
    AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                      wIndexReg, allMaskU32);

    AscendC::Reg::Adds((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                       highOutputOffset, allMaskU32);

    if constexpr (IS_MUL_NC == 1) {
        AscendC::Reg::RegTensor<int32_t> highIncReg;
        AscendC::Reg::Arange(highIncReg, 0);
        AscendC::Reg::RegTensor<int32_t> constReg;
        AscendC::Reg::Duplicate(constReg, highArgmaxPlaneActual);
        AscendC::Reg::Div(highIncReg, highIncReg, constReg, allMaskU32);
        AscendC::Reg::Muls(highIncReg, highIncReg, highOutputPlaneActual, allMaskU32);
        AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                          highIncReg, allMaskU32);
    }
}
__aicore__ inline void FilterMask3D(Reg::MaskReg& preg, Reg::RegTensor<int32_t>& dIndexReg,
                                    Reg::RegTensor<int32_t>& hIndexReg, Reg::RegTensor<int32_t>& wIndexReg,
                                    Reg::RegTensor<int32_t>& zeroConstReg, Reg::RegTensor<int32_t>& dMaxReg,
                                    Reg::RegTensor<int32_t>& hMaxReg, Reg::RegTensor<int32_t>& wMaxReg)
{
    AscendC::Reg::MaskReg gtMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(gtMask, hIndexReg, zeroConstReg, gtMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(gtMask, hMaxReg, hIndexReg, gtMask);

    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);

    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(gtMask, dIndexReg, zeroConstReg, gtMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(gtMask, dMaxReg, dIndexReg, gtMask);
    AscendC::Reg::MaskAnd(preg, preg, gtMask, allMask);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNchw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                      __local_mem__ T2* argmaxAddr, Reg::RegTensor<uint32_t>& parallelRegIndex,
                                      Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                      Reg::RegTensor<T3>& hwOutputConstReg, Reg::RegTensor<T3>& wOutputConstReg,
                                      int64_t curDIndex, int64_t curHIndex, int64_t curWIndex, int32_t hOutputActual,
                                      int32_t wOutputAligned, int32_t highOutputOffset,
                                      Reg::RegTensor<int32_t>& zeroConstReg, Reg::RegTensor<int32_t>& wMaxReg,
                                      Reg::RegTensor<int32_t>& hMaxReg, Reg::RegTensor<int32_t>& dMaxReg)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<T3> argmaxReg;

    AscendC::Reg::RegTensor<int32_t> dIndexReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);

    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1,
                                   pregT2);
    IndexConvNcdhw<T3>(argmaxReg, dIndexReg, hIndexReg, wIndexReg, hwOutputConstReg, wOutputConstReg, curDIndex,
                       curHIndex, curWIndex, hOutputActual, wOutputAligned, highOutputOffset, 0, 0);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask3D(pregArgmax, dIndexReg, hIndexReg, wIndexReg, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
    }
    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNcdhw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                       __local_mem__ T2* argmaxAddr, Reg::RegTensor<uint32_t>& parallelRegIndex,
                                       Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                       Reg::RegTensor<T3>& hwOutputConstReg, Reg::RegTensor<T3>& wOutputConstReg,
                                       int64_t curDIndex, int64_t curHIndex, int64_t curWIndex, int32_t wOutputAligned,
                                       int32_t highOutputOffset, int32_t hOutputActual,
                                       Reg::RegTensor<int32_t>& zeroConstReg, Reg::RegTensor<int32_t>& dMaxReg,
                                       Reg::RegTensor<int32_t>& hMaxReg, Reg::RegTensor<int32_t>& wMaxReg)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<T3> argmaxReg;
    AscendC::Reg::RegTensor<int32_t> dIndexReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1,
                                   pregT2);
    IndexConvNcdhw<T3>(argmaxReg, dIndexReg, hIndexReg, wIndexReg, hwOutputConstReg, wOutputConstReg, curDIndex,
                       curHIndex, curWIndex, hOutputActual, wOutputAligned, highOutputOffset, 0, 0);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask3D(pregArgmax, dIndexReg, hIndexReg, wIndexReg, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulNCNcdhw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                    __local_mem__ T2* argmaxAddr, Reg::RegTensor<uint32_t>& parallelRegIndex,
                                    Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                    Reg::RegTensor<T3>& hwOutputConstReg, Reg::RegTensor<T3>& wOutputConstReg,
                                    int64_t curDIndex, int64_t curHIndex, int64_t curWIndex, int32_t wOutputAligned,
                                    int32_t highOutputOffset, int32_t hOutputActual,
                                    Reg::RegTensor<int32_t>& zeroConstReg, Reg::RegTensor<int32_t>& dMaxReg,
                                    Reg::RegTensor<int32_t>& hMaxReg, Reg::RegTensor<int32_t>& wMaxReg,
                                    int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual,
                                    __local_mem__ uint32_t* helpAddr)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<T3> argmaxReg;
    AscendC::Reg::RegTensor<int32_t> dIndexReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;
    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1,
                                   pregT2);

    IndexConvNcdhw<T3, 1>(argmaxReg, dIndexReg, hIndexReg, wIndexReg, hwOutputConstReg, wOutputConstReg, curDIndex,
                          curHIndex, curWIndex, hOutputActual, wOutputAligned, highOutputOffset, highOutputPlaneActual,
                          highArgmaxPlaneActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask3D(pregArgmax, dIndexReg, hIndexReg, wIndexReg, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T>
__aicore__ inline void GenInitial1DIndices(Reg::RegTensor<T>& indexReg, int64_t colGenRate)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, T(colGenRate), preg);
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
__aicore__ inline void DhwGen2DIndexOne(Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, T(rowGenRate * colNumAligned), preg);
}

template <typename T>
__aicore__ inline void Gen2DIndexOne(Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, T(rowGenRate * colNumAligned), preg);
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
} // namespace MaxPool3DSmallKernelNameSpace
#endif // MAX_POOL3D_GRAD_SMALL_KERNEL_SCATTER_H
