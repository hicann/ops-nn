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
constexpr uint32_t INDEX_SIX= 6;
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

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310 || __NPU_ARCH == 5102
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
template <typename T2, typename T3>
__aicore__ inline MicroAPI::MaskReg GenT2Mask(uint32_t& maskCount)
{
    MicroAPI::MaskReg reg;
    if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int64_t>::value) {
        reg = AscendC::MicroAPI::UpdateMask<T2, AscendC::MicroAPI::RegTraitNumTwo>(maskCount);
    } else {
        reg = AscendC::MicroAPI::UpdateMask<T2>(maskCount);
    }
    return reg;
}

template <typename T>
__aicore__ inline void GradientAcc(__local_mem__ computeType* yAddr, MicroAPI::RegTensor<computeType>& gradReg,
                                   MicroAPI::RegTensor<T>& argmaxReg, MicroAPI::MaskReg& pregArgmax)
{
    AscendC::MicroAPI::RegTensor<computeType> scatterAccResReg;
    AscendC::MicroAPI::DataCopyGather(scatterAccResReg, yAddr, (AscendC::MicroAPI::RegTensor<uint32_t>&)argmaxReg,
                                      pregArgmax);
    AscendC::MicroAPI::Add(scatterAccResReg, scatterAccResReg, gradReg, pregArgmax);
    AscendC::MicroAPI::DataCopyScatter(yAddr, scatterAccResReg, (AscendC::MicroAPI::RegTensor<uint32_t>&)argmaxReg,
                                       pregArgmax);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void GetConCurrentInput(MicroAPI::RegTensor<T3>& argmaxReg, MicroAPI::RegTensor<computeType>& gradReg,
                                          __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
                                          MicroAPI::RegTensor<uint32_t>& parallelRegIndex, MicroAPI::RegTensor<uint32_t>& parallelRegGrad, 
                                          MicroAPI::MaskReg& pregT1, MicroAPI::MaskReg& pregT2)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::MicroAPI::RegTensor<T1> gradRegT1;
        AscendC::MicroAPI::RegTensor<uint16_t> parallelRegGradU16;
        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Cast<uint16_t, uint32_t, castTraitU32U16>(parallelRegGradU16, parallelRegGrad, allMaskU32);
        AscendC::MicroAPI::Pack(parallelRegGradU16, (AscendC::MicroAPI::RegTensor<int32_t>&)parallelRegGradU16);
        AscendC::MicroAPI::DataCopyGather(gradRegT1, gradAddr, parallelRegGradU16, pregT1);
        AscendC::MicroAPI::UnPack((AscendC::MicroAPI::RegTensor<uint32_t>&)gradRegT1,
                                  (AscendC::MicroAPI::RegTensor<uint16_t>&)gradRegT1);
        AscendC::MicroAPI::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::MicroAPI::DataCopyGather(gradReg, gradAddr, parallelRegGrad, pregT1);
    }

    if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int32_t>::value) {
        AscendC::MicroAPI::DataCopyGather(argmaxReg, argmaxAddr, parallelRegIndex, pregT2);
    } else if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int64_t>::value) {
        AscendC::MicroAPI::RegTensor<T2, AscendC::MicroAPI::RegTraitNumTwo> argmaxRegTwo;
        AscendC::MicroAPI::DataCopyGather(argmaxRegTwo, argmaxAddr, parallelRegIndex, pregT2);
        argmaxReg = (AscendC::MicroAPI::RegTensor<T3>&)argmaxRegTwo.reg[0];
    } else if constexpr (std::is_same<T3, int64_t>::value && std::is_same<T2, int64_t>::value) {
        AscendC::MicroAPI::DataCopyGather(argmaxReg, argmaxAddr, parallelRegIndex, pregT2);
    }
}

namespace MaxPool3DSmallKernelNameSpace {
__aicore__ inline int64_t PStart(int64_t index, int64_t pad, int64_t kernel, int64_t dilation, int64_t stride)
{   
    if(stride == 0) {
        return 0;
    }
    return (index + pad < (kernel - 1) * dilation + 1) ? 0 : (index + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
};
__aicore__ inline int64_t PEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize)
{
    if(stride == 0) {
        return 0;
    }
    return (index + pad) / stride + 1 < pooledSize ? (index + pad) / stride + 1 : pooledSize;
};

template <typename T, const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNcdhw(MicroAPI::RegTensor<T>& argmaxReg,
                                        MicroAPI::RegTensor<int32_t>& dIndexReg,
                                        MicroAPI::RegTensor<int32_t>& hIndexReg,
                                        MicroAPI::RegTensor<int32_t>& wIndexReg,
                                        MicroAPI::RegTensor<T>& hwOutputConstReg, MicroAPI::RegTensor<T>& wOutputConstReg,
                                     int64_t curDIndex, int64_t curHIndex, int64_t curWIndex,
                                     int32_t hOutputActual, int32_t wOutputAligned,
                                     int32_t highOutputOffset, int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual)
{
    AscendC::MicroAPI::RegTensor<T> dTmpIndexReg;
    AscendC::MicroAPI::RegTensor<T> hTmpIndexReg;
    AscendC::MicroAPI::RegTensor<T> wTmpIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> dhwTmpIndexReg;
    AscendC::MicroAPI::RegTensor<T> remReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg allMaskU32 =
        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Div(dTmpIndexReg, argmaxReg, hwOutputConstReg, allMask);
    AscendC::MicroAPI::Mul(remReg, dTmpIndexReg, hwOutputConstReg, allMask);
    AscendC::MicroAPI::Sub(remReg, argmaxReg, remReg, allMask);
    AscendC::MicroAPI::Div(hTmpIndexReg, remReg, wOutputConstReg, allMask);
    AscendC::MicroAPI::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMask);
    AscendC::MicroAPI::Sub(wTmpIndexReg, remReg, wTmpIndexReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::MicroAPI::Adds(tmpReg, dTmpIndexReg, T(-curDIndex), allMask);
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(dIndexReg, tmpReg, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)dIndexReg,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)dIndexReg);

        AscendC::MicroAPI::Adds(tmpReg, hTmpIndexReg, T(-curHIndex), allMask);
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(hIndexReg, tmpReg, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)hIndexReg,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)hIndexReg);

        AscendC::MicroAPI::Adds(tmpReg, wTmpIndexReg, T(-curWIndex), allMask);
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(wIndexReg, tmpReg, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)wIndexReg,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)wIndexReg);
    } else {
        AscendC::MicroAPI::Adds(dIndexReg, dTmpIndexReg, T(-curDIndex), allMask);
        AscendC::MicroAPI::Adds(hIndexReg, hTmpIndexReg, T(-curHIndex), allMask);
        AscendC::MicroAPI::Adds(wIndexReg, wTmpIndexReg, T(-curWIndex), allMask);
    }

    int32_t hwOutputAligned = hOutputActual * wOutputAligned;
    AscendC::MicroAPI::Muls((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, hIndexReg, T(wOutputAligned),
                            allMaskU32);

    AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                           (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, wIndexReg, allMaskU32);

    AscendC::MicroAPI::Adds((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                            (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, highOutputOffset, allMaskU32);

    AscendC::MicroAPI::Muls(dhwTmpIndexReg, dIndexReg, T(hwOutputAligned), allMaskU32);

    AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                           (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, dhwTmpIndexReg, allMaskU32);

    if constexpr (IS_MUL_NC == 1) {
        AscendC::MicroAPI::RegTensor<int32_t> highIncReg;
        AscendC::MicroAPI::Arange(highIncReg, 0);
        AscendC::MicroAPI::RegTensor<int32_t> constReg;
        AscendC::MicroAPI::Duplicate(constReg, highArgmaxPlaneActual);
        AscendC::MicroAPI::Div(highIncReg, highIncReg, constReg, allMaskU32);
        AscendC::MicroAPI::Muls(highIncReg, highIncReg, highOutputPlaneActual,
                            allMaskU32);
        AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                           (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, highIncReg,
                           allMaskU32);
    }
}

template <typename T, const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNchw(MicroAPI::RegTensor<T>& argmaxReg,
                                     MicroAPI::RegTensor<int32_t>& dIndexReg,
                                     MicroAPI::RegTensor<int32_t>& hIndexReg,
                                     MicroAPI::RegTensor<int32_t>& wIndexReg,
                                     MicroAPI::RegTensor<T>& hwOutputConstReg, MicroAPI::RegTensor<T>& wOutputConstReg,
                                     int64_t curDIndex, int64_t curHIndex, int64_t curWIndex,
                                     int32_t hOutputActual, int32_t wOutputAligned,
                                     int32_t highOutputOffset,
                                     int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual)
{
    AscendC::MicroAPI::RegTensor<T> dTmpIndexReg;
    AscendC::MicroAPI::RegTensor<T> hTmpIndexReg;
    AscendC::MicroAPI::RegTensor<T> wTmpIndexReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg allMaskU32 =
        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Div(dTmpIndexReg, argmaxReg, hwOutputConstReg, allMask);

    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::MicroAPI::Adds(tmpReg, dTmpIndexReg, T(-curDIndex), allMask);
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(dIndexReg, tmpReg, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)dIndexReg,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)dIndexReg);
    } else {
        AscendC::MicroAPI::Adds(dIndexReg, dTmpIndexReg, T(-curDIndex), allMask);
    }

    AscendC::MicroAPI::Mul(hTmpIndexReg, dTmpIndexReg, hwOutputConstReg, allMask);

    AscendC::MicroAPI::Sub(dTmpIndexReg, argmaxReg, hTmpIndexReg, allMask);

    AscendC::MicroAPI::Div(hTmpIndexReg, dTmpIndexReg, wOutputConstReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::MicroAPI::Adds(tmpReg, hTmpIndexReg, T(-curHIndex), allMask);
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(hIndexReg, tmpReg, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)hIndexReg,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)hIndexReg);
    } else {
        AscendC::MicroAPI::Adds(hIndexReg, hTmpIndexReg, T(-curHIndex), allMask);
    }

    AscendC::MicroAPI::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMask);
    AscendC::MicroAPI::Sub(wTmpIndexReg, dTmpIndexReg, wTmpIndexReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::MicroAPI::Adds(tmpReg, wTmpIndexReg, T(-curWIndex), allMask);
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(wIndexReg, tmpReg, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)wIndexReg,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)wIndexReg);
    } else {
        AscendC::MicroAPI::Adds(wIndexReg, wTmpIndexReg, T(-curWIndex), allMask);
    }

    AscendC::MicroAPI::Muls((AscendC::MicroAPI::RegTensor<int32_t>&)dTmpIndexReg, dIndexReg, T(wOutputAligned * hOutputActual),
                            allMaskU32);

    AscendC::MicroAPI::Muls((AscendC::MicroAPI::RegTensor<int32_t>&)hTmpIndexReg, hIndexReg, T(wOutputAligned),
                            allMaskU32);
    AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                           (AscendC::MicroAPI::RegTensor<int32_t>&)dTmpIndexReg, hTmpIndexReg, allMaskU32);
    AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                           (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, wIndexReg, allMaskU32);

    AscendC::MicroAPI::Adds((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                            (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, highOutputOffset, allMaskU32);

    if constexpr (IS_MUL_NC == 1) {
        AscendC::MicroAPI::RegTensor<int32_t> highIncReg;
        AscendC::MicroAPI::Arange(highIncReg, 0);
        AscendC::MicroAPI::RegTensor<int32_t> constReg;
        AscendC::MicroAPI::Duplicate(constReg, highArgmaxPlaneActual);
        AscendC::MicroAPI::Div(highIncReg, highIncReg, constReg, allMaskU32);
        AscendC::MicroAPI::Muls(highIncReg, highIncReg, highOutputPlaneActual,
                            allMaskU32);
        AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                           (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, highIncReg,
                           allMaskU32);
    }
}
__aicore__ inline void FilterMask3D(MicroAPI::MaskReg& preg, MicroAPI::RegTensor<int32_t>& dIndexReg, MicroAPI::RegTensor<int32_t>& hIndexReg,
                                  MicroAPI::RegTensor<int32_t>& wIndexReg, MicroAPI::RegTensor<int32_t>& zeroConstReg,
                                  MicroAPI::RegTensor<int32_t>& dMaxReg, MicroAPI::RegTensor<int32_t>& hMaxReg, MicroAPI::RegTensor<int32_t>& wMaxReg)
{
    AscendC::MicroAPI::MaskReg gtMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, hIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, hMaxReg, hIndexReg, gtMask);

    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);

    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, dIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, dMaxReg, dIndexReg, gtMask);
    AscendC::MicroAPI::MaskAnd(preg, preg, gtMask, allMask);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNchw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                              __local_mem__ T2* argmaxAddr, MicroAPI::RegTensor<uint32_t>& parallelRegIndex, MicroAPI::RegTensor<uint32_t>& parallelRegGrad,
                              uint32_t argmaxMaskCount, MicroAPI::RegTensor<T3>& hwOutputConstReg, MicroAPI::RegTensor<T3>& wOutputConstReg,
                              int64_t curDIndex, int64_t curHIndex, int64_t curWIndex,
                              int32_t hOutputActual, int32_t wOutputAligned, int32_t highOutputOffset,
                              MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& wMaxReg,
                              MicroAPI::RegTensor<int32_t>& hMaxReg, MicroAPI::RegTensor<int32_t>& dMaxReg)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<T3> argmaxReg;

    AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);

    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1, pregT2);
    IndexConvNcdhw<T3>(argmaxReg, dIndexReg, hIndexReg, wIndexReg,
                        hwOutputConstReg, wOutputConstReg,
                        curDIndex, curHIndex, curWIndex,
                        hOutputActual, wOutputAligned,
                        highOutputOffset, 0 ,0);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask3D(pregArgmax, dIndexReg, hIndexReg, wIndexReg, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
    }
    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}


template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNcdhw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                              __local_mem__ T2* argmaxAddr, MicroAPI::RegTensor<uint32_t>& parallelRegIndex,
                              MicroAPI::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount, MicroAPI::RegTensor<T3>& hwOutputConstReg,
                              MicroAPI::RegTensor<T3>& wOutputConstReg, int64_t curDIndex, int64_t curHIndex,
                              int64_t curWIndex, int32_t wOutputAligned, int32_t highOutputOffset, 
                              int32_t hOutputActual, MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& dMaxReg, 
                              MicroAPI::RegTensor<int32_t>& hMaxReg, MicroAPI::RegTensor<int32_t>& wMaxReg)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<T3> argmaxReg;
    AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1, pregT2);
    IndexConvNcdhw<T3>(argmaxReg, dIndexReg, hIndexReg, wIndexReg, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, hOutputActual, wOutputAligned,
                      highOutputOffset, 0, 0);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask3D(pregArgmax, dIndexReg, hIndexReg, wIndexReg, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulNCNcdhw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                              __local_mem__ T2* argmaxAddr, MicroAPI::RegTensor<uint32_t>& parallelRegIndex,
                              MicroAPI::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount, MicroAPI::RegTensor<T3>& hwOutputConstReg, 
                              MicroAPI::RegTensor<T3>& wOutputConstReg, int64_t curDIndex, int64_t curHIndex,
                              int64_t curWIndex, int32_t wOutputAligned, int32_t highOutputOffset,
                              int32_t hOutputActual, MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& dMaxReg, 
                              MicroAPI::RegTensor<int32_t>& hMaxReg, MicroAPI::RegTensor<int32_t>& wMaxReg,
                              int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual, __local_mem__ uint32_t* helpAddr)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<T3> argmaxReg; 
    AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1, pregT2);
    
    IndexConvNcdhw<T3, 1>(argmaxReg, dIndexReg, hIndexReg, wIndexReg, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, hOutputActual, wOutputAligned,
                      highOutputOffset, highOutputPlaneActual, highArgmaxPlaneActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask3D(pregArgmax, dIndexReg, hIndexReg, wIndexReg, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T>
__aicore__ inline void GenInitial1DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(colGenRate), preg);
}

template <typename T>
__aicore__ inline void GenInitial2DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum)
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
__aicore__ inline void DhwGenInitial2DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum)
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
__aicore__ inline void DhwGen2DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rowGenRate * colNumAligned), preg);
}

template <typename T>
__aicore__ inline void Gen2DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rowGenRate * colNumAligned), preg);
}

template <typename T>
__aicore__ inline void GenInitial3DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate, int64_t colGenRate,
                                           int64_t fullBatchRowNum, int64_t rowNumCount, int64_t fullBatchColNum, int64_t colNumAligned)
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
__aicore__ inline void Gen3DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t dGenRate, int64_t rowGenRate, int64_t colNumAligned,
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
__aicore__ inline void GenInitial3DHighIndices(MicroAPI::RegTensor<T>& indexReg, int64_t highStride, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum, int64_t fullBatchRowNum)
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
__aicore__ inline void Gen3DHighIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t highStride, int64_t rowGenRate, int64_t colNumAligned,
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
__aicore__ inline void GenInitial4DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum, int64_t fullBatchRowNum, int64_t fullBatchDepthNum,
                                           int64_t depthStride, int64_t highStride)
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

    //组装offset
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
__aicore__ inline void Gen4DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned,
                                     int64_t fullBatchRowNum, int64_t fullBatchDepthNum, int64_t depthStride, int64_t highStride)
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
} 
#endif  // MAX_POOL3D_GRAD_SMALL_KERNEL_SCATTER_H
