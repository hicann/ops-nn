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
 * \file pool_grad_nchw_scatter_index.h
 * \brief 池化反向（MaxPoolGrad 系列）NCHW 格式 kernel 共用的 scatter 索引生成、
 *        argmax 索引换算（含 FastDiv 版本）与单列/多列梯度回散接口。
 */

#ifndef POOL_UTILS_ARCH35_INDEX_POOL_GRAD_NCHW_SCATTER_INDEX_H_
#define POOL_UTILS_ARCH35_INDEX_POOL_GRAD_NCHW_SCATTER_INDEX_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "pool_utils/arch35/compute/pool_fast_div.h"
#include "pool_utils/arch35/compute/pool_grad_scatter_compute.h"

namespace PoolUtils {
namespace Index {

template <typename T, const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNchw(AscendC::Reg::RegTensor<T>& argmaxReg, AscendC::Reg::RegTensor<int32_t>& hIndexReg,
                                     AscendC::Reg::RegTensor<int32_t>& wIndexReg,
                                     AscendC::Reg::RegTensor<T>& wOutputConstReg, int64_t curHIndex, int64_t curWIndex,
                                     int32_t wOutputAligned, int32_t highOutputOffset, int32_t highOutputPlaneActual,
                                     int32_t highArgmaxPlaneActual)
{
    AscendC::Reg::RegTensor<T> hTmpIndexReg;
    AscendC::Reg::RegTensor<T> wTmpIndexReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Div(hTmpIndexReg, argmaxReg, wOutputConstReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::Reg::Adds(tmpReg, hTmpIndexReg, T(-curHIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, PoolUtils::Compute::castTraitI64I32>(hIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)hIndexReg, (AscendC::Reg::RegTensor<int64_t>&)hIndexReg);
    } else {
        AscendC::Reg::Adds(hIndexReg, hTmpIndexReg, T(-curHIndex), allMask);
    }

    AscendC::Reg::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMask);
    AscendC::Reg::Sub(wTmpIndexReg, argmaxReg, wTmpIndexReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::Reg::Adds(tmpReg, wTmpIndexReg, T(-curWIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, PoolUtils::Compute::castTraitI64I32>(wIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)wIndexReg, (AscendC::Reg::RegTensor<int64_t>&)wIndexReg);
    } else {
        AscendC::Reg::Adds(wIndexReg, wTmpIndexReg, T(-curWIndex), allMask);
    }

    AscendC::Reg::Muls((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, hIndexReg, T(wOutputAligned), allMaskU32);
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

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNchw(__local_mem__ PoolUtils::Compute::computeType* yAddr, __local_mem__ T1* gradAddr,
                                      __local_mem__ T2* argmaxAddr, AscendC::Reg::RegTensor<uint32_t>& parallelRegIndex,
                                      AscendC::Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                      AscendC::Reg::RegTensor<T3>& wOutputConstReg, int64_t curHIndex,
                                      int64_t curWIndex, int32_t wOutputAligned, int32_t highOutputOffset,
                                      AscendC::Reg::RegTensor<int32_t>& zeroConstReg,
                                      AscendC::Reg::RegTensor<int32_t>& wMaxReg,
                                      AscendC::Reg::RegTensor<int32_t>& hMaxReg)
{
    AscendC::Reg::RegTensor<PoolUtils::Compute::computeType> gradReg;
    AscendC::Reg::RegTensor<T3> argmaxReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = PoolUtils::Compute::GenT2Mask<T2, T3>(maskT2);
    PoolUtils::Compute::GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex,
                                                       parallelRegGrad, pregT1, pregT2);
    IndexConvNchw<T3>(argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                      highOutputOffset, 0, 0);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        PoolUtils::Compute::FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    PoolUtils::Compute::GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulNCNchw(__local_mem__ PoolUtils::Compute::computeType* yAddr, __local_mem__ T1* gradAddr,
                                   __local_mem__ T2* argmaxAddr, AscendC::Reg::RegTensor<uint32_t>& parallelRegIndex,
                                   AscendC::Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                   AscendC::Reg::RegTensor<T3>& wOutputConstReg, int64_t curHIndex, int64_t curWIndex,
                                   int32_t wOutputAligned, int32_t highOutputOffset,
                                   AscendC::Reg::RegTensor<int32_t>& zeroConstReg,
                                   AscendC::Reg::RegTensor<int32_t>& wMaxReg, AscendC::Reg::RegTensor<int32_t>& hMaxReg,
                                   int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual)
{
    AscendC::Reg::RegTensor<PoolUtils::Compute::computeType> gradReg;
    AscendC::Reg::RegTensor<T3> argmaxReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = PoolUtils::Compute::GenT2Mask<T2, T3>(maskT2);
    PoolUtils::Compute::GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex,
                                                       parallelRegGrad, pregT1, pregT2);
    IndexConvNchw<T3, 1>(argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                         highOutputOffset, highOutputPlaneActual, highArgmaxPlaneActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        PoolUtils::Compute::FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    PoolUtils::Compute::GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T>
__aicore__ inline void GenInitial3DIndicesNchw(AscendC::Reg::RegTensor<T>& indexReg, int64_t colGenRate,
                                               int64_t rowGenRate, int64_t colNumAligned, int64_t fullBatchColNum,
                                               int64_t fullBatchRowNum, int64_t rowNumCount)
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

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rowNumCount * colNumAligned), preg);

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
__aicore__ inline void Gen3DIndexOneNchw(AscendC::Reg::RegTensor<T>& indexReg, int64_t rowGenRate,
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

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rowNumCount * colNumAligned), preg);

    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

template <const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNchwFastDiv(AscendC::Reg::RegTensor<int32_t>& argmaxReg,
                                            AscendC::Reg::RegTensor<int32_t>& hIndexReg,
                                            AscendC::Reg::RegTensor<int32_t>& wIndexReg,
                                            AscendC::Reg::RegTensor<uint32_t>& magicReg, int16_t shift,
                                            int64_t curHIndex, int64_t curWIndex, int32_t wOutput,
                                            int32_t wOutputAligned, int32_t highOutputOffset,
                                            int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual)
{
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<uint32_t> hTmpU32;
    AscendC::Reg::RegTensor<uint32_t> wTmpU32;
    AscendC::Reg::RegTensor<int32_t> highIncReg;
    AscendC::Reg::RegTensor<uint32_t> magicHighReg;
    AscendC::Reg::RegTensor<uint32_t> highIncU32;

    PoolUtils::Compute::FastDivImpl(hTmpU32, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, magicReg, shift, allMask);

    AscendC::Reg::Adds(hIndexReg, (AscendC::Reg::RegTensor<int32_t>&)hTmpU32, int32_t(-curHIndex), allMask);

    AscendC::Reg::Muls(wTmpU32, hTmpU32, uint32_t(wOutput), allMask);
    AscendC::Reg::Sub(wTmpU32, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, wTmpU32, allMask);

    AscendC::Reg::Adds(wIndexReg, (AscendC::Reg::RegTensor<int32_t>&)wTmpU32, int32_t(-curWIndex), allMask);

    AscendC::Reg::Muls(argmaxReg, hIndexReg, int32_t(wOutputAligned), allMask);
    AscendC::Reg::Add(argmaxReg, argmaxReg, wIndexReg, allMask);
    AscendC::Reg::Adds(argmaxReg, argmaxReg, highOutputOffset, allMask);

    if constexpr (IS_MUL_NC == 1) {
        AscendC::Reg::RegTensor<int32_t> highIncReg;
        AscendC::Reg::Arange(highIncReg, 0);
        AscendC::Reg::RegTensor<uint32_t> magicHighReg;
        uint32_t magicHigh = 0;
        uint32_t shiftHigh = 0;
        AscendC::GetUintDivMagicAndShift<uint32_t>(magicHigh, shiftHigh, static_cast<uint32_t>(highArgmaxPlaneActual));
        AscendC::Reg::Duplicate(magicHighReg, magicHigh);
        AscendC::Reg::RegTensor<uint32_t> highIncU32;
        PoolUtils::Compute::FastDivImpl(highIncU32, (AscendC::Reg::RegTensor<uint32_t>&)highIncReg, magicHighReg,
                                        static_cast<int16_t>(shiftHigh), allMask);
        AscendC::Reg::Muls(highIncReg, (AscendC::Reg::RegTensor<int32_t>&)highIncU32, highOutputPlaneActual, allMask);
        AscendC::Reg::Add(argmaxReg, argmaxReg, highIncReg, allMask);
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE, const bool IS_OVERLAP>
__aicore__ inline void DoSingleNCNchwFastDiv(
    __local_mem__ PoolUtils::Compute::computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ int32_t* argmaxAddr,
    AscendC::Reg::RegTensor<uint32_t>& parallelRegIndex, AscendC::Reg::RegTensor<uint32_t>& parallelRegGrad,
    uint32_t argmaxMaskCount, AscendC::Reg::RegTensor<uint32_t>& magicReg, int16_t shift, int64_t curHIndex,
    int64_t curWIndex, int32_t wOutput, int32_t wOutputAligned, int32_t highOutputOffset,
    AscendC::Reg::RegTensor<int32_t>& zeroConstReg, AscendC::Reg::RegTensor<int32_t>& wMaxReg,
    AscendC::Reg::RegTensor<int32_t>& hMaxReg)
{
    AscendC::Reg::RegTensor<PoolUtils::Compute::computeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> argmaxReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = PoolUtils::Compute::GenT2Mask<int32_t, int32_t>(maskT2);
    PoolUtils::Compute::GetConCurrentInput<T1, int32_t, int32_t>(argmaxReg, gradReg, gradAddr, argmaxAddr,
                                                                 parallelRegIndex, parallelRegGrad, pregT1, pregT2);
    IndexConvNchwFastDiv<0>(argmaxReg, hIndexReg, wIndexReg, magicReg, shift, curHIndex, curWIndex, wOutput,
                            wOutputAligned, highOutputOffset, 0, 0);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        PoolUtils::Compute::FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }
    if constexpr (IS_OVERLAP) {
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }

    PoolUtils::Compute::GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, const uint32_t IS_CHECK_RANGE, const bool IS_OVERLAP>
__aicore__ inline void DoMulNCNchwFastDiv(
    __local_mem__ PoolUtils::Compute::computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ int32_t* argmaxAddr,
    AscendC::Reg::RegTensor<uint32_t>& parallelRegIndex, AscendC::Reg::RegTensor<uint32_t>& parallelRegGrad,
    uint32_t argmaxMaskCount, AscendC::Reg::RegTensor<uint32_t>& magicReg, int16_t shift, int64_t curHIndex,
    int64_t curWIndex, int32_t wOutput, int32_t wOutputAligned, int32_t highOutputOffset,
    AscendC::Reg::RegTensor<int32_t>& zeroConstReg, AscendC::Reg::RegTensor<int32_t>& wMaxReg,
    AscendC::Reg::RegTensor<int32_t>& hMaxReg, int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual)
{
    AscendC::Reg::RegTensor<PoolUtils::Compute::computeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> argmaxReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = PoolUtils::Compute::GenT2Mask<int32_t, int32_t>(maskT2);
    PoolUtils::Compute::GetConCurrentInput<T1, int32_t, int32_t>(argmaxReg, gradReg, gradAddr, argmaxAddr,
                                                                 parallelRegIndex, parallelRegGrad, pregT1, pregT2);
    IndexConvNchwFastDiv<1>(argmaxReg, hIndexReg, wIndexReg, magicReg, shift, curHIndex, curWIndex, wOutput,
                            wOutputAligned, highOutputOffset, highOutputPlaneActual, highArgmaxPlaneActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        PoolUtils::Compute::FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }
    if constexpr (IS_OVERLAP) {
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }

    PoolUtils::Compute::GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T>
__simd_callee__ inline void GenInitial1DIndicesVF(AscendC::Reg::RegTensor<T>& indexReg, int64_t colGenRate)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, T(colGenRate), preg);
}

__simd_callee__ inline void FastDivImplVF(AscendC::Reg::RegTensor<uint32_t>& res,
                                          AscendC::Reg::RegTensor<uint32_t>& src,
                                          AscendC::Reg::RegTensor<uint32_t>& magic, int16_t shift,
                                          AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<uint32_t> tmp;
    AscendC::Reg::Mull(tmp, res, src, magic, mask);
    AscendC::Reg::Add(tmp, src, res, mask);
    AscendC::Reg::ShiftRights(res, tmp, shift, mask);
}

template <const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexInt32FastDiv(AscendC::Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,
                                                int32_t left, int32_t wInputActualNoPad, int32_t hIndexBase,
                                                AscendC::Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset,
                                                uint32_t magic, uint32_t shift)
{
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::RegTensor<uint32_t> divResultU32;
    AscendC::Reg::RegTensor<uint32_t> magicReg;
    AscendC::Reg::MaskReg negInfMask;
    AscendC::Reg::MaskReg allMaskB32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(zeroReg, static_cast<int32_t>(0));
    AscendC::Reg::Duplicate(magicReg, magic);
    AscendC::Reg::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);

    PoolUtils::Compute::FastDivImpl(divResultU32, (AscendC::Reg::RegTensor<uint32_t>&)srcReg, magicReg,
                                    static_cast<int16_t>(shift), allMaskB32);

    AscendC::Reg::Adds(hIndexReg, (AscendC::Reg::RegTensor<int32_t>&)divResultU32, hIndexBase, allMaskB32);

    if constexpr (IS_PAD) {
        AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskB32);
        AscendC::Reg::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }

    AscendC::Reg::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskB32);

    AscendC::Reg::Muls(divResultU32, divResultU32, wStrideOffset, allMaskB32);
    AscendC::Reg::Sub((AscendC::Reg::RegTensor<uint32_t>&)srcReg, (AscendC::Reg::RegTensor<uint32_t>&)srcReg,
                      divResultU32, allMaskB32);
    AscendC::Reg::Adds(wIndexReg, srcReg, left, allMaskB32);

    if constexpr (IS_PAD) {
        AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskB32);
        AscendC::Reg::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }

    AscendC::Reg::Add(dstReg, hIndexReg, wIndexReg, allMaskB32);
}

template <const uint32_t IS_PAD>
__simd_callee__ inline void ConvertIndexInt32FastDivVF(AscendC::Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,
                                                       int32_t left, int32_t wInputActualNoPad, int32_t hIndexBase,
                                                       AscendC::Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset,
                                                       uint32_t magic, uint32_t shift)
{
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::RegTensor<uint32_t> divResultU32;
    AscendC::Reg::RegTensor<uint32_t> magicReg;
    AscendC::Reg::MaskReg negInfMask;
    AscendC::Reg::MaskReg allMaskB32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(zeroReg, static_cast<int32_t>(0));
    AscendC::Reg::Duplicate(magicReg, magic);
    AscendC::Reg::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);

    FastDivImplVF(divResultU32, (AscendC::Reg::RegTensor<uint32_t>&)srcReg, magicReg, static_cast<int16_t>(shift),
                  allMaskB32);

    AscendC::Reg::Adds(hIndexReg, (AscendC::Reg::RegTensor<int32_t>&)divResultU32, hIndexBase, allMaskB32);

    if constexpr (IS_PAD) {
        AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskB32);
        AscendC::Reg::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }

    AscendC::Reg::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskB32);

    AscendC::Reg::Muls(divResultU32, divResultU32, wStrideOffset, allMaskB32);
    AscendC::Reg::Sub((AscendC::Reg::RegTensor<uint32_t>&)srcReg, (AscendC::Reg::RegTensor<uint32_t>&)srcReg,
                      divResultU32, allMaskB32);
    AscendC::Reg::Adds(wIndexReg, srcReg, left, allMaskB32);

    if constexpr (IS_PAD) {
        AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskB32);
        AscendC::Reg::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }

    AscendC::Reg::Add(dstReg, hIndexReg, wIndexReg, allMaskB32);
}

template <const uint32_t IS_PAD>
__simd_callee__ inline void ConvertIndexNcInt32FastDivVF(
    AscendC::Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, int32_t left, int32_t wInputActualNoPad,
    int32_t hIndexBase, AscendC::Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount,
    int32_t inputNcSize, uint32_t magicNc, uint32_t shiftNc, uint32_t magicWStride, uint32_t shiftWStride)
{
    AscendC::Reg::RegTensor<int32_t> ncIndexReg;
    AscendC::Reg::RegTensor<uint32_t> divResultU32;
    AscendC::Reg::RegTensor<uint32_t> magicNcReg;
    AscendC::Reg::MaskReg allMaskB32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(magicNcReg, magicNc);
    AscendC::Reg::Arange(ncIndexReg, static_cast<int32_t>(0));
    FastDivImplVF(divResultU32, (AscendC::Reg::RegTensor<uint32_t>&)ncIndexReg, magicNcReg,
                  static_cast<int16_t>(shiftNc), allMaskB32);
    AscendC::Reg::Muls(ncIndexReg, (AscendC::Reg::RegTensor<int32_t>&)divResultU32, inputNcSize, allMaskB32);
    AscendC::Reg::Sub(srcReg, srcReg, ncIndexReg, allMaskB32);

    ConvertIndexInt32FastDivVF<IS_PAD>(srcReg, wStrideOffset, left, wInputActualNoPad, hIndexBase, dstReg,
                                       ncInputOffset, magicWStride, shiftWStride);
}

} // namespace Index
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_INDEX_POOL_GRAD_NCHW_SCATTER_INDEX_H_
