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
 * \file pool_grad_nhwc_scatter_index.h
 * \brief 池化反向（MaxPoolGrad 系列）NHWC 格式 kernel 共用的 scatter 索引生成、
 *        argmax 索引换算与单通道/多通道梯度回散接口。
 */

#ifndef POOL_UTILS_ARCH35_INDEX_POOL_GRAD_NHWC_SCATTER_INDEX_H_
#define POOL_UTILS_ARCH35_INDEX_POOL_GRAD_NHWC_SCATTER_INDEX_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "pool_utils/arch35/compute/pool_grad_scatter_compute.h"

namespace PoolUtils {
namespace Index {

constexpr int32_t VER_NORMAL = 0;

template <typename T, const uint32_t IS_MUL_C = 0>
__aicore__ inline void IndexConvNhwc(AscendC::Reg::RegTensor<T>& argmaxReg, AscendC::Reg::RegTensor<int32_t>& hIndexReg,
                                     AscendC::Reg::RegTensor<int32_t>& wIndexReg,
                                     AscendC::Reg::RegTensor<T>& wOutputConstReg, int64_t curHIndex, int64_t curWIndex,
                                     int32_t wOutputActual, int32_t cOutputAligned, int32_t cOffset, int32_t nOffset,
                                     int32_t cOutputActual)
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

    AscendC::Reg::Muls((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, hIndexReg, wOutputActual, allMaskU32);
    AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                      wIndexReg, allMaskU32);

    AscendC::Reg::Muls((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                       cOutputAligned, allMaskU32);
    AscendC::Reg::RegTensor<int32_t> cIncReg;
    AscendC::Reg::Arange(cIncReg, (cOffset));
    if constexpr (IS_MUL_C == 1) {
        AscendC::Reg::RegTensor<int32_t> constReg;
        AscendC::Reg::Duplicate(constReg, cOutputActual);
        AscendC::Reg::RegTensor<int32_t> tmpReg;
        AscendC::Reg::Div(tmpReg, cIncReg, constReg, allMaskU32);
        AscendC::Reg::Mul(tmpReg, tmpReg, constReg, allMaskU32);
        AscendC::Reg::Sub(cIncReg, cIncReg, tmpReg, allMaskU32);
    }

    AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                      cIncReg, allMaskU32);
    AscendC::Reg::Adds((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                       nOffset, allMaskU32);
}

template <typename T>
__aicore__ inline void GenInitial3DIndicesNhwc(AscendC::Reg::RegTensor<T>& indexReg, int64_t colGenRate,
                                               int64_t rowGenRate, int64_t colNum, int64_t fullBatchColNum,
                                               int64_t cOutputActual, int64_t cOutputAligned)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum * cOutputActual));

    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);

    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum * cOutputActual), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNum * cOutputAligned), preg);

    AscendC::Reg::Duplicate(constReg, T(cOutputActual));
    AscendC::Reg::Div(segmentScalarReg2, segmentIncReg, constReg, preg);

    AscendC::Reg::Muls(segmentIncReg2, segmentScalarReg2, T(cOutputActual), preg);
    AscendC::Reg::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);

    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(colGenRate * cOutputAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOneNhwc(AscendC::Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNum,
                                         int64_t cOutputActual, int64_t cOutputAligned)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(cOutputActual));

    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);

    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(cOutputActual), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNum * cOutputAligned), preg);
    AscendC::Reg::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER>
__aicore__ inline void DoSingleCNhwc(__local_mem__ PoolUtils::Compute::computeType* yAddr, __local_mem__ T1* gradAddr,
                                     __local_mem__ T2* argmaxAddr, uint32_t argmaxOffset, uint32_t argmaxMaskCount,
                                     int64_t curHIndex, int64_t curWIndex, int32_t wOutputActual,
                                     int32_t cOutputAligned, int32_t cOffset, int32_t nOffset, int32_t cOutputActual,
                                     int32_t cOutput, AscendC::Reg::RegTensor<int32_t>& zeroConstReg,
                                     AscendC::Reg::RegTensor<int32_t>& wMaxReg,
                                     AscendC::Reg::RegTensor<int32_t>& hMaxReg,
                                     AscendC::Reg::RegTensor<T3>& wOutputConstReg)
{
    AscendC::Reg::RegTensor<PoolUtils::Compute::computeType> gradReg;
    AscendC::Reg::RegTensor<T3> argmaxReg;

    PoolUtils::Compute::GetContinuousInput(argmaxReg, gradReg, gradAddr, argmaxAddr, argmaxOffset);
    if constexpr (VER == VER_NORMAL) {
        PoolUtils::Compute::TransHWC2HW<T3>(argmaxReg, cOutput);
    }
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;
    IndexConvNhwc<T3, 0>(argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputActual,
                         cOutputAligned, cOffset, nOffset, cOutputActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        PoolUtils::Compute::FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    PoolUtils::Compute::GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER>
__aicore__ inline void DoMulCNhwc(__local_mem__ PoolUtils::Compute::computeType* yAddr, __local_mem__ T1* gradAddr,
                                  __local_mem__ T2* argmaxAddr, AscendC::Reg::RegTensor<uint32_t>& parallelRegIndex,
                                  uint32_t argmaxMaskCount, int64_t curHIndex, int64_t curWIndex, int32_t wOutputActual,
                                  int32_t hOutputActual, int32_t cOutputAligned, int32_t cOffset, int32_t nOffset,
                                  int32_t cOutputActual, AscendC::Reg::RegTensor<T3>& wOutputConstReg)
{
    AscendC::Reg::RegTensor<PoolUtils::Compute::computeType> gradReg;
    AscendC::Reg::RegTensor<T3> argmaxReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = PoolUtils::Compute::GenT2Mask<T2, T3>(maskT2);

    PoolUtils::Compute::GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex,
                                                       pregT1, pregT2);
    if constexpr (VER == VER_NORMAL) {
        PoolUtils::Compute::TransHWC2HW<T3>(argmaxReg, cOutputActual);
    }
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;
    IndexConvNhwc<T3, 1>(argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputActual,
                         cOutputAligned, cOffset, nOffset, cOutputActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        AscendC::Reg::Duplicate(zeroConstReg, T2(0));
        AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
        AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        PoolUtils::Compute::FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    PoolUtils::Compute::GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

} // namespace Index
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_INDEX_POOL_GRAD_NHWC_SCATTER_INDEX_H_
