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
 * \file pool_grad_scatter_compute.h
 * \brief 池化反向（MaxPoolGrad/AvgPoolGrad 系列）kernel 共用的 gather/scatter 梯度累加基础原语，
 *        包含掩码过滤、梯度累加、并发取数与池化窗口边界计算接口。
 */

#ifndef POOL_UTILS_ARCH35_COMPUTE_POOL_GRAD_SCATTER_COMPUTE_H_
#define POOL_UTILS_ARCH35_COMPUTE_POOL_GRAD_SCATTER_COMPUTE_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

namespace PoolUtils {
namespace Compute {

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

__aicore__ inline int64_t PStart(int64_t index, int64_t pad, int64_t kernel, int64_t dilation, int64_t stride)
{
    return (index + pad < (kernel - 1) * dilation + 1) ? 0 : (index + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

__aicore__ inline int64_t PEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize)
{
    return (index + pad) / stride + 1 < pooledSize ? (index + pad) / stride + 1 : pooledSize;
}

template <typename T2, typename T3>
__aicore__ inline AscendC::Reg::MaskReg GenT2Mask(uint32_t& maskCount)
{
    AscendC::Reg::MaskReg reg;
    if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int64_t>::value) {
        reg = AscendC::Reg::UpdateMask<T2, AscendC::Reg::RegTraitNumTwo>(maskCount);
    } else {
        reg = AscendC::Reg::UpdateMask<T2>(maskCount);
    }
    return reg;
}

__aicore__ inline void FilterMask(AscendC::Reg::MaskReg& preg, AscendC::Reg::RegTensor<int32_t>& hIndexReg,
                                  AscendC::Reg::RegTensor<int32_t>& wIndexReg,
                                  AscendC::Reg::RegTensor<int32_t>& zeroConstReg,
                                  AscendC::Reg::RegTensor<int32_t>& wMaxReg, AscendC::Reg::RegTensor<int32_t>& hMaxReg)
{
    AscendC::Reg::MaskReg gtMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::GE>(gtMask, hIndexReg, zeroConstReg, gtMask);
    AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::GT>(gtMask, hMaxReg, hIndexReg, gtMask);

    AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);
    AscendC::Reg::MaskAnd(preg, preg, gtMask, allMask);
}

template <typename T>
__aicore__ inline void GradientAcc(__local_mem__ computeType* yAddr, AscendC::Reg::RegTensor<computeType>& gradReg,
                                   AscendC::Reg::RegTensor<T>& argmaxReg, AscendC::Reg::MaskReg& pregArgmax)
{
    AscendC::Reg::RegTensor<computeType> scatterAccResReg;
    AscendC::Reg::DataCopyGather(scatterAccResReg, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
    AscendC::Reg::Add(scatterAccResReg, scatterAccResReg, gradReg, pregArgmax);
    AscendC::Reg::DataCopyScatter(yAddr, scatterAccResReg, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void GetConCurrentInput(AscendC::Reg::RegTensor<T3>& argmaxReg,
                                          AscendC::Reg::RegTensor<computeType>& gradReg, __local_mem__ T1* gradAddr,
                                          __local_mem__ T2* argmaxAddr,
                                          AscendC::Reg::RegTensor<uint32_t>& parallelRegIndex,
                                          AscendC::Reg::RegTensor<uint32_t>& parallelRegGrad,
                                          AscendC::Reg::MaskReg& pregT1, AscendC::Reg::MaskReg& pregT2)
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

template <typename T1, typename T2, typename T3>
__aicore__ inline void GetConCurrentInput(AscendC::Reg::RegTensor<T3>& argmaxReg,
                                          AscendC::Reg::RegTensor<computeType>& gradReg, __local_mem__ T1* gradAddr,
                                          __local_mem__ T2* argmaxAddr,
                                          AscendC::Reg::RegTensor<uint32_t>& parallelRegIndex,
                                          AscendC::Reg::MaskReg& pregT1, AscendC::Reg::MaskReg& pregT2)
{
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex, pregT1,
                                   pregT2);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void GetContinuousInput(AscendC::Reg::RegTensor<T3>& argmaxReg,
                                          AscendC::Reg::RegTensor<computeType>& gradReg, __local_mem__ T1* gradAddr,
                                          __local_mem__ T2* argmaxAddr, uint32_t argmaxOffset)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::Reg::RegTensor<T1> gradRegT1;
        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::DataCopy(gradRegT1, gradAddr + argmaxOffset);
        AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint32_t>&)gradRegT1,
                             (AscendC::Reg::RegTensor<uint16_t>&)gradRegT1);
        AscendC::Reg::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::Reg::DataCopy(gradReg, gradAddr + argmaxOffset);
    }

    if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int32_t>::value) {
        AscendC::Reg::DataCopy(argmaxReg, argmaxAddr + argmaxOffset);
    } else if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int64_t>::value) {
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> argmaxRegTwo;
        AscendC::Reg::DataCopy(argmaxRegTwo, argmaxAddr + argmaxOffset);
        argmaxReg = (AscendC::Reg::RegTensor<T3>&)argmaxRegTwo.reg[0];
    } else if constexpr (std::is_same<T3, int64_t>::value && std::is_same<T2, int64_t>::value) {
        AscendC::Reg::DataCopy(argmaxReg, argmaxAddr + argmaxOffset);
    }
}

template <typename T>
__aicore__ inline void TransHWC2HW(AscendC::Reg::RegTensor<T>& argmaxReg, int32_t cOutputActual)
{
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::Duplicate(constReg, cOutputActual);
    AscendC::Reg::Div(argmaxReg, argmaxReg, constReg, allMask);
}

} // namespace Compute
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_COMPUTE_POOL_GRAD_SCATTER_COMPUTE_H_
