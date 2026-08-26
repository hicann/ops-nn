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
 * \file max_pool_grad_with_argmax_base_common.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_BASE_COMMON_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_BASE_COMMON_H_

#include "max_pool_grad_with_argmax_struct_common.h"

using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t DOUBLE = 2;
constexpr uint32_t HELP_BUFFER = 1024;

constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;
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

__aicore__ inline int64_t PStart(int64_t index, int64_t pad, int64_t kernel, int64_t dilation, int64_t stride)
{
    return (index + pad < (kernel - 1) * dilation + 1) ? 0 : (index + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
};
__aicore__ inline int64_t PEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize)
{
    return (index + pad) / stride + 1 < pooledSize ? (index + pad) / stride + 1 : pooledSize;
};

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
    AscendC::Reg::MaskAnd(preg, preg, gtMask, allMask);
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

template <typename T1, typename T2, typename T3>
__aicore__ inline void GetConCurrentInput(Reg::RegTensor<T3>& argmaxReg, Reg::RegTensor<computeType>& gradReg,
                                          __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
                                          Reg::RegTensor<uint32_t>& parallelRegIndex, Reg::MaskReg& pregT1,
                                          Reg::MaskReg& pregT2)
{
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex, pregT1,
                                   pregT2);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER>
__aicore__ inline void TransArgmaxHWC2HW(Reg::RegTensor<T3>& argmaxReg, int64_t curCIndex, int32_t cOutputActual)
{
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<T3, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Sub(argmaxReg, argmaxReg, curCIndex, allMask);
    AscendC::Reg::Div(argmaxReg, argmaxReg, cOutputActual, allMask);
}
#endif // MAX_POOL_GRAD_WITH_ARGMAX_BASE_COMMON_H_
