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

__aicore__ inline int64_t PStart(int64_t index, int64_t pad, int64_t kernel, int64_t dilation, int64_t stride)
{
    return (index + pad < (kernel - 1) * dilation + 1) ? 0 : (index + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
};
__aicore__ inline int64_t PEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize)
{
    return (index + pad) / stride + 1 < pooledSize ? (index + pad) / stride + 1 : pooledSize;
};

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

__aicore__ inline void FilterMask(MicroAPI::MaskReg& preg, MicroAPI::RegTensor<int32_t>& hIndexReg,
                                  MicroAPI::RegTensor<int32_t>& wIndexReg, MicroAPI::RegTensor<int32_t>& zeroConstReg,
                                  MicroAPI::RegTensor<int32_t>& wMaxReg, MicroAPI::RegTensor<int32_t>& hMaxReg)
{
    AscendC::MicroAPI::MaskReg gtMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, hIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, hMaxReg, hIndexReg, gtMask);

    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GE>(gtMask, wIndexReg, zeroConstReg, gtMask);
    AscendC::MicroAPI::Compare<int32_t, CMPMODE::GT>(gtMask, wMaxReg, wIndexReg, gtMask);
    AscendC::MicroAPI::MaskAnd(preg, preg, gtMask, allMask);
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
                                          MicroAPI::RegTensor<uint32_t>& parallelRegIndex, MicroAPI::MaskReg& pregT1,
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
        AscendC::MicroAPI::UnPack((AscendC::MicroAPI::RegTensor<uint32_t>&)gradRegT1,
                                  (AscendC::MicroAPI::RegTensor<uint16_t>&)gradRegT1);
        AscendC::MicroAPI::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::MicroAPI::DataCopyGather(gradReg, gradAddr, parallelRegIndex, pregT1);
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

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER>
__aicore__ inline void TransArgmaxHWC2HW(MicroAPI::RegTensor<T3>& argmaxReg, int64_t curCIndex, int32_t cOutputActual) {
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Sub(argmaxReg, argmaxReg, curCIndex, allMask);
    AscendC::MicroAPI::Div(argmaxReg, argmaxReg, cOutputActual, allMask);
}
#endif  // MAX_POOL_GRAD_WITH_ARGMAX_BASE_COMMON_H_
