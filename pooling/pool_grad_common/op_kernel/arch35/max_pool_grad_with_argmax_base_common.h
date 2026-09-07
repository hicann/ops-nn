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
#include "pool_utils/arch35/compute/pool_grad_scatter_compute.h"

using namespace AscendC;
using PoolUtils::Compute::castTraitI64I32;
using PoolUtils::Compute::castTraitT1ComputeType;
using PoolUtils::Compute::castTraitU32U16;
using PoolUtils::Compute::FilterMask;
using PoolUtils::Compute::GenT2Mask;
using PoolUtils::Compute::GetConCurrentInput;
using PoolUtils::Compute::GradientAcc;
using PoolUtils::Compute::PEnd;
using PoolUtils::Compute::PStart;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t DOUBLE = 2;
constexpr uint32_t HELP_BUFFER = 1024;

constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;
using computeType = float;

constexpr uint32_t VER_NORMAL = 0;
constexpr uint32_t VER_V3 = 1;

__aicore__ inline constexpr uint32_t GetUbBlockSize() { return 32U; }

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310 || __NPU_ARCH == 5102
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE, int32_t VER>
__aicore__ inline void TransArgmaxHWC2HW(Reg::RegTensor<T3>& argmaxReg, int64_t curCIndex, int32_t cOutputActual)
{
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<T3, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Sub(argmaxReg, argmaxReg, curCIndex, allMask);
    AscendC::Reg::Div(argmaxReg, argmaxReg, cOutputActual, allMask);
}
#endif // MAX_POOL_GRAD_WITH_ARGMAX_BASE_COMMON_H_
