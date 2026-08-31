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
 * \file max_pool_with_argmax_base.h
 * \brief
 */

#ifndef MAX_POOL_WITH_ARGMAX_BASE_H_
#define MAX_POOL_WITH_ARGMAX_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void DuplicateLowestReg(Reg::RegTensor<T>& negInfReg)
{
    // min
    constexpr uint32_t FLOAT32_MIN = 0xFF7FFFFF;
    constexpr uint16_t FLOAT16_MIN = 0xFBFF;
    constexpr uint16_t BFLOAT16_MIN = 0xFF7F;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT32_MIN));
    } else if constexpr (std::is_same<T, half>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT16_MIN));
    } else {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (BFLOAT16_MIN));
    }
}

#endif // MAX_POOL_WITH_ARGMAX_BASE_H_
