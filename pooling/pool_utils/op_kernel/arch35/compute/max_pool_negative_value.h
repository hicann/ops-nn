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
 * \file max_pool_negative_value.h
 * \brief MaxPool/MaxPool3D 共用的负无穷寄存器生成与 UB 缓冲负无穷填充接口，按数据类型填充对应位模式的 -inf 值。
 */

#ifndef POOL_UTILS_ARCH35_COMPUTE_MAX_POOL_NEGATIVE_VALUE_H_
#define POOL_UTILS_ARCH35_COMPUTE_MAX_POOL_NEGATIVE_VALUE_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

namespace PoolUtils {
namespace Compute {

template <typename T>
__aicore__ inline void DuplicateNegInfReg(AscendC::Reg::RegTensor<T>& negInfReg)
{
    // -inf
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr (std::is_same<T, half>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

template <typename T>
__simd_callee__ inline void DuplicateNegInfRegVF(AscendC::Reg::RegTensor<T>& negInfReg)
{
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr (std::is_same<T, half>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

/*
 * 功能：把 UB 缓冲按 repeatElm 为步长整体填充为负无穷，loop 为整块次数，tail 为尾块元素数。
 */
template <typename T>
__aicore__ inline void DupBufferNegInfCommon(__ubuf__ T* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail)
{
    AscendC::Reg::RegTensor<T> v0;
    PoolUtils::Compute::DuplicateNegInfReg<T>(v0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < loop; i++) {
        AscendC::Reg::StoreAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
    }
    preg = AscendC::Reg::UpdateMask<T>(tail);
    AscendC::Reg::StoreAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
}

} // namespace Compute
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_COMPUTE_MAX_POOL_NEGATIVE_VALUE_H_
