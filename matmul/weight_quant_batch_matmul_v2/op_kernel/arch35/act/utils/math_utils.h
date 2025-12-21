/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_ACT_UTILS_MATH_UTILS_H
#define ARCH35_ACT_UTILS_MATH_UTILS_H

#include "device_utils.h"

namespace WeightQuantBatchMatmulV2::Arch35::Act {
template <typename T>
DEVICE constexpr T CeilDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { X_LOG("Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
DEVICE constexpr T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { X_LOG("Division by zero error!"); });
    return (a + b - 1) / b * b;
}

template <typename T>
DEVICE T Min(T a, T b)
{
#if defined(__CCE_KT_TEST__)
    return a < b ? a : b;
#else
    return min(a, b);
#endif
}

template <typename T>
DEVICE T Max(T a, T b)
{
#if defined(__CCE_KT_TEST__)
    return a < b ? b : a;
#else
    return max(a, b);
#endif
}

__aicore__ inline uint64_t CalcRealLen(uint64_t fullLen, uint64_t offset, uint64_t tileLen)
{
    return offset + tileLen > fullLen ? fullLen - offset : tileLen;
}
template <typename Dtype, typename Int>
__aicore__ inline Int ElemToByte(Int count)
{
    static_assert(
        AscendC::SupportType<
            Dtype, uint8_t, int8_t, half, bfloat16_t, int4b_t, int4x2_t, fp8_e5m2_t, fp8_e4m3fn_t, hifloat8_t>(),
        "not support this dtype");
    if (AscendC::Std::is_same_v<Dtype, int4b_t> || AscendC::Std::is_same_v<Dtype, int4x2_t>) {
        // int4类型的元素数量必须是2的倍数
        return count >> 1;
    }
    return count * sizeof(Dtype);
}

template <typename Dtype, typename Int, Int Count>
__aicore__ inline Int ElemToByte(AscendC::Std::integral_constant<Int, Count>)
{
    static_assert(
        AscendC::SupportType<
            Dtype, uint8_t, int8_t, half, bfloat16_t, int4b_t, int4x2_t, fp8_e5m2_t, fp8_e4m3fn_t, hifloat8_t>(),
        "not support this dtype");
    if (AscendC::Std::is_same_v<Dtype, int4b_t> || AscendC::Std::is_same_v<Dtype, int4x2_t>) {
        // int4类型的元素数量必须是2的倍数
        return Count >> 1;
    }
    return Count * sizeof(Dtype);
}
} // namespace WeightQuantBatchMatmulV2::Arch35::Act
#endif