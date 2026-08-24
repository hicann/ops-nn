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
 * \file bn_infer.h
 * \brief BNInfer AscendC common helpers.
 */
#ifndef BN_INFER_H
#define BN_INFER_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "bn_infer_tiling_def.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "bn_infer_regbase_common.h"

namespace BNInferOps {
__aicore__ inline constexpr uint32_t GetUbBlockSize() { return 32U; }

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310 || __NPU_ARCH == 5102
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T>
__aicore__ inline T FloorDiv(T a, T b)
{
    return a / b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t,
                                           uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T, typename U>
__aicore__ inline auto AlignUp(T value, U base) -> typename std::common_type<T, U>::type
{
    using R = typename std::common_type<T, U>::type;
    return CeilDiv(static_cast<R>(value), static_cast<R>(base)) * static_cast<R>(base);
}
} // namespace BNInferOps

#endif
