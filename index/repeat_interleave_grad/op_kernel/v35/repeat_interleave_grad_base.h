/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_base.h
 * \brief
 */

#ifndef REPEAT_INTERLEAVE_GRAD_BASE_H
#define REPEAT_INTERLEAVE_GRAD_BASE_H

#include "kernel_operator.h"
#include "kernel_utils.h"
#include "platform.h"
#include "repeat_interleave_grad_david_tiling_data.h"

namespace RepeatInterleaveGrad {
using namespace AscendC;

constexpr int32_t CONST1 = 1;
constexpr int32_t CONST2 = 2;
constexpr int32_t CONST4 = 4;
constexpr int64_t CONST_SIXTY_THREE = 63;

constexpr int32_t VL_LENGTH_B = platform::GetVRegSize();
constexpr uint64_t BLOCK_SIZE_BYTE = platform::GetUbBlockSize();

namespace __RIGType {
template <typename DType>
struct GetPromoteType {
};

template <>
struct GetPromoteType<half> {
    using T = float;
};

template <>
struct GetPromoteType<bfloat16_t> {
    using T = float;
};

template <>
struct GetPromoteType<float> {
    using T = float;
};

} // namespace __RIGType

namespace __RIGUtil {
template <HardEvent EVENT>
__aicore__ inline void SetEvent(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<EVENT>(eventId);
    WaitFlag<EVENT>(eventId);
}

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

__aicore__ inline int32_t CalPow2(int32_t index)
{
    return 1 << index;
}

__aicore__ inline int32_t min(int32_t a, int32_t b)
{
    return a > b ? b : a;
}

__aicore__ inline int64_t GetCacheID(const int64_t idx)
{
    return ScalarGetCountOfValue<CONST1>(idx ^ (idx + CONST1)) - CONST1;
}

__aicore__ inline uint64_t FindNearestPower2(const uint64_t value)
{
    if (value == 0) {
        return 0;
    } else if (value <= CONST2) {
        return CONST1;
    } else if (value <= CONST4) {
        return CONST2;
    } else {
        const uint64_t num = value - CONST1;
        const uint64_t pow = CONST_SIXTY_THREE - ScalarCountLeadingZero(num);
        return (CONST1 << pow);
    }
}

__aicore__ inline uint64_t CalLog2(uint64_t value)
{
    uint64_t res = 0;
    while (value > 1) {
        value = value >> 1;
        res++;
    }
    return res;
}

__aicore__ inline void DoUbSplit(int64_t splitLen, int64_t splitFactor, UbParaUnit& ubPara)
{
    if (splitLen <= 0) {
        return;
    }
    ubPara.ubFactor = splitLen < splitFactor ? splitLen : splitFactor;
    ubPara.ubCount = ops::CeilDiv(splitLen, static_cast<int64_t>(ubPara.ubFactor));
    ubPara.ubTailFactor = (splitLen % ubPara.ubFactor == 0) ? ubPara.ubFactor : splitLen % ubPara.ubFactor;
    ubPara.ubFactor = (ubPara.ubCount == 1) ? 0 : ubPara.ubFactor;
}

} // namespace __RIGUtil
} // namespace RepeatInterleaveGrad
#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_H