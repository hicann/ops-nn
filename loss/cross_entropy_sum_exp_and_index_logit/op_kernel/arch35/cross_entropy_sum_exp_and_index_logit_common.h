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
 * \file cross_entropy_sum_exp_and_index_logit_common.h
 * \brief A5 (ascend950) common utilities and constants
 */
#ifndef CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_COMMON_H_
#define CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_COMMON_H_

#include "kernel_operator.h"

namespace CrossEntropySumExpAndIndexLogit {
using namespace AscendC;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t BLOCK_SIZE = 32;       // 32B data block
constexpr int64_t FP32_PER_BLOCK = 8;    // FP32 每 32B block 元素数
constexpr int32_t MAX_REPEAT = 255;      // 基础归约指令 repeatTime 上限
constexpr int32_t FP32_REPEAT_ELEM = 64; // FP32 单 repeat 归约元素上限（8 datablock）
constexpr int64_t BUFFER_NUM = 2;        // 双缓冲
constexpr int64_t REPEAT_SIZE = 256;     // 矢量单次迭代 256B（VF LoadAlign/StoreAlign repeat 步进）

// BF16→FP32 Cast trait（高位零扩展，mask 外清零；与 common/inc/op_kernel/load_store_utils.h castTraitB162B32 一致）
constexpr AscendC::Reg::CastTrait castTraitBf16ToFp32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <HardEvent event>
__aicore__ inline void SetWaitFlag()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    return static_cast<T1>((static_cast<int64_t>(a) + static_cast<int64_t>(b) - 1) / static_cast<int64_t>(b));
}

// 元素数向上对齐到 32B（按 sizeof(T) 计算每 block 元素数）
template <typename T>
__aicore__ inline int64_t AlignUp32(int64_t n)
{
    int64_t perBlock = BLOCK_SIZE / sizeof(T);
    return (n + perBlock - NUM_ONE) / perBlock * perBlock;
}

// 元素数向上对齐到 256B（VF LoadAlign 无 mask 按 256B 读，buffer 必须对齐防越界）
template <typename T>
__aicore__ inline int64_t AlignUp256(int64_t n)
{
    int64_t perRepeat = REPEAT_SIZE / sizeof(T);
    return (n + perRepeat - NUM_ONE) / perRepeat * perRepeat;
}

} // namespace CrossEntropySumExpAndIndexLogit

#endif // CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_COMMON_H_
