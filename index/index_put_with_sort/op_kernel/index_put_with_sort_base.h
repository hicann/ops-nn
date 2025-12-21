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
 * \file index_put_with_sort_base.h
 * \brief
 */

#ifndef INDEX_PUT_WITH_SORTED_BASH_H
#define INDEX_PUT_WITH_SORTED_BASH_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace IndexPutWithSort {
using namespace AscendC;

constexpr uint64_t INDEX_UB_NUMS = 2048; // one time process index nums
constexpr uint64_t VALUES_UB_NUMS = 20480; // gather block size
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t GROUP_NUM = 10; // add group
constexpr uint64_t SYNC_NUMS = 3;
constexpr uint64_t TBUF_BLOCK_SIZE = 12288; // scatter block size
constexpr uint64_t SYNC_SIZE = 8;
constexpr uint64_t ALIGNED_NUM = 16;

__aicore__ inline void PIPE_V_S() {
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

__aicore__ inline void PIPE_MTE2_S() {
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

__aicore__ inline void PIPE_MTE3_S() {
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

__aicore__ inline void PIPE_S_MTE3() {
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}

template<typename T>
__aicore__ inline void CopyGm2Ub(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor, const uint32_t count) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
}

template<typename T>
__aicore__ inline void CopyUb2Gm(GlobalTensor<T> dstTensor, LocalTensor<T> srcTensor, const uint32_t count) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(dstTensor, srcTensor, copyParams);
}

template<typename T>
__aicore__ inline void AddUb2Gm(GlobalTensor<T> dstTensor, LocalTensor<T> srcTensor, const uint32_t count) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    SetAtomicAdd<T>();
    DataCopyPad(dstTensor, srcTensor, copyParams);
    SetAtomicNone();
}

/**
    @brief 对任务进行划分，划分为整份和尾份
    @param taskNums 任务个数
    @param blockSize 整份任务的大小
    @param taskBlocks 整份任务份数
    @param taskLeft 剩余任务个数
*/
template<typename T>
__aicore__ inline void TaskDivision(const T taskNums, const T blockSize, T& taskBlocks, T& taskLeft) {
    taskBlocks = taskNums / blockSize;
    taskLeft = taskNums - taskBlocks * blockSize;
}
}

#endif // INDEX_PUT_WITH_SORTED_BASH_H