/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file clear_output.h
 * \brief
 */

#ifndef CLEAR_OUTPUT_H
#define CLEAR_OUTPUT_H

#include "kernel_operator.h"
#include "segment_sum_struct.h"

namespace SegmentSum
{
using namespace AscendC;

template <typename T>
class AllClear
{
public:
    __aicore__ inline AllClear(){};
    __aicore__ inline void Init(GM_ADDR output, const SegmentSumSimdTilingData* tilingData, TPipe& pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    const SegmentSumSimdTilingData* tilingData_;
    GlobalTensor<T> outputGm_;
    uint32_t blockIdx_;
    int64_t curCoreProcessNumForClear_;
};

template <typename T>
__aicore__ inline void AllClear<T>::Init(GM_ADDR output, const SegmentSumSimdTilingData* tilingData, TPipe& pipeIn)
{   tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_->usedCoreNumForClear) {
        return;
    }

    // shield global memory address between different core
    uint64_t intraCoreOffset = blockIdx_ * tilingData_->normalCoreClearNum;

    // shield normal core and tail core
    curCoreProcessNumForClear_ = (blockIdx_ == tilingData_->usedCoreNumForClear - 1) ? tilingData_->tailCoreClearNum : tilingData_->normalCoreClearNum;

    outputGm_.SetGlobalBuffer((__gm__ T*)output + intraCoreOffset);
}

template <typename T>
__aicore__ inline void AllClear<T>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNumForClear) {
        return;
    }

    InitOutput<T>(outputGm_, curCoreProcessNumForClear_, 0);
}

template <typename T>
__aicore__ inline void AllClear<T>::SyncALLCores()
{
    SyncAll();
}

}  // namespace SegmentSum

#endif  // CLEAR_OUTPUT_H