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
 * \file all_clear.h
 * \brief
 */

#ifndef ALL_CLEAR_H
#define ALL_CLEAR_H

#include "kernel_operator.h"

namespace SparseSegmentMeanNameSpace
{
using namespace AscendC;

template <typename T>
class AllClear
{
public:
    __aicore__ inline AllClear(){};
    __aicore__ inline void Init(GM_ADDR output, const SparseSegmentMeanSimdTilingData* tilingData, TPipe& pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void SyncALLCoresSimt();

    const SparseSegmentMeanSimdTilingData* tilingData_;
    GlobalTensor<T> outputGm_;
    uint32_t blockIdx_;
    int64_t curCoreProcessNumForClear_;
};

template <typename T>
__aicore__ inline void AllClear<T>::Init(GM_ADDR output, const SparseSegmentMeanSimdTilingData* tilingData, TPipe& pipeIn)
{   tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_->usedCoreNumForClear) {
        return;
    }

    // shield global memory address between different core
    uint64_t intraCoreOffset = blockIdx_ * tilingData_->normalCoreProcessNumForClear;

    // shield normal core and tail core
    curCoreProcessNumForClear_ = (blockIdx_ + 1 == tilingData_->usedCoreNumForClear) ? tilingData_->tailCoreProcessNumForClear : tilingData_->normalCoreProcessNumForClear;

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

}  // namespace SparseSegmentMeanNameSpace

#endif  // ALL_CLEAR_H