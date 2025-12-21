/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file cross_entropy_loss_empty.h
 * \brief
 */

#ifndef CROSS_ENTROPY_LOSS_EMPTY_H
#define CROSS_ENTROPY_LOSS_EMPTY_H
#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace CrossEntropyLoss {
using namespace AscendC;

template <typename T1, uint64_t reduction>
class CrossEntropyLossEmpty
{
public:
    __aicore__ inline CrossEntropyLossEmpty(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR logProb, GM_ADDR zloss,
        GM_ADDR lseForZloss, GM_ADDR workspace, const CrossEntropyLossRegBaseTilingData* __restrict tilingData,
        TPipe* pipe);
    __aicore__ inline void Process();

private:
    GlobalTensor<T1> lossGm_;
};

template <typename T1, uint64_t reduction>
__aicore__ inline void CrossEntropyLossEmpty<T1, reduction>::Init(
    GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR logProb, GM_ADDR zloss, GM_ADDR lseForZloss,
    GM_ADDR workspace, const CrossEntropyLossRegBaseTilingData* __restrict tilingData, TPipe* pipe)
{
    lossGm_.SetGlobalBuffer((__gm__ T1*)loss);
}

template <typename T1, uint64_t reduction>
__aicore__ inline void CrossEntropyLossEmpty<T1, reduction>::Process()
{
    if constexpr (reduction == 0) { // none
        return;
    }
    if constexpr (reduction == 1) { // mean
        lossGm_.SetValue(0, static_cast<T1>(NAN));
    }
    if constexpr (reduction == 2) { // sum
        lossGm_.SetValue(0, 0.0f);
    }
    DataCacheCleanAndInvalid<T1, CacheLine::ENTIRE_DATA_CACHE>(lossGm_);
}
} // namespace CrossEntropyLoss
#endif // namespace CrossEntropyLossEmpty