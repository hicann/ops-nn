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
 * \file nll_loss_empty.h
 * \brief nll_loss_empty
 */

#ifndef NLL_LOSS_EMPTY_H
#define NLL_LOSS_EMPTY_H
#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

template <typename T, uint64_t reduction>
class KernelNLLLossEmpty {
public:
    __aicore__ inline KernelNLLLossEmpty(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR totalWeight, GM_ADDR workspace,
        NLLLossACTilingData tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> yGm_;
    GlobalTensor<T> totalWeightGm_;
};

template <typename T, uint64_t reduction>
__aicore__ inline void KernelNLLLossEmpty<T, reduction>::Init(
    GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR totalWeight, GM_ADDR workspace,
    NLLLossACTilingData tilingData)
{
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    totalWeightGm_.SetGlobalBuffer((__gm__ T*)totalWeight);
}

template <typename T, uint64_t reduction>
__aicore__ inline void KernelNLLLossEmpty<T, reduction>::Process()
{
    if constexpr (reduction == 0) {
        return;
    } else if constexpr (reduction == 1) {
        yGm_.SetValue(0, static_cast<T>(NAN));
    } else {
        yGm_.SetValue(0, 0.0f);
    }

    totalWeightGm_.SetValue(0, 0.0f);

    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(yGm_);
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(totalWeightGm_);
}
#endif