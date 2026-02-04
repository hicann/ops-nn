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
 * \file embedding_bag_regbase_empty_simt.h
 * \brief
 */

#ifndef EMBEDDING_BAG_REGBASE_EMPTY_SIMT_H
#define EMBEDDING_BAG_REGBASE_EMPTY_SIMT_H

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "embedding_bag_regbase_common.h"

namespace EmbeddingBag {
using namespace AscendC;

template <typename P, typename COMP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUM) inline void SimtComputeOffst2Bag(
    __gm__ P* offset2bag, COMP_T numIndices)
{
    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < numIndices;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        offset2bag[i] = 0;
    }
}

template <typename P, typename COMP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUM) inline void SimtComputeBagSize(
    __gm__ P* bagSize, COMP_T numBags)
{
    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < numBags;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        bagSize[i] = 0;
    }
}

template<typename P, typename COMP_T>
class EmbeddingBagRegBaseSimtEmpty {
public:
    __aicore__ inline EmbeddingBagRegBaseSimtEmpty(const EmbeddingBagTilingData &tilingData, TPipe &pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR gmParam[TENSOR_COUNT]);
    __aicore__ inline void Process();

private:
    TPipe& pipe_;
    GlobalTensor<P> offset2bagGm_;
    GlobalTensor<P> bagSizeGm_;
    const EmbeddingBagTilingData& tilingData_;
};

template<typename P, typename COMP_T>
__aicore__ inline void EmbeddingBagRegBaseSimtEmpty<P, COMP_T>::Init(GM_ADDR gmParam[TENSOR_COUNT])
{
    offset2bagGm_.SetGlobalBuffer((__gm__ P*)(gmParam[OFFSET2BAG_OUTPUT_IDX]));
    bagSizeGm_.SetGlobalBuffer((__gm__ P*)(gmParam[BAGSIZE_OUTPUT_IDX]));
}

template<typename P, typename COMP_T>
__aicore__ inline void EmbeddingBagRegBaseSimtEmpty<P, COMP_T>::Process()
{
    COMP_T numBags = tilingData_.nBags;
    COMP_T numIndices = tilingData_.indicesNumel;

    Simt::VF_CALL<SimtComputeOffst2Bag<P, COMP_T>>(Simt::Dim3{USED_THREAD_NUM}, 
            (__gm__ P*)(offset2bagGm_.GetPhyAddr()), numIndices);

    if (tilingData_.inclueLastOfst) {
        numBags += 1;
    }
    Simt::VF_CALL<SimtComputeBagSize<P, COMP_T>>(Simt::Dim3{USED_THREAD_NUM}, 
            (__gm__ P*)(bagSizeGm_.GetPhyAddr()), numBags);
}
}

#endif // EMBEDDING_BAG_H_REGBASE_EMPTY_SIMT_H