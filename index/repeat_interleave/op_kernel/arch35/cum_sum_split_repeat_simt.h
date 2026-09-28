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
 * \file cum_sum_split_repeat_simt.h
 * \brief
 */

#ifndef CUM_SUM_SPLIT_REPEAT_SIMT_H
#define CUM_SUM_SPLIT_REPEAT_SIMT_H

#include "op_kernel/platform_util.h"
#include "repeat_interleave_base.h"
#include "op_kernel/math_util.h"

namespace RepeatInterleave {
using namespace AscendC;

constexpr uint32_t SEARCH_THREAD_NUM = 2;

template <typename T, typename U, typename V, typename AddrType>
class SplitRepeatSumSimt {
public:
    __aicore__ inline SplitRepeatSumSimt(const RepeatInterleaveCumSumTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace);

    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<U> repeatsGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<V> prefixSumGm_;

    TBuf<QuePosition::VECCALC> tmpBuf_;

    TPipe& pipe_;
    const RepeatInterleaveCumSumTilingData& tilingData_;

    AddrType startRepeatsIdx_ = 0; // 当前核处理的repeats开头对应的repeatsIdx
    U startRepeatsIdxResNum_ = 0;  // 当前核处理的repeats开头对应的repeatsIdx剩余复制几次
    AddrType endRepeatsIdx_ = 0;   // 当前核处理的repeats结尾对应的repeatsIdx
    U endRepeatsIdxResNum_ = 0;    // 当前核处理的repeats结尾对应的repeatsIdx剩余复制几次

    AddrType repeatsStart_ = 0; // 当前核处理的repeats开头
    AddrType repeatsNum_ = 0;   // 当前核处理的repeats数
    AddrType repeatsEnd_ = 0;   // 当前核处理的repeats结尾

    int64_t batchCoreIdx_ = 0;
    int64_t repeatCoreIdx_ = 0;
};

template <typename T, typename U, typename V, typename AddrType>
__aicore__ inline void SplitRepeatSumSimt<T, U, V, AddrType>::Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y,
                                                                   GM_ADDR workspace)
{
    batchCoreIdx_ = AscendC::GetBlockIdx() / tilingData_.repeatsCoreNum;
    repeatCoreIdx_ = AscendC::GetBlockIdx() % tilingData_.repeatsCoreNum;

    xGm_.SetGlobalBuffer((__gm__ T*)x + batchCoreIdx_ * tilingData_.mergedDims[1] * tilingData_.mergedDims[2]);
    repeatsGm_.SetGlobalBuffer((__gm__ U*)repeats);
    yGm_.SetGlobalBuffer((__gm__ T*)y + batchCoreIdx_ * tilingData_.totalRepeatSum * tilingData_.mergedDims[2]);
    prefixSumGm_.SetGlobalBuffer((__gm__ V*)workspace);

    pipe_.InitBuffer(tmpBuf_, platform::GetUbBlockSize());

    repeatsStart_ = repeatCoreIdx_ * tilingData_.normalCoreOutputRepeats;
    repeatsNum_ = repeatCoreIdx_ != (tilingData_.repeatsCoreNum - 1) ? tilingData_.normalCoreOutputRepeats :
                                                                       tilingData_.tailCoreOutputRepeats;
    repeatsEnd_ = repeatsStart_ + repeatsNum_ - 1;
}

template <typename T, typename U, typename V, typename AddrType>
__aicore__ inline void SplitRepeatSumSimt<T, U, V, AddrType>::Process()
{
    if (AscendC::GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }

    LocalTensor<AddrType> tmpLocal = tmpBuf_.Get<AddrType>();

    asc_vf_call<SimtSearchStartEnd<U, V, AddrType>>(
        dim3(SEARCH_THREAD_NUM), tilingData_.mergedDims[1], repeatsNum_, repeatsStart_, repeatsEnd_,
        (__ubuf__ AddrType*)(tmpLocal.GetPhyAddr()), (__gm__ V*)(prefixSumGm_.GetPhyAddr()));

    auto sWiatVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(sWiatVEventID);
    WaitFlag<HardEvent::V_S>(sWiatVEventID);

    startRepeatsIdx_ = tmpLocal.GetValue(0);
    startRepeatsIdxResNum_ = tmpLocal.GetValue(1);
    endRepeatsIdx_ = tmpLocal.GetValue(2);
    endRepeatsIdxResNum_ = tmpLocal.GetValue(3);

    auto sWiatVEventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(sWiatVEventID2);
    WaitFlag<HardEvent::S_V>(sWiatVEventID2);

    uint32_t threadNumX = static_cast<uint32_t>(tilingData_.threadNumX);
    uint32_t threadNumY = static_cast<uint32_t>(tilingData_.threadNumY);

    asc_vf_call<SimtSplitRepeats<T, U, V, AddrType>>(
        dim3{threadNumX, threadNumY}, startRepeatsIdx_, endRepeatsIdx_, startRepeatsIdxResNum_, endRepeatsIdxResNum_,
        tilingData_.mergedDims[2], (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ U*)(repeatsGm_.GetPhyAddr()),
        (__gm__ T*)(yGm_.GetPhyAddr()), (__gm__ V*)(prefixSumGm_.GetPhyAddr()));
}

} // namespace RepeatInterleave

#endif
