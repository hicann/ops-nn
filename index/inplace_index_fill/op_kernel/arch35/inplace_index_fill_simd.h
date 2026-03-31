/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_simd.h
 * \brief indices process and other public function
 */
#ifndef INPLACE_INDEX_FILL_SIMD_H
#define INPLACE_INDEX_FILL_SIMD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "inplace_index_fill_struct.h"

namespace InplaceIndexFill {
using namespace AscendC;

template <typename X_T, typename INDICES_T>
class InplaceIndexFillSimd 
{
public:
  __aicore__ inline InplaceIndexFillSimd(const InplaceIndexFillSimdTilingData& tilingData, TPipe& pipe):tilingData_(tilingData), pipe_(pipe) {};

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR workspace);

  __aicore__ inline void Process();

  __aicore__ inline void CopyOutValues(int64_t offset, int64_t dataLen, LocalTensor<X_T>& qValuesLocal);

  __aicore__ inline void CopyInIndices(int64_t offset, int64_t dataLen);

  __aicore__ inline INDICES_T GetIndex(int64_t idx);

  __aicore__ inline int64_t NormalizeIndex(int64_t indiceValue);

  __aicore__ inline void ProcessSplitQ(LocalTensor<X_T>& qValues);

  __aicore__ inline void ProcessNonSplitQ(LocalTensor<X_T>& qValues);

private:
    TPipe& pipe_;
    const InplaceIndexFillSimdTilingData& tilingData_;
    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<X_T> valueGm_;
    
    TBuf<QuePosition::VECCALC> qValueBuf_;
    TBuf<QuePosition::VECCALC> indicesBuf_;

    LocalTensor<INDICES_T> indicesLocal_;
    uint32_t blockIdx_ = 0;
    int64_t start_ = 0;
    int64_t end_ = 0;
    int64_t currentBlockElements_ = 0;
    int64_t blockOffsetBase_ = 0;
    int64_t indicesOffsetBase_ = 0;
    int64_t copyInIndicesCount_ = 0;
    X_T value_ = 0;
};

template <typename X_T, typename INDICES_T>
__aicore__ inline void InplaceIndexFillSimd<X_T, INDICES_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)(indices));
    valueGm_.SetGlobalBuffer((__gm__ X_T*)(value));

    pipe_.InitBuffer(qValueBuf_, tilingData_.qBufferSize);
    pipe_.InitBuffer(indicesBuf_, tilingData_.indicesBufferSize);
    value_ = valueGm_(0);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
}

template <typename X_T, typename INDICES_T>
__aicore__ inline int64_t InplaceIndexFillSimd<X_T, INDICES_T>::NormalizeIndex(int64_t indiceValue)
{
    if (indiceValue < -tilingData_.dimSize || indiceValue >= tilingData_.dimSize) {
        return -1;
    }
    if (indiceValue < 0) {
        indiceValue = indiceValue + tilingData_.dimSize;
    }
    return indiceValue;
}

template <typename X_T, typename INDICES_T>
__aicore__ inline void InplaceIndexFillSimd<X_T, INDICES_T>::ProcessSplitQ(
    LocalTensor<X_T>& qValues)
{
    int64_t pmIdx = blockIdx_ / tilingData_.qUsedCoreNum;
    int64_t qCoreId = blockIdx_ - pmIdx * tilingData_.qUsedCoreNum;
    int64_t pIdx = pmIdx / tilingData_.indicesNum;
    int64_t mIdx = pmIdx - pIdx * tilingData_.indicesNum;

    int64_t indiceValue = static_cast<int64_t>(indicesGm_(mIdx));
    indiceValue = NormalizeIndex(indiceValue);
    if (indiceValue < 0) {
        return;
    }

    int64_t qLoopSize = tilingData_.qLoopSize;
    int64_t qUbTailFactor = tilingData_.qUbTailFactor;

    // 重新计算按Q切分分核场景，每一行的行尾的Q对应的blockFactor, loopSize
    if (qCoreId == tilingData_.qUsedCoreNum - 1) {
        int64_t tailQBlockFactor = tilingData_.postDimProduct -
            (tilingData_.qUsedCoreNum - 1) * tilingData_.qBlockFactor;
        qLoopSize = Ops::Base::CeilDiv(tailQBlockFactor, tilingData_.qUbFactor);
        qUbTailFactor = tailQBlockFactor - (qLoopSize - 1) * tilingData_.qUbFactor;
    }

    int64_t baseOffset = pIdx * tilingData_.dimSize * tilingData_.postDimProduct +
        indiceValue * tilingData_.postDimProduct +
        qCoreId * tilingData_.qBlockFactor;
    for (int i = 0; i < qLoopSize; i++) {
        int64_t xGmOffset = baseOffset + i * tilingData_.qUbFactor;
        int64_t dataLen = i == qLoopSize - 1 ? qUbTailFactor : tilingData_.qUbFactor;
        CopyOutValues(xGmOffset, dataLen, qValues);
    }
}

template <typename X_T, typename INDICES_T>
__aicore__ inline void InplaceIndexFillSimd<X_T, INDICES_T>::ProcessNonSplitQ(
    LocalTensor<X_T>& qValues)
{
    start_ = blockOffsetBase_;
    end_ = blockOffsetBase_ + currentBlockElements_;
    copyInIndicesCount_ = tilingData_.indicesUbFactor;

    int64_t startPIdx = start_ / tilingData_.indicesNum;
    int64_t endPIdx = end_ / tilingData_.indicesNum;
    if (startPIdx == endPIdx) {
        int64_t startMIdx = start_ - startPIdx * tilingData_.indicesNum;
        int64_t endMIdx = end_ - endPIdx * tilingData_.indicesNum;
        if (copyInIndicesCount_ > endMIdx - startMIdx + 1) {
            copyInIndicesCount_ = endMIdx - startMIdx + 1;
        }
    }

    // 设置初值
    indicesOffsetBase_ = -tilingData_.indicesUbFactor - 1;
    indicesLocal_ = indicesBuf_.Get<INDICES_T>();
    for (int64_t idx = start_; idx < end_; idx++) {
        int64_t pIdx = idx / tilingData_.indicesNum;
        int64_t mIdx = idx - pIdx * tilingData_.indicesNum;
        int64_t indiceValue = static_cast<int64_t>(GetIndex(mIdx));
        indiceValue = NormalizeIndex(indiceValue);
        if (indiceValue < 0) {
            continue;
        }
        int64_t baseOffset = pIdx * tilingData_.dimSize * tilingData_.postDimProduct +
            indiceValue * tilingData_.postDimProduct;
        for (int i = 0; i < tilingData_.qLoopSize; i++) {
            int64_t xGmOffset = baseOffset + i * tilingData_.qUbFactor;
            int64_t dataLen = i == tilingData_.qLoopSize - 1 ?
                tilingData_.qUbTailFactor : tilingData_.qUbFactor;
            CopyOutValues(xGmOffset, dataLen, qValues);
        }
    }
}

template <typename X_T, typename INDICES_T>
__aicore__ inline void InplaceIndexFillSimd<X_T, INDICES_T>::Process()
{
    if (blockIdx_ > tilingData_.usedCoreNum) {
        return;
    }
    if (blockIdx_ < tilingData_.tailBlockData) {
        // 前面的每个核多处理一个
        blockOffsetBase_= (tilingData_.perBlockData + 1) * blockIdx_;
        currentBlockElements_ = tilingData_.perBlockData + 1;
    } else {
        blockOffsetBase_= tilingData_.perBlockData  * blockIdx_ + tilingData_.tailBlockData;
        currentBlockElements_ = tilingData_.perBlockData;
    }

    LocalTensor<X_T> qValues_ = qValueBuf_.Get<X_T>();
    //Duplicate
    Duplicate(qValues_, value_, tilingData_.qUbFactor);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

    if (tilingData_.qUsedCoreNum > 1) {
        ProcessSplitQ(qValues_);
    } else {
        ProcessNonSplitQ(qValues_);
    }
} 

template <typename X_T, typename INDICES_T> 
__aicore__ inline void InplaceIndexFillSimd<X_T, INDICES_T>::CopyOutValues(int64_t offset, int64_t count, LocalTensor<X_T>& valuesLocal) 
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(count * sizeof(X_T)), 0, 0, 0};
    DataCopyPad(xGm_[offset], valuesLocal, copyParams);
}

template <typename X_T, typename INDICES_T> 
__aicore__ inline void InplaceIndexFillSimd<X_T, INDICES_T>::CopyInIndices(int64_t offset, int64_t count) 
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(count * sizeof(INDICES_T)), 0, 0, 0};
    DataCopyPadExtParams<INDICES_T> padParams = {false, 0, 0, 0};
    DataCopyPad(indicesLocal_, indicesGm_[offset], copyParams, padParams);
}

template <typename X_T, typename INDICES_T> 
__aicore__ inline INDICES_T InplaceIndexFillSimd<X_T, INDICES_T>::GetIndex(int64_t idx) 
{
    //gm idx不在当前ub里，需要重新取, 每次取一个indicesUbFactor或者取到indices尾部
    if (idx >= indicesOffsetBase_ + copyInIndicesCount_ || idx < indicesOffsetBase_){
        if (idx + copyInIndicesCount_ > tilingData_.indicesNum) {
            copyInIndicesCount_ = tilingData_.indicesNum - idx;
        }
        CopyInIndices(idx, copyInIndicesCount_);
        indicesOffsetBase_ = idx;
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    } else {
        // 命中UB缓存，恢复copyInIndicesCount_为默认值，避免尾部缩小后影响下次搬运判断
        copyInIndicesCount_ = tilingData_.indicesUbFactor;
    } 
    return indicesLocal_.GetValue(idx - indicesOffsetBase_);
}
}
#endif