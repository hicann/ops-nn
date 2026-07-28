/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HINGE_LOSS_H
#define HINGE_LOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hinge_loss_tiling_data.h"

namespace NsHingeLoss {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;

__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t align) { return (value + align - 1) / align * align; }

template <typename T>
class HingeLoss {
public:
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR target, GM_ADDR loss, const HingeLossTilingData* tilingData,
                                TPipe& pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint64_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(uint64_t progress);

    TQue<QuePosition::VECIN, BUFFER_NUM> predictQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> targetQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> lossQueue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> castPredictBuf;
    TBuf<QuePosition::VECCALC> castTargetBuf;
    TBuf<QuePosition::VECCALC> castResultBuf;
    GlobalTensor<T> predictGm;
    GlobalTensor<T> targetGm;
    GlobalTensor<T> lossGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename T>
__aicore__ inline void HingeLoss<T>::Init(GM_ADDR predict, GM_ADDR target, GM_ADDR loss,
                                          const HingeLossTilingData* tilingData, TPipe& pipe)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    const uint64_t coreIdx = AscendC::GetBlockIdx();
    uint64_t globalOffset = tilingData->bigCoreDataNum * coreIdx;
    tileDataNum = tilingData->tileDataNum;
    if (coreIdx < tilingData->tailBlockNum) {
        coreDataNum = tilingData->bigCoreDataNum;
        tileNum = tilingData->finalBigTileNum;
        tailDataNum = tilingData->bigTailDataNum;
    } else {
        coreDataNum = tilingData->smallCoreDataNum;
        tileNum = tilingData->finalSmallTileNum;
        tailDataNum = tilingData->smallTailDataNum;
        globalOffset = tilingData->smallCoreDataNum * coreIdx;
        if (tilingData->tailBlockNum > 0) {
            globalOffset += (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * tilingData->tailBlockNum;
        }
    }
    predictGm.SetGlobalBuffer((__gm__ T*)predict + globalOffset, coreDataNum);
    targetGm.SetGlobalBuffer((__gm__ T*)target + globalOffset, coreDataNum);
    lossGm.SetGlobalBuffer((__gm__ T*)loss + globalOffset, coreDataNum);
    const uint64_t dataAlignNum = AlignUp(tileDataNum * sizeof(T), BLOCK_SIZE) / sizeof(T);
    const uint64_t floatAlignNum = AlignUp(tileDataNum * sizeof(float), BLOCK_SIZE) / sizeof(float);
    pipe.InitBuffer(predictQueue, BUFFER_NUM, dataAlignNum * sizeof(T));
    pipe.InitBuffer(targetQueue, BUFFER_NUM, dataAlignNum * sizeof(T));
    pipe.InitBuffer(lossQueue, BUFFER_NUM, dataAlignNum * sizeof(T));
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        pipe.InitBuffer(castPredictBuf, floatAlignNum * sizeof(float));
        pipe.InitBuffer(castTargetBuf, floatAlignNum * sizeof(float));
        pipe.InitBuffer(castResultBuf, floatAlignNum * sizeof(float));
    } else {
        pipe.InitBuffer(tmpBuf, floatAlignNum * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void HingeLoss<T>::Process()
{
    processDataNum = tileDataNum;
    for (uint64_t index = 0; index < tileNum; ++index) {
        if (index == tileNum - 1) {
            processDataNum = tailDataNum;
        }
        CopyIn(index);
        Compute();
        CopyOut(index);
    }
}

template <typename T>
__aicore__ inline void HingeLoss<T>::CopyIn(uint64_t progress)
{
    LocalTensor<T> predictLocal = predictQueue.AllocTensor<T>();
    LocalTensor<T> targetLocal = targetQueue.AllocTensor<T>();
    DataCopyExtParams params = {1, static_cast<uint32_t>(processDataNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad = {true, 0, 0, static_cast<T>(0)};
    DataCopyPad(predictLocal, predictGm[progress * tileDataNum], params, pad);
    DataCopyPad(targetLocal, targetGm[progress * tileDataNum], params, pad);
    predictQueue.EnQue(predictLocal);
    targetQueue.EnQue(targetLocal);
}

template <typename T>
__aicore__ inline void HingeLoss<T>::Compute()
{
    LocalTensor<T> predictLocal = predictQueue.DeQue<T>();
    LocalTensor<T> targetLocal = targetQueue.DeQue<T>();
    LocalTensor<T> lossLocal = lossQueue.AllocTensor<T>();
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> predictFloat = castPredictBuf.Get<float>();
        LocalTensor<float> targetFloat = castTargetBuf.Get<float>();
        LocalTensor<float> resultFloat = castResultBuf.Get<float>();
        Cast(predictFloat, predictLocal, RoundMode::CAST_NONE, processDataNum);
        Cast(targetFloat, targetLocal, RoundMode::CAST_NONE, processDataNum);
        Mul(resultFloat, targetFloat, predictFloat, processDataNum);
        Muls(resultFloat, resultFloat, -1.0f, processDataNum);
        Adds(resultFloat, resultFloat, 1.0f, processDataNum);
        Maxs(resultFloat, resultFloat, 0.0f, processDataNum);
        Cast(lossLocal, resultFloat, RoundMode::CAST_RINT, processDataNum);
    } else {
        LocalTensor<float> margin = tmpBuf.Get<float>();
        Mul(margin, targetLocal, predictLocal, processDataNum);
        Muls(margin, margin, -1.0f, processDataNum);
        Adds(margin, margin, 1.0f, processDataNum);
        Maxs(lossLocal, margin, 0.0f, processDataNum);
    }
    lossQueue.EnQue(lossLocal);
    predictQueue.FreeTensor(predictLocal);
    targetQueue.FreeTensor(targetLocal);
}

template <typename T>
__aicore__ inline void HingeLoss<T>::CopyOut(uint64_t progress)
{
    LocalTensor<T> lossLocal = lossQueue.DeQue<T>();
    DataCopyExtParams params = {1, static_cast<uint32_t>(processDataNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(lossGm[progress * tileDataNum], lossLocal, params);
    lossQueue.FreeTensor(lossLocal);
}
} // namespace NsHingeLoss
#endif
