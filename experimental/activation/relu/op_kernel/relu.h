/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_ACTIVATION_RELU_H_
#define EXPERIMENTAL_ACTIVATION_RELU_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t QUEUE_DEPTH = 1;

template <typename T>
__aicore__ inline void CopyInFromGm(TQue<TPosition::VECIN, BUFFER_NUM> &inQueueX, GlobalTensor<T> &xGm,
                                    int64_t progress, int64_t tileLength, int64_t curTileLength)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[progress * tileLength], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void CopyOutToGm(TQue<TPosition::VECOUT, BUFFER_NUM> &outQueueY, GlobalTensor<T> &yGm,
                                   int64_t progress, int64_t tileLength, int64_t curTileLength)
{
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm[progress * tileLength], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void CopyInFromGm(
    TQueBind<TPosition::VECIN, TPosition::VECOUT, QUEUE_DEPTH> &inOutQueue, GlobalTensor<T> &xGm, int64_t progress,
    int64_t tileLength, int64_t curTileLength)
{
    LocalTensor<T> xLocal = inOutQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[progress * tileLength], copyParams, padParams);
    inOutQueue.template EnQue<QuePosition::GM, QuePosition::VECIN, T>(xLocal);
}

template <typename T>
__aicore__ inline void CopyOutToGm(
    TQueBind<TPosition::VECIN, TPosition::VECOUT, QUEUE_DEPTH> &inOutQueue, GlobalTensor<T> &yGm,
    int64_t progress, int64_t tileLength, int64_t curTileLength)
{
    LocalTensor<T> yLocal = inOutQueue.template DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm[progress * tileLength], yLocal, copyParams);
    inOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void InitBlockGm(GlobalTensor<T> &xGm, GlobalTensor<T> &yGm, int64_t &blockLength,
                                   int64_t &tileLength, GM_ADDR x, GM_ADDR y, const ReluTilingData *tilingData)
{
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx < tilingData->formerNum) {
        blockLength = tilingData->formerLength;
        int64_t offset = tilingData->formerLength * blockIdx;
        xGm.SetGlobalBuffer((__gm__ T *)x + offset, tilingData->formerLength);
        yGm.SetGlobalBuffer((__gm__ T *)y + offset, tilingData->formerLength);
    } else {
        blockLength = tilingData->tailLength;
        int64_t offset = tilingData->formerLength * tilingData->formerNum;
        xGm.SetGlobalBuffer((__gm__ T *)x + offset, tilingData->tailLength);
        yGm.SetGlobalBuffer((__gm__ T *)y + offset, tilingData->tailLength);
    }

    tileLength = tilingData->tileLength;
}

template <typename Kernel>
__aicore__ inline void ProcessTiles(Kernel &kernel, int64_t blockLength, int64_t tileLength)
{
    int64_t tileNum = (blockLength + tileLength - 1) / tileLength;
    if (tileNum == 0) {
        return;
    }

    int64_t tailTileLength = blockLength - (tileNum - 1) * tileLength;
    for (int64_t i = 0; i < tileNum - 1; ++i) {
        kernel.CopyIn(i, tileLength);
        kernel.Compute(tileLength);
        kernel.CopyOut(i, tileLength);
    }
    kernel.CopyIn(tileNum - 1, tailTileLength);
    kernel.Compute(tailTileLength);
    kernel.CopyOut(tileNum - 1, tailTileLength);
}

template <typename Derived, typename T>
class KernelReluBase {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData *tilingData)
    {
        InitBlockGm(xGm_, yGm_, blockLength_, tileLength_, x, y, tilingData);
        pipe_.InitBuffer(inOutQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        ProcessTiles(static_cast<Derived &>(*this), blockLength_, tileLength_);
    }

    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        CopyInFromGm(inOutQueue_, xGm_, progress, tileLength_, curTileLength);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        CopyOutToGm(inOutQueue_, yGm_, progress, tileLength_, curTileLength);
    }

protected:
    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, QUEUE_DEPTH> inOutQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
};

template <typename T>
class KernelRelu : public KernelReluBase<KernelRelu<T>, T> {
public:
    __aicore__ inline KernelRelu() {}

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        Relu(xLocal, xLocal, curTileLength);
        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
    }
};

template <typename T, typename MidT>
class KernelReluUpcast : public KernelReluBase<KernelReluUpcast<T, MidT>, T> {
public:
    __aicore__ inline KernelReluUpcast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData *tilingData)
    {
        KernelReluBase<KernelReluUpcast<T, MidT>, T>::Init(x, y, tilingData);
        this->pipe_.InitBuffer(tmpBufX_, this->tileLength_ * sizeof(MidT));
    }

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        LocalTensor<MidT> xMid = tmpBufX_.Get<MidT>();
        Cast(xMid, xLocal, RoundMode::CAST_NONE, curTileLength);
        Relu(xMid, xMid, curTileLength);
        Cast(xLocal, xMid, RoundMode::CAST_RINT, curTileLength);
        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
    }

private:
    TBuf<TPosition::VECCALC> tmpBufX_;
};

template <typename T>
class KernelReluScalarInt64 {
public:
    __aicore__ inline KernelReluScalarInt64() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData *tilingData)
    {
        InitBlockGm(xGm_, yGm_, blockLength_, tileLength_, x, y, tilingData);
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int64_t tileNum = (blockLength_ + tileLength_ - 1) / tileLength_;
        if (tileNum == 0) {
            return;
        }

        int64_t tailTileLength = blockLength_ - (tileNum - 1) * tileLength_;
        for (int64_t i = 0; i < tileNum - 1; ++i) {
            CopyIn(i, tileLength_);
            Compute(tileLength_);
            CopyOut(i, tileLength_);
        }
        CopyIn(tileNum - 1, tailTileLength);
        Compute(tailTileLength);
        CopyOut(tileNum - 1, tailTileLength);
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        CopyInFromGm(inQueueX_, xGm_, progress, tileLength_, curTileLength);
    }

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();
        for (int64_t index = 0; index < curTileLength; ++index) {
            T value = xLocal.GetValue(index);
            yLocal.SetValue(index, value > static_cast<T>(0) ? value : static_cast<T>(0));
        }
        outQueueY_.EnQue<T>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        CopyOutToGm(outQueueY_, yGm_, progress, tileLength_, curTileLength);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
};

}  // namespace NsRelu

#endif
