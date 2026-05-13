/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_ACTIVATION_SIGMOID_H_
#define EXPERIMENTAL_ACTIVATION_SIGMOID_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_tiling_data.h"
#include "sigmoid_tiling_key.h"

namespace NsSigmoid {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t QUEUE_DEPTH = 1;

template <typename T, typename Derived>
class KernelSigmoidBase {
public:
    __aicore__ inline KernelSigmoidBase() {}

    __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR y, const SigmoidTilingData *tilingData)
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx < tilingData->formerNum) {
            blockLength_ = tilingData->formerLength;
            int64_t offset = tilingData->formerLength * blockIdx;
            xGm_.SetGlobalBuffer((__gm__ T *)x + offset, tilingData->formerLength);
            yGm_.SetGlobalBuffer((__gm__ T *)y + offset, tilingData->formerLength);
        } else {
            blockLength_ = tilingData->tailLength;
            int64_t offset = tilingData->formerLength * tilingData->formerNum;
            xGm_.SetGlobalBuffer((__gm__ T *)x + offset, tilingData->tailLength);
            yGm_.SetGlobalBuffer((__gm__ T *)y + offset, tilingData->tailLength);
        }

        tileLength_ = tilingData->tileLength;
        pipe_.InitBuffer(inOutQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
    }

    template <typename ComputeFunc>
    __aicore__ inline void ProcessBase(ComputeFunc &&compute)
    {
        int64_t tileNum = (blockLength_ + tileLength_ - 1) / tileLength_;
        if (tileNum == 0) {
            return;
        }

        int64_t tailTileLength = blockLength_ - (tileNum - 1) * tileLength_;
        for (int64_t i = 0; i < tileNum - 1; ++i) {
            CopyInBase(i, tileLength_);
            compute(tileLength_);
            CopyOutBase(i, tileLength_);
        }

        CopyInBase(tileNum - 1, tailTileLength);
        compute(tailTileLength);
        CopyOutBase(tileNum - 1, tailTileLength);
    }

    __aicore__ inline void Process()
    {
        this->ProcessBase([this](int64_t curTileLength) {
            static_cast<Derived *>(this)->ComputeBase(curTileLength);
        });
    }

protected:
    __aicore__ inline void CopyInBase(int64_t progress, int64_t curTileLength)
    {
        LocalTensor<T> xLocal = inOutQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, xGm_[progress * tileLength_], copyParams, padParams);
        inOutQueue_.template EnQue<QuePosition::GM, QuePosition::VECIN, T>(xLocal);
    }

    __aicore__ inline void CopyOutBase(int64_t progress, int64_t curTileLength)
    {
        LocalTensor<T> yLocal =
            inOutQueue_.template DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[progress * tileLength_], yLocal, copyParams);
        inOutQueue_.FreeTensor(yLocal);
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
class KernelSigmoid : public KernelSigmoidBase<T, KernelSigmoid<T>> {
public:
    __aicore__ inline KernelSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData *tilingData)
    {
        this->InitBase(x, y, tilingData);
        this->pipe_.InitBuffer(tmpBufOne_, this->tileLength_ * sizeof(float));
    }

    __aicore__ inline void ComputeBase(int64_t curTileLength)
    {
        LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        LocalTensor<float> oneLocal = tmpBufOne_.Get<float>();

        Muls(xLocal, xLocal, -1.0f, curTileLength);
        Exp(xLocal, xLocal, curTileLength);
        Adds(xLocal, xLocal, 1.0f, curTileLength);
        Duplicate(oneLocal, 1.0f, curTileLength);
        Div(xLocal, oneLocal, xLocal, curTileLength);

        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
    }

    TBuf<TPosition::VECCALC> tmpBufOne_;
};

template <typename T>
class KernelSigmoidUpcast : public KernelSigmoidBase<T, KernelSigmoidUpcast<T>> {
public:
    __aicore__ inline KernelSigmoidUpcast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SigmoidTilingData *tilingData)
    {
        this->InitBase(x, y, tilingData);
        this->pipe_.InitBuffer(tmpBufXFp32_, this->tileLength_ * sizeof(float));
        this->pipe_.InitBuffer(tmpBufFp32_, this->tileLength_ * sizeof(float));
    }

    __aicore__ inline void ComputeBase(int64_t curTileLength)
    {
        LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        LocalTensor<float> xFp32 = tmpBufXFp32_.Get<float>();
        LocalTensor<float> tmpLocal = tmpBufFp32_.Get<float>();

        Cast(xFp32, xLocal, RoundMode::CAST_NONE, curTileLength);
        Muls(xFp32, xFp32, -1.0f, curTileLength);
        Exp(xFp32, xFp32, curTileLength);
        Adds(xFp32, xFp32, 1.0f, curTileLength);
        Duplicate(tmpLocal, 1.0f, curTileLength);
        Div(xFp32, tmpLocal, xFp32, curTileLength);
        Cast(xLocal, xFp32, RoundMode::CAST_RINT, curTileLength);

        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
    }

    TBuf<TPosition::VECCALC> tmpBufXFp32_;
    TBuf<TPosition::VECCALC> tmpBufFp32_;
};

}  // namespace NsSigmoid

#endif
