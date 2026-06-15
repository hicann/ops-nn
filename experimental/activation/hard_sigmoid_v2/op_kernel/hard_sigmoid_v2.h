/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_ACTIVATION_HARD_SIGMOID_V2_H_
#define EXPERIMENTAL_ACTIVATION_HARD_SIGMOID_V2_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_sigmoid_v2_tiling_data.h"
#include "hard_sigmoid_v2_tiling_key.h"

namespace NsHardSigmoidV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t QUEUE_DEPTH = 1;
constexpr float SHIFT_VALUE = 3.0f;
constexpr float CLAMP_VALUE = 6.0f;
constexpr float SCALE_VALUE = 1.0f / 6.0f;
constexpr int32_t INT32_SHIFT_VALUE = -2;
constexpr int32_t INT32_CLAMP_VALUE = 1;

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
class KernelHardSigmoidV2Base {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const HardSigmoidV2TilingData *tilingData)
    {
        InitBlockGm(x, y, tilingData);
        pipe_.InitBuffer(inOutQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        static_cast<Derived &>(*this).InitExtraBuffers();
    }

    __aicore__ inline void Process()
    {
        ProcessTiles(static_cast<Derived &>(*this), blockLength_, tileLength_);
    }

    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        LocalTensor<T> xLocal = inOutQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, xGm_[progress * tileLength_], copyParams, padParams);
        inOutQueue_.template EnQue<QuePosition::GM, QuePosition::VECIN, T>(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        LocalTensor<T> yLocal = inOutQueue_.template DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[progress * tileLength_], yLocal, copyParams);
        inOutQueue_.FreeTensor(yLocal);
    }

protected:
    __aicore__ inline void InitBlockGm(GM_ADDR x, GM_ADDR y, const HardSigmoidV2TilingData *tilingData)
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
    }

    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, QUEUE_DEPTH> inOutQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
};

template <typename T>
class KernelHardSigmoidV2 : public KernelHardSigmoidV2Base<KernelHardSigmoidV2<T>, T> {
public:
    __aicore__ inline KernelHardSigmoidV2() {}

    __aicore__ inline void InitExtraBuffers() {}

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        T shiftValue = static_cast<T>(SHIFT_VALUE);
        T clampValue = static_cast<T>(CLAMP_VALUE);
        T scaleValue = static_cast<T>(SCALE_VALUE);

        LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        Adds(xLocal, xLocal, shiftValue, curTileLength);
        Relu(xLocal, xLocal, curTileLength);
        Mins(xLocal, xLocal, clampValue, curTileLength);
        Muls(xLocal, xLocal, scaleValue, curTileLength);

        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
    }
};

template <typename T>
class KernelHardSigmoidV2Upcast : public KernelHardSigmoidV2Base<KernelHardSigmoidV2Upcast<T>, T> {
public:
    __aicore__ inline KernelHardSigmoidV2Upcast() {}

    __aicore__ inline void InitExtraBuffers()
    {
        this->pipe_.InitBuffer(tmpBufFp32_, this->tileLength_ * sizeof(float));
    }

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        LocalTensor<float> xFp32 = tmpBufFp32_.Get<float>();

        Cast(xFp32, xLocal, RoundMode::CAST_NONE, curTileLength);
        Adds(xFp32, xFp32, SHIFT_VALUE, curTileLength);
        Relu(xFp32, xFp32, curTileLength);
        Mins(xFp32, xFp32, CLAMP_VALUE, curTileLength);
        Muls(xFp32, xFp32, SCALE_VALUE, curTileLength);
        Cast(xLocal, xFp32, RoundMode::CAST_ROUND, curTileLength);

        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
    }

private:
    TBuf<TPosition::VECCALC> tmpBufFp32_;
};

class KernelHardSigmoidV2Int32 : public KernelHardSigmoidV2Base<KernelHardSigmoidV2Int32, int32_t> {
public:
    __aicore__ inline KernelHardSigmoidV2Int32() {}

    __aicore__ inline void InitExtraBuffers() {}

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        LocalTensor<int32_t> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, int32_t>();

        Adds(xLocal, xLocal, INT32_SHIFT_VALUE, curTileLength);
        Relu(xLocal, xLocal, curTileLength);
        Mins(xLocal, xLocal, INT32_CLAMP_VALUE, curTileLength);

        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, int32_t>(xLocal);
    }
};

}  // namespace NsHardSigmoidV2

#endif
