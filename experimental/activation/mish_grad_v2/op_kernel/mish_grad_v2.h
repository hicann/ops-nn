/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_ACTIVATION_MISH_GRAD_V2_H_
#define EXPERIMENTAL_ACTIVATION_MISH_GRAD_V2_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mish_grad_v2_tiling_data.h"
#include "mish_grad_v2_tiling_key.h"

namespace NsMishGradV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t DATA_COPY_ALIGN_BYTES = 32;

__aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
{
    int64_t safeAlign = (align <= 0) ? 1 : align;
    return ((value + safeAlign - 1) / safeAlign) * safeAlign;
}

template <typename T>
__aicore__ inline void CopyOutTile(GlobalTensor<T> &yGm, TQue<TPosition::VECOUT, BUFFER_NUM> &outQueueY,
                                   int64_t progress, int64_t tileLength, int64_t validLength)
{
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm[progress * tileLength], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

__aicore__ inline void ComputeCoreCommon(LocalTensor<float> gradLocal, LocalTensor<float> xLocal,
                                         LocalTensor<float> tanhxLocal, LocalTensor<float> yLocal,
                                         LocalTensor<float> tmp0, int64_t alignedLength)
{
    Mul(yLocal, tanhxLocal, tanhxLocal, alignedLength);
    Muls(yLocal, yLocal, -1.0f, alignedLength);
    Adds(yLocal, yLocal, 1.0f, alignedLength);

    Muls(tmp0, xLocal, -1.0f, alignedLength);
    Exp(tmp0, tmp0, alignedLength);
    Adds(tmp0, tmp0, 1.0f, alignedLength);

    Div(yLocal, yLocal, tmp0, alignedLength);
    Mul(yLocal, yLocal, xLocal, alignedLength);
    Add(yLocal, yLocal, tanhxLocal, alignedLength);
    Mul(yLocal, yLocal, gradLocal, alignedLength);
}

__aicore__ inline void ComputeTanhxCommon(LocalTensor<float> xLocal, LocalTensor<float> tanhxLocal,
                                          LocalTensor<float> tmp0, LocalTensor<float> tmp1,
                                          int64_t alignedLength)
{
    Relu(tanhxLocal, xLocal, alignedLength);
    Abs(tmp0, xLocal, alignedLength);
    Muls(tmp0, tmp0, -1.0f, alignedLength);
    Exp(tmp0, tmp0, alignedLength);
    Adds(tmp0, tmp0, 1.0f, alignedLength);
    Ln(tmp1, tmp0, alignedLength);
    Add(tanhxLocal, tanhxLocal, tmp1, alignedLength);
    Tanh(tanhxLocal, tanhxLocal, alignedLength);
}

class KernelMishGradFloat {
public:
    __aicore__ inline KernelMishGradFloat() {}

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR tanhx, GM_ADDR y,
                                const MishGradV2TilingData *tilingData, TPipe *pipe)
    {
        pipe_ = pipe;
        hasTanhx_ = tilingData->haveTanhx != 0;
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx < tilingData->formerNum) {
            blockLength_ = tilingData->formerLength;
            int64_t offset = tilingData->formerLength * blockIdx;
            gradGm_.SetGlobalBuffer((__gm__ float *)grad + offset, tilingData->formerLength);
            xGm_.SetGlobalBuffer((__gm__ float *)x + offset, tilingData->formerLength);
            if (hasTanhx_) {
                tanhxGm_.SetGlobalBuffer((__gm__ float *)tanhx + offset, tilingData->formerLength);
            }
            yGm_.SetGlobalBuffer((__gm__ float *)y + offset, tilingData->formerLength);
        } else {
            blockLength_ = tilingData->tailLength;
            int64_t offset = tilingData->formerLength * tilingData->formerNum;
            gradGm_.SetGlobalBuffer((__gm__ float *)grad + offset, tilingData->tailLength);
            xGm_.SetGlobalBuffer((__gm__ float *)x + offset, tilingData->tailLength);
            if (hasTanhx_) {
                tanhxGm_.SetGlobalBuffer((__gm__ float *)tanhx + offset, tilingData->tailLength);
            }
            yGm_.SetGlobalBuffer((__gm__ float *)y + offset, tilingData->tailLength);
        }

        tileLength_ = tilingData->tileLength;
        pipe_->InitBuffer(inQueueGrad_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_->InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(float));
        if (hasTanhx_) {
            pipe_->InitBuffer(inQueueTanhx_, BUFFER_NUM, tileLength_ * sizeof(float));
        } else {
            pipe_->InitBuffer(tmpBuf1_, tileLength_ * sizeof(float));
        }
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_->InitBuffer(tmpBuf0_, tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t tileNum = (blockLength_ + tileLength_ - 1) / tileLength_;
        if (tileNum == 0) {
            return;
        }

        constexpr int64_t alignNum = DATA_COPY_ALIGN_BYTES / static_cast<int64_t>(sizeof(float));
        for (int64_t i = 0; i < tileNum; ++i) {
            int64_t validLength = tileLength_;
            if (i == tileNum - 1) {
                validLength = blockLength_ - (tileNum - 1) * tileLength_;
            }
            int64_t alignedLength = AlignUp(validLength, alignNum);
            CopyIn(i, validLength, alignedLength);
            Compute(validLength, alignedLength);
            CopyOutTile(yGm_, outQueueY_, i, tileLength_, validLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t validLength, int64_t alignedLength)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(alignedLength - validLength), 0.0f};

        LocalTensor<float> gradLocal = inQueueGrad_.AllocTensor<float>();
        LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();

        DataCopyPad(gradLocal, gradGm_[progress * tileLength_], copyParams, padParams);
        DataCopyPad(xLocal, xGm_[progress * tileLength_], copyParams, padParams);

        inQueueGrad_.EnQue(gradLocal);
        inQueueX_.EnQue(xLocal);
        if (hasTanhx_) {
            LocalTensor<float> tanhxLocal = inQueueTanhx_.AllocTensor<float>();
            DataCopyPad(tanhxLocal, tanhxGm_[progress * tileLength_], copyParams, padParams);
            inQueueTanhx_.EnQue(tanhxLocal);
        }
    }

    __aicore__ inline void Compute(int64_t validLength, int64_t alignedLength)
    {
        LocalTensor<float> gradLocal = inQueueGrad_.DeQue<float>();
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();
        LocalTensor<float> tmp0 = tmpBuf0_.Get<float>();
        LocalTensor<float> tanhxLocal;

        if (hasTanhx_) {
            tanhxLocal = inQueueTanhx_.DeQue<float>();
        } else {
            LocalTensor<float> tmp1 = tmpBuf1_.Get<float>();
            tanhxLocal = tmp1;
            ComputeTanhxCommon(xLocal, tanhxLocal, yLocal, tmp0, alignedLength);
        }

        ComputeCoreCommon(gradLocal, xLocal, tanhxLocal, yLocal, tmp0, alignedLength);

        outQueueY_.EnQue<float>(yLocal);
        inQueueGrad_.FreeTensor(gradLocal);
        inQueueX_.FreeTensor(xLocal);
        if (hasTanhx_) {
            inQueueTanhx_.FreeTensor(tanhxLocal);
        }
    }

private:
    TPipe *pipe_ = nullptr;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueGrad_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueTanhx_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> tmpBuf0_;
    TBuf<TPosition::VECCALC> tmpBuf1_;
    GlobalTensor<float> gradGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> tanhxGm_;
    GlobalTensor<float> yGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
    bool hasTanhx_ = false;
};

template <typename T>
class KernelMishGradCast {
public:
    __aicore__ inline KernelMishGradCast() {}

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR tanhx, GM_ADDR y,
                                const MishGradV2TilingData *tilingData, TPipe *pipe)
    {
        pipe_ = pipe;
        hasTanhx_ = tilingData->haveTanhx != 0;
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx < tilingData->formerNum) {
            blockLength_ = tilingData->formerLength;
            int64_t offset = tilingData->formerLength * blockIdx;
            gradGm_.SetGlobalBuffer((__gm__ T *)grad + offset, tilingData->formerLength);
            xGm_.SetGlobalBuffer((__gm__ T *)x + offset, tilingData->formerLength);
            if (hasTanhx_) {
                tanhxGm_.SetGlobalBuffer((__gm__ T *)tanhx + offset, tilingData->formerLength);
            }
            yGm_.SetGlobalBuffer((__gm__ T *)y + offset, tilingData->formerLength);
        } else {
            blockLength_ = tilingData->tailLength;
            int64_t offset = tilingData->formerLength * tilingData->formerNum;
            gradGm_.SetGlobalBuffer((__gm__ T *)grad + offset, tilingData->tailLength);
            xGm_.SetGlobalBuffer((__gm__ T *)x + offset, tilingData->tailLength);
            if (hasTanhx_) {
                tanhxGm_.SetGlobalBuffer((__gm__ T *)tanhx + offset, tilingData->tailLength);
            }
            yGm_.SetGlobalBuffer((__gm__ T *)y + offset, tilingData->tailLength);
        }

        tileLength_ = tilingData->tileLength;
        pipe_->InitBuffer(inQueueGrad_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_->InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(T));
        if (hasTanhx_) {
            pipe_->InitBuffer(inQueueTanhx_, BUFFER_NUM, tileLength_ * sizeof(T));
        } else {
            pipe_->InitBuffer(tmpBuf1_, tileLength_ * sizeof(float));
        }
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_->InitBuffer(gradBufFp32_, tileLength_ * sizeof(float));
        pipe_->InitBuffer(xBufFp32_, tileLength_ * sizeof(float));
        pipe_->InitBuffer(tanhxBufFp32_, tileLength_ * sizeof(float));
        pipe_->InitBuffer(tmpBuf0_, tileLength_ * sizeof(float));
        pipe_->InitBuffer(yBufFp32_, tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t tileNum = (blockLength_ + tileLength_ - 1) / tileLength_;
        if (tileNum == 0) {
            return;
        }

        int64_t alignNum = DATA_COPY_ALIGN_BYTES / static_cast<int64_t>(sizeof(T));
        for (int64_t i = 0; i < tileNum; ++i) {
            int64_t validLength = tileLength_;
            if (i == tileNum - 1) {
                validLength = blockLength_ - (tileNum - 1) * tileLength_;
            }
            int64_t alignedLength = AlignUp(validLength, alignNum);
            CopyIn(i, validLength, alignedLength);
            Compute(alignedLength);
            CopyOutTile(yGm_, outQueueY_, i, tileLength_, validLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t validLength, int64_t alignedLength)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(alignedLength - validLength), static_cast<T>(0)};

        LocalTensor<T> gradLocal = inQueueGrad_.AllocTensor<T>();
        LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();

        DataCopyPad(gradLocal, gradGm_[progress * tileLength_], copyParams, padParams);
        DataCopyPad(xLocal, xGm_[progress * tileLength_], copyParams, padParams);

        inQueueGrad_.EnQue(gradLocal);
        inQueueX_.EnQue(xLocal);
        if (hasTanhx_) {
            LocalTensor<T> tanhxLocal = inQueueTanhx_.AllocTensor<T>();
            DataCopyPad(tanhxLocal, tanhxGm_[progress * tileLength_], copyParams, padParams);
            inQueueTanhx_.EnQue(tanhxLocal);
        }
    }

    __aicore__ inline void Compute(int64_t alignedLength)
    {
        LocalTensor<T> gradLocal = inQueueGrad_.DeQue<T>();
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();

        LocalTensor<float> gradFp32 = gradBufFp32_.Get<float>();
        LocalTensor<float> xFp32 = xBufFp32_.Get<float>();
        LocalTensor<float> tanhxFp32 = tanhxBufFp32_.Get<float>();
        LocalTensor<float> tmp0 = tmpBuf0_.Get<float>();
        LocalTensor<float> yFp32 = yBufFp32_.Get<float>();

        Cast(gradFp32, gradLocal, RoundMode::CAST_NONE, alignedLength);
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, alignedLength);
        if (hasTanhx_) {
            LocalTensor<T> tanhxLocal = inQueueTanhx_.DeQue<T>();
            Cast(tanhxFp32, tanhxLocal, RoundMode::CAST_NONE, alignedLength);
            inQueueTanhx_.FreeTensor(tanhxLocal);
        } else {
            LocalTensor<float> tmp1 = tmpBuf1_.Get<float>();
            ComputeTanhxCommon(xFp32, tanhxFp32, tmp1, tmp0, alignedLength);
        }

        ComputeCoreCommon(gradFp32, xFp32, tanhxFp32, yFp32, tmp0, alignedLength);
        Cast(yLocal, yFp32, RoundMode::CAST_ROUND, alignedLength);

        outQueueY_.EnQue<T>(yLocal);
        inQueueGrad_.FreeTensor(gradLocal);
        inQueueX_.FreeTensor(xLocal);
    }

private:
    TPipe *pipe_ = nullptr;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueGrad_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueTanhx_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> gradBufFp32_;
    TBuf<TPosition::VECCALC> xBufFp32_;
    TBuf<TPosition::VECCALC> tanhxBufFp32_;
    TBuf<TPosition::VECCALC> tmpBuf0_;
    TBuf<TPosition::VECCALC> tmpBuf1_;
    TBuf<TPosition::VECCALC> yBufFp32_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> tanhxGm_;
    GlobalTensor<T> yGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
    bool hasTanhx_ = false;
};

}  // namespace NsMishGradV2

#endif
