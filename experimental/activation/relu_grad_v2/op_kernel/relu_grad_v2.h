/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_ACTIVATION_RELU_GRAD_V2_H_
#define EXPERIMENTAL_ACTIVATION_RELU_GRAD_V2_H_

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_grad_v2_tiling_data.h"
#include "relu_grad_v2_tiling_key.h"

namespace NsReluGradV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t DATA_COPY_ALIGN_BYTES = 32;

__aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
{
    if (align == 0) {
        return value;
    }
    if (align < 0) {
        return value;
    }
    int64_t remainder = value % align;
    if (remainder == 0) {
        return value;
    }
    return value + align - remainder;
}

template <typename T, typename Derived>
class KernelReluGradBase {
public:
    __aicore__ inline KernelReluGradBase() = default;

    __aicore__ inline void Init(GM_ADDR gradients, GM_ADDR features, GM_ADDR backprops,
                                const ReluGradV2TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        InitGlobalTensors(gradients, features, backprops, tilingData);
        tileLength_ = tilingData->tileLength;
        pipe_->InitBuffer(inQueueGradients_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_->InitBuffer(inQueueFeatures_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_->InitBuffer(outQueueBackprops_, BUFFER_NUM, tileLength_ * sizeof(T));
        static_cast<Derived*>(this)->InitExtraBuffers();
    }

    __aicore__ inline void Process()
    {
        int64_t tileNum = (blockLength_ + tileLength_ - 1) / tileLength_;
        if (tileNum == 0) {
            return;
        }

        Derived* derived = static_cast<Derived*>(this);
        int64_t alignNum = derived->GetAlignNum();
        for (int64_t i = 0; i < tileNum; ++i) {
            int64_t validLength = tileLength_;
            if (i == tileNum - 1) {
                validLength = blockLength_ - (tileNum - 1) * tileLength_;
            }
            int64_t paddedLength = AlignUp(validLength, alignNum);
            CopyIn(i, validLength, paddedLength);
            derived->Compute(derived->GetComputeLength(validLength, paddedLength));
            CopyOut(i, validLength);
        }
    }

    __aicore__ inline int64_t GetAlignNum() const { return DATA_COPY_ALIGN_BYTES / static_cast<int64_t>(sizeof(T)); }

    __aicore__ inline int64_t GetComputeLength(int64_t validLength, int64_t paddedLength) const
    {
        (void)paddedLength;
        return validLength;
    }

protected:
    __aicore__ inline void CopyIn(int64_t progress, int64_t validLength, int64_t paddedLength)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(paddedLength - validLength), static_cast<T>(0)};

        LocalTensor<T> gradientsLocal = inQueueGradients_.AllocTensor<T>();
        LocalTensor<T> featuresLocal = inQueueFeatures_.AllocTensor<T>();

        DataCopyPad(gradientsLocal, gradientsGm_[progress * tileLength_], copyParams, padParams);
        DataCopyPad(featuresLocal, featuresGm_[progress * tileLength_], copyParams, padParams);

        inQueueGradients_.EnQue(gradientsLocal);
        inQueueFeatures_.EnQue(featuresLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t validLength)
    {
        LocalTensor<T> backpropsLocal = outQueueBackprops_.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(backpropsGm_[progress * tileLength_], backpropsLocal, copyParams);
        outQueueBackprops_.FreeTensor(backpropsLocal);
    }

    __aicore__ inline void FinishCompute(LocalTensor<T> backpropsLocal, LocalTensor<T> gradientsLocal,
                                         LocalTensor<T> featuresLocal)
    {
        outQueueBackprops_.EnQue<T>(backpropsLocal);
        inQueueGradients_.FreeTensor(gradientsLocal);
        inQueueFeatures_.FreeTensor(featuresLocal);
    }

protected:
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueGradients_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueFeatures_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueBackprops_;

private:
    __aicore__ inline void InitGlobalTensors(GM_ADDR gradients, GM_ADDR features, GM_ADDR backprops,
                                             const ReluGradV2TilingData* tilingData)
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t offset = 0;
        int64_t blockSize = tilingData->tailLength;
        if (blockIdx < tilingData->formerNum) {
            offset = tilingData->formerLength * blockIdx;
            blockSize = tilingData->formerLength;
        } else {
            offset = tilingData->formerLength * tilingData->formerNum;
        }

        blockLength_ = blockSize;
        gradientsGm_.SetGlobalBuffer((__gm__ T*)gradients + offset, blockSize);
        featuresGm_.SetGlobalBuffer((__gm__ T*)features + offset, blockSize);
        backpropsGm_.SetGlobalBuffer((__gm__ T*)backprops + offset, blockSize);
    }

protected:
    TPipe* pipe_ = nullptr;
    GlobalTensor<T> gradientsGm_;
    GlobalTensor<T> featuresGm_;
    GlobalTensor<T> backpropsGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
};

template <typename T>
class KernelReluGradScalar : public KernelReluGradBase<T, KernelReluGradScalar<T>> {
public:
    __aicore__ inline KernelReluGradScalar() = default;

    __aicore__ inline void InitExtraBuffers() {}

    __aicore__ inline void Compute(int64_t validLength)
    {
        LocalTensor<T> gradientsLocal = this->inQueueGradients_.template DeQue<T>();
        LocalTensor<T> featuresLocal = this->inQueueFeatures_.template DeQue<T>();
        LocalTensor<T> backpropsLocal = this->outQueueBackprops_.template AllocTensor<T>();

        for (int64_t idx = 0; idx < validLength; ++idx) {
            bool keepGradient = false;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
                float featureValue = static_cast<float>(featuresLocal.GetValue(idx));
                keepGradient = featureValue > 0.0f || featureValue != featureValue;
            } else {
                T featureValue = featuresLocal.GetValue(idx);
                keepGradient = featureValue > static_cast<T>(0);
            }
            backpropsLocal.SetValue(idx, keepGradient ? gradientsLocal.GetValue(idx) : static_cast<T>(0));
        }

        this->FinishCompute(backpropsLocal, gradientsLocal, featuresLocal);
    }
};

template <typename T>
class KernelReluGradCastScalar : public KernelReluGradBase<T, KernelReluGradCastScalar<T>> {
public:
    __aicore__ inline KernelReluGradCastScalar() = default;

    __aicore__ inline void InitExtraBuffers()
    {
        this->pipe_->InitBuffer(gradientsFp32Buf_, this->tileLength_ * sizeof(float));
        this->pipe_->InitBuffer(featuresFp32Buf_, this->tileLength_ * sizeof(float));
        this->pipe_->InitBuffer(backpropsFp32Buf_, this->tileLength_ * sizeof(float));
    }

    __aicore__ inline void Compute(int64_t validLength)
    {
        LocalTensor<T> gradientsLocal = this->inQueueGradients_.template DeQue<T>();
        LocalTensor<T> featuresLocal = this->inQueueFeatures_.template DeQue<T>();
        LocalTensor<T> backpropsLocal = this->outQueueBackprops_.template AllocTensor<T>();
        LocalTensor<float> gradientsFp32 = gradientsFp32Buf_.Get<float>();
        LocalTensor<float> featuresFp32 = featuresFp32Buf_.Get<float>();
        LocalTensor<float> backpropsFp32 = backpropsFp32Buf_.Get<float>();

        Cast(gradientsFp32, gradientsLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(validLength));
        Cast(featuresFp32, featuresLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(validLength));
        PipeBarrier<PIPE_V>();

        for (int64_t idx = 0; idx < validLength; ++idx) {
            float featureValue = featuresFp32.GetValue(idx);
            backpropsFp32.SetValue(
                idx, featureValue > 0.0f || featureValue != featureValue ? gradientsFp32.GetValue(idx) : 0.0f);
        }

        Cast(backpropsLocal, backpropsFp32, RoundMode::CAST_ROUND, static_cast<uint32_t>(validLength));
        this->FinishCompute(backpropsLocal, gradientsLocal, featuresLocal);
    }

private:
    TBuf<TPosition::VECCALC> gradientsFp32Buf_;
    TBuf<TPosition::VECCALC> featuresFp32Buf_;
    TBuf<TPosition::VECCALC> backpropsFp32Buf_;
};

} // namespace NsReluGradV2

#endif
