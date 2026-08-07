/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_ACTIVATION_RELU_GRAD_V4_H_
#define EXPERIMENTAL_ACTIVATION_RELU_GRAD_V4_H_

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_grad_v4_tiling_data.h"
#include "relu_grad_v4_tiling_key.h"

namespace NsReluGradV4 {

using namespace AscendC;

// ReluGradV4 semantics: backprops = mask ? gradients : 0   (mask is uint8 {0,1}).
// Inputs have DIFFERENT dtypes: gradients = compute dtype T, mask = uint8 always.
// mask==0 must give exactly 0 even for inf/nan gradients -> use Select, not grad*mask.

constexpr int32_t BUFFER_NUM = 2;

// Base: gradients<T> + mask<uint8_t> -> backprops<T>.
template <typename T, typename Derived>
class KernelReluGradBase {
public:
    __aicore__ inline KernelReluGradBase() = default;

    __aicore__ inline void Init(GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops,
                                const ReluGradV4TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        InitGlobalTensors(gradients, mask, backprops, tilingData);
        tileLength_ = tilingData->tileLength;
        pipe_->InitBuffer(inQueueGradients_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_->InitBuffer(inQueueMask_, BUFFER_NUM, tileLength_ * sizeof(uint8_t));
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
        for (int64_t i = 0; i < tileNum; ++i) {
            int64_t validLength = tileLength_;
            if (i == tileNum - 1) {
                validLength = blockLength_ - (tileNum - 1) * tileLength_;
            }
            CopyIn(i, validLength);
            derived->Compute(tileLength_);
            CopyOut(i, validLength);
        }
    }

protected:
    __aicore__ inline void CopyIn(int64_t progress, int64_t validLength)
    {
        LocalTensor<T> gradientsLocal = inQueueGradients_.template AllocTensor<T>();
        LocalTensor<uint8_t> maskLocal = inQueueMask_.template AllocTensor<uint8_t>();

        DataCopyExtParams gParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> gPad{true, 0, 0, static_cast<T>(0)};
        DataCopyPad(gradientsLocal, gradientsGm_[progress * tileLength_], gParams, gPad);

        DataCopyExtParams mParams{1, static_cast<uint32_t>(validLength * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> mPad{true, 0, 0, static_cast<uint8_t>(0)};
        DataCopyPad(maskLocal, maskGm_[progress * tileLength_], mParams, mPad);

        inQueueGradients_.EnQue(gradientsLocal);
        inQueueMask_.EnQue(maskLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t validLength)
    {
        LocalTensor<T> backpropsLocal = outQueueBackprops_.template DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(backpropsGm_[progress * tileLength_], backpropsLocal, copyParams);
        outQueueBackprops_.FreeTensor(backpropsLocal);
    }

    __aicore__ inline void FinishCompute(LocalTensor<T> backpropsLocal, LocalTensor<T> gradientsLocal,
                                         LocalTensor<uint8_t> maskLocal)
    {
        outQueueBackprops_.template EnQue<T>(backpropsLocal);
        inQueueGradients_.FreeTensor(gradientsLocal);
        inQueueMask_.FreeTensor(maskLocal);
    }

    // Compare(maskFp > 0) into a select bitmask. mask values are exactly {0,1} so fp32 compare is exact.
    __aicore__ inline void MaskToBits(LocalTensor<uint8_t> maskLocal, LocalTensor<half> maskHalf,
                                      LocalTensor<float> maskFp, LocalTensor<float> zeroFp, LocalTensor<uint8_t> bits,
                                      int32_t n)
    {
        Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, n); // uint8 -> half
        Cast(maskFp, maskHalf, RoundMode::CAST_NONE, n);    // half -> fp32
        Compare(bits, maskFp, zeroFp, CMPMODE::GT, n);      // {0,1} -> exact bitmask
    }

protected:
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueGradients_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueMask_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueBackprops_;

private:
    __aicore__ inline void InitGlobalTensors(GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops,
                                             const ReluGradV4TilingData* tilingData)
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
        maskGm_.SetGlobalBuffer((__gm__ uint8_t*)mask + offset, blockSize);
        backpropsGm_.SetGlobalBuffer((__gm__ T*)backprops + offset, blockSize);
    }

protected:
    TPipe* pipe_ = nullptr;
    GlobalTensor<T> gradientsGm_;
    GlobalTensor<uint8_t> maskGm_;
    GlobalTensor<T> backpropsGm_;
    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
};

// float / half / bfloat16: Select directly in T (gradient native, no cast). out = bits ? grad : 0.
template <typename T>
class KernelReluGradSelect : public KernelReluGradBase<T, KernelReluGradSelect<T>> {
public:
    __aicore__ inline KernelReluGradSelect() = default;

    __aicore__ inline void InitExtraBuffers()
    {
        int64_t n = this->tileLength_;
        this->pipe_->InitBuffer(maskHalfBuf_, n * sizeof(half));
        this->pipe_->InitBuffer(maskFpBuf_, n * sizeof(float));
        this->pipe_->InitBuffer(zeroFpBuf_, n * sizeof(float));
        this->pipe_->InitBuffer(zeroTBuf_, n * sizeof(T));
        this->pipe_->InitBuffer(bitsBuf_, n * sizeof(uint8_t));
        Duplicate(zeroFpBuf_.template Get<float>(), 0.0f, static_cast<int32_t>(n));
        Duplicate(zeroTBuf_.template Get<T>(), static_cast<T>(0), static_cast<int32_t>(n));
    }

    __aicore__ inline void Compute(int64_t computeLength)
    {
        LocalTensor<T> gradientsLocal = this->inQueueGradients_.template DeQue<T>();
        LocalTensor<uint8_t> maskLocal = this->inQueueMask_.template DeQue<uint8_t>();
        LocalTensor<T> backpropsLocal = this->outQueueBackprops_.template AllocTensor<T>();
        LocalTensor<half> maskHalf = maskHalfBuf_.template Get<half>();
        LocalTensor<float> maskFp = maskFpBuf_.template Get<float>();
        LocalTensor<float> zeroFp = zeroFpBuf_.template Get<float>();
        LocalTensor<T> zeroT = zeroTBuf_.template Get<T>();
        LocalTensor<uint8_t> bits = bitsBuf_.template Get<uint8_t>();
        int32_t n = static_cast<int32_t>(computeLength);

        this->MaskToBits(maskLocal, maskHalf, maskFp, zeroFp, bits, n);
        Select(backpropsLocal, bits, gradientsLocal, zeroT, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        this->FinishCompute(backpropsLocal, gradientsLocal, maskLocal);
    }

private:
    TBuf<TPosition::VECCALC> maskHalfBuf_;
    TBuf<TPosition::VECCALC> maskFpBuf_;
    TBuf<TPosition::VECCALC> zeroFpBuf_;
    TBuf<TPosition::VECCALC> zeroTBuf_;
    TBuf<TPosition::VECCALC> bitsBuf_;
};

// bfloat16: bf16 has no native vsel and its range exceeds half -> cast grad bf16->fp32, Select in
// fp32, cast back to bf16. out = bits ? grad : 0.
template <typename T>
class KernelReluGradCastSelect : public KernelReluGradBase<T, KernelReluGradCastSelect<T>> {
public:
    __aicore__ inline KernelReluGradCastSelect() = default;

    __aicore__ inline void InitExtraBuffers()
    {
        int64_t n = this->tileLength_;
        this->pipe_->InitBuffer(gradFpBuf_, n * sizeof(float));
        this->pipe_->InitBuffer(outFpBuf_, n * sizeof(float));
        this->pipe_->InitBuffer(maskHalfBuf_, n * sizeof(half));
        this->pipe_->InitBuffer(maskFpBuf_, n * sizeof(float));
        this->pipe_->InitBuffer(zeroFpBuf_, n * sizeof(float));
        this->pipe_->InitBuffer(bitsBuf_, n * sizeof(uint8_t));
        Duplicate(zeroFpBuf_.template Get<float>(), 0.0f, static_cast<int32_t>(n));
    }

    __aicore__ inline void Compute(int64_t computeLength)
    {
        LocalTensor<T> gradientsLocal = this->inQueueGradients_.template DeQue<T>();
        LocalTensor<uint8_t> maskLocal = this->inQueueMask_.template DeQue<uint8_t>();
        LocalTensor<T> backpropsLocal = this->outQueueBackprops_.template AllocTensor<T>();
        LocalTensor<float> gradFp = gradFpBuf_.template Get<float>();
        LocalTensor<float> outFp = outFpBuf_.template Get<float>();
        LocalTensor<half> maskHalf = maskHalfBuf_.template Get<half>();
        LocalTensor<float> maskFp = maskFpBuf_.template Get<float>();
        LocalTensor<float> zeroFp = zeroFpBuf_.template Get<float>();
        LocalTensor<uint8_t> bits = bitsBuf_.template Get<uint8_t>();
        int32_t n = static_cast<int32_t>(computeLength);

        Cast(gradFp, gradientsLocal, RoundMode::CAST_NONE, n); // bf16 -> fp32
        this->MaskToBits(maskLocal, maskHalf, maskFp, zeroFp, bits, n);
        Select(outFp, bits, gradFp, zeroFp, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        Cast(backpropsLocal, outFp, RoundMode::CAST_RINT, n); // fp32 -> bf16
        this->FinishCompute(backpropsLocal, gradientsLocal, maskLocal);
    }

private:
    TBuf<TPosition::VECCALC> gradFpBuf_;
    TBuf<TPosition::VECCALC> outFpBuf_;
    TBuf<TPosition::VECCALC> maskHalfBuf_;
    TBuf<TPosition::VECCALC> maskFpBuf_;
    TBuf<TPosition::VECCALC> zeroFpBuf_;
    TBuf<TPosition::VECCALC> bitsBuf_;
};

// int32: cmpsel unreliable on int32 -> multiply by {0,1} int32 (exact). out = grad * mask.
template <typename T>
class KernelReluGradInt32 : public KernelReluGradBase<T, KernelReluGradInt32<T>> {
public:
    __aicore__ inline KernelReluGradInt32() = default;

    __aicore__ inline void InitExtraBuffers()
    {
        int64_t n = this->tileLength_;
        this->pipe_->InitBuffer(maskHalfBuf_, n * sizeof(half));
        this->pipe_->InitBuffer(maskFpBuf_, n * sizeof(float));
        this->pipe_->InitBuffer(maskIntBuf_, n * sizeof(int32_t));
    }

    __aicore__ inline void Compute(int64_t computeLength)
    {
        LocalTensor<T> gradientsLocal = this->inQueueGradients_.template DeQue<T>();
        LocalTensor<uint8_t> maskLocal = this->inQueueMask_.template DeQue<uint8_t>();
        LocalTensor<T> backpropsLocal = this->outQueueBackprops_.template AllocTensor<T>();
        LocalTensor<half> maskHalf = maskHalfBuf_.template Get<half>();
        LocalTensor<float> maskFp = maskFpBuf_.template Get<float>();
        LocalTensor<int32_t> maskInt = maskIntBuf_.template Get<int32_t>();
        int32_t n = static_cast<int32_t>(computeLength);

        Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, n); // uint8 -> half
        Cast(maskFp, maskHalf, RoundMode::CAST_NONE, n);    // half -> fp32
        Cast(maskInt, maskFp, RoundMode::CAST_RINT, n);     // fp32 -> int32 {0,1}
        Mul(backpropsLocal, gradientsLocal, maskInt, n);    // grad * {0,1}, exact
        this->FinishCompute(backpropsLocal, gradientsLocal, maskLocal);
    }

private:
    TBuf<TPosition::VECCALC> maskHalfBuf_;
    TBuf<TPosition::VECCALC> maskFpBuf_;
    TBuf<TPosition::VECCALC> maskIntBuf_;
};

// int8 / uint8: 1-byte lacks Select -> cast grad int8->half, Select in half (bits from uint8 mask),
// cast back. grad in [-128,127] and results are exact in half.
template <typename T>
class KernelReluGradInt8 : public KernelReluGradBase<T, KernelReluGradInt8<T>> {
public:
    __aicore__ inline KernelReluGradInt8() = default;

    __aicore__ inline void InitExtraBuffers()
    {
        int64_t n = this->tileLength_;
        this->pipe_->InitBuffer(gradHalfBuf_, n * sizeof(half));
        this->pipe_->InitBuffer(outHalfBuf_, n * sizeof(half));
        this->pipe_->InitBuffer(maskHalfBuf_, n * sizeof(half));
        this->pipe_->InitBuffer(maskFpBuf_, n * sizeof(float));
        this->pipe_->InitBuffer(zeroFpBuf_, n * sizeof(float));
        this->pipe_->InitBuffer(zeroHalfBuf_, n * sizeof(half));
        this->pipe_->InitBuffer(bitsBuf_, n * sizeof(uint8_t));
        Duplicate(zeroFpBuf_.template Get<float>(), 0.0f, static_cast<int32_t>(n));
        Duplicate(zeroHalfBuf_.template Get<half>(), static_cast<half>(0), static_cast<int32_t>(n));
    }

    __aicore__ inline void Compute(int64_t computeLength)
    {
        LocalTensor<T> gradientsLocal = this->inQueueGradients_.template DeQue<T>();
        LocalTensor<uint8_t> maskLocal = this->inQueueMask_.template DeQue<uint8_t>();
        LocalTensor<T> backpropsLocal = this->outQueueBackprops_.template AllocTensor<T>();
        LocalTensor<half> gradHalf = gradHalfBuf_.template Get<half>();
        LocalTensor<half> outHalf = outHalfBuf_.template Get<half>();
        LocalTensor<half> maskHalf = maskHalfBuf_.template Get<half>();
        LocalTensor<float> maskFp = maskFpBuf_.template Get<float>();
        LocalTensor<float> zeroFp = zeroFpBuf_.template Get<float>();
        LocalTensor<half> zeroHalf = zeroHalfBuf_.template Get<half>();
        LocalTensor<uint8_t> bits = bitsBuf_.template Get<uint8_t>();
        int32_t n = static_cast<int32_t>(computeLength);

        Cast(gradHalf, gradientsLocal, RoundMode::CAST_NONE, n); // int8/uint8 -> half (exact)
        this->MaskToBits(maskLocal, maskHalf, maskFp, zeroFp, bits, n);
        Select(outHalf, bits, gradHalf, zeroHalf, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        Cast(backpropsLocal, outHalf, RoundMode::CAST_RINT, n); // half -> int8/uint8 (exact)
        this->FinishCompute(backpropsLocal, gradientsLocal, maskLocal);
    }

private:
    TBuf<TPosition::VECCALC> gradHalfBuf_;
    TBuf<TPosition::VECCALC> outHalfBuf_;
    TBuf<TPosition::VECCALC> maskHalfBuf_;
    TBuf<TPosition::VECCALC> maskFpBuf_;
    TBuf<TPosition::VECCALC> zeroFpBuf_;
    TBuf<TPosition::VECCALC> zeroHalfBuf_;
    TBuf<TPosition::VECCALC> bitsBuf_;
};

} // namespace NsReluGradV4

#endif
