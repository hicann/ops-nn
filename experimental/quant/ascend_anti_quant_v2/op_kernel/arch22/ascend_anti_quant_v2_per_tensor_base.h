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
 * \file ascend_anti_quant_v2_per_tensor_base.h
 * \brief ascendantiquantv2 kernel
 */

#ifndef ASCEND_ANTI_QUANT_V2_PER_TENSOR_BASE_H_
#define ASCEND_ANTI_QUANT_V2_PER_TENSOR_BASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "ascend_anti_quant_v2_common.h"

namespace AscendAntiQuantV2 {
using namespace AscendC;
template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
class AscendAntiQuantV2PerTensorBase : public AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode> {
public:
    __aicore__ inline AscendAntiQuantV2PerTensorBase(const AscendAntiQuantV2TilingData* tilingData)
        : tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t offset, float sValue, float oValue);
    __aicore__ inline void CopyInScale();
    __aicore__ inline void CopyInOffset();
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);
    __aicore__ inline void Compute(int64_t dataCount, float sValue, float oValue);

private:
    using xCopyDtype = std::conditional_t<IsSameType<T, int4b_t>::value, uint8_t, T>;
    constexpr static int32_t bufferNum_ = 2;

    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<T1> scaleGm_;
    GlobalTensor<T2> offsetGm_;
    GlobalTensor<U> yGm_;

    TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueScale_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueOffset_;
    TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;

    TBuf<TPosition::VECCALC> fp32Buf;
    TBuf<TPosition::VECCALC> fp16Buf;
    TBuf<TPosition::VECCALC> tmpBuf;

    const AscendAntiQuantV2TilingData* tilingData_;
    int32_t blockIdx_ = 0;
    int64_t gmXOffset_ = 0;
    int64_t blockN_ = 1;
    int64_t blockLen_ = 1;
};

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerTensorBase<T, T1, T2, U, SqrtMode>::Init(GM_ADDR x, GM_ADDR scale,
                                                                                    GM_ADDR offset, GM_ADDR y,
                                                                                    TPipe* pipeIn)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(scale));
    offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(offset));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));

    blockN_ = tilingData_->dim0;
    if (blockIdx_ == tilingData_->numCore - 1) {
        blockLen_ = tilingData_->blockTailFactor;
    } else {
        blockLen_ = tilingData_->blockFactor;
    }

    // calc n size to alloc queue
    pipeIn->InitBuffer(inQueueX_, bufferNum_, tilingData_->baseN * tilingData_->baseLen * sizeof(xCopyDtype));
    pipeIn->InitBuffer(inQueueScale_, bufferNum_, this->BLOCK_SIZE);
    pipeIn->InitBuffer(inQueueOffset_, bufferNum_, this->BLOCK_SIZE);

    pipeIn->InitBuffer(outQueueY_, bufferNum_, tilingData_->baseN * tilingData_->baseLen * sizeof(U));

    pipeIn->InitBuffer(fp32Buf, tilingData_->baseN * tilingData_->baseLen * sizeof(float));
    pipeIn->InitBuffer(fp16Buf, tilingData_->baseN * tilingData_->baseLen * sizeof(half));
    pipeIn->InitBuffer(tmpBuf, 2 * 256);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerTensorBase<T, T1, T2, U, SqrtMode>::Process()
{
    if (blockIdx_ >= tilingData_->numCore) {
        return;
    }
    gmXOffset_ = blockIdx_ * tilingData_->blockFactor;
    // main loop with column, for scale and offset only need copy once
    int64_t lenLoopNum = blockLen_ / tilingData_->baseLen;
    int64_t lenLoopTail = blockLen_ % tilingData_->baseLen;
    CopyInScale();
    CopyInOffset();
    LocalTensor<T1> sLocal = inQueueScale_.DeQue<T1>();
    LocalTensor<T2> oLocal = inQueueOffset_.DeQue<T2>();

    LocalTensor<float> tmp = tmpBuf.Get<float>();
    LocalTensor<float> floats = tmp;
    LocalTensor<float> floato = tmp[64];
    // ld and cast for scale
    if constexpr (IsSameType<T1, float>::value) {
        // fp32
        AscendC::Adds(floats, sLocal, (float)(0.0), 1);
    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
        // bf16
        AscendC::Cast<float, bfloat16_t>(floats, sLocal, AscendC::RoundMode::CAST_NONE, 1);
    }
    // ld and cast for offset
    if constexpr (IsSameType<T1, float>::value) {
        // fp32
        AscendC::Adds(floato, oLocal, (float)(0.0), 1);
    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
        // bf16
        AscendC::Cast<float, bfloat16_t>(floato, oLocal, AscendC::RoundMode::CAST_NONE, 1);
    }
    // compute
    if constexpr (SqrtMode == TPL_SQRT_MODE) {
        AscendC::Mul(floats, floats, floats, 1);
    }

    float sValue = floats.GetValue(0);
    float oValue = floato.GetValue(0);

    for (int64_t i = 0; i < lenLoopNum; ++i) {
        CopyXAndCompute(tilingData_->baseLen, gmXOffset_ + i * tilingData_->baseLen, sValue, oValue);
    }
    if (lenLoopTail != 0) {
        CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_->baseLen, sValue, oValue);
    }
    inQueueScale_.FreeTensor(sLocal);
    inQueueOffset_.FreeTensor(oLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerTensorBase<T, T1, T2, U, SqrtMode>::CopyInScale()
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(T1);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
    LocalTensor<T1> sLocal = inQueueScale_.AllocTensor<T1>();
    DataCopyPad(sLocal, scaleGm_, copyParams, {false, 0, 0, 0});
    inQueueScale_.EnQue(sLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerTensorBase<T, T1, T2, U, SqrtMode>::CopyInOffset()
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(T2);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
    LocalTensor<T2> oLocal = inQueueOffset_.AllocTensor<T2>();
    DataCopyPad(oLocal, offsetGm_, copyParams, {false, 0, 0, 0});
    inQueueOffset_.EnQue(oLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerTensorBase<T, T1, T2, U, SqrtMode>::CopyXAndCompute(int64_t dataCount,
                                                                                               int64_t offset,
                                                                                               float sValue,
                                                                                               float oValue)
{
    CopyInX(dataCount, offset);
    Compute(dataCount, sValue, oValue);
    CopyOutY(dataCount, offset);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerTensorBase<T, T1, T2, U, SqrtMode>::CopyInX(int64_t xLen, int64_t xInOffset)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        xInOffset = xInOffset >> 1;
    }

    LocalTensor<uint8_t> xLocal = inQueueX_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    this->GetXInCopyParams(tilingData_->dim1, tilingData_->baseLen, tilingData_->baseN, xLen, copyParams);
    DataCopyPad<uint8_t>(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerTensorBase<T, T1, T2, U, SqrtMode>::Compute(int64_t dataCount, float sValue,
                                                                                       float oValue)
{
    LocalTensor<xCopyDtype> xLocal = inQueueX_.DeQue<xCopyDtype>();
    LocalTensor<U> outLocal = outQueueY_.AllocTensor<U>();

    LocalTensor<float> floatX = fp32Buf.Get<float>();
    LocalTensor<half> halfX = fp16Buf.Get<half>();

    if constexpr (IsSameType<T, int8_t>::value) {
        // int8
        AscendC::Cast(halfX, xLocal, AscendC::RoundMode::CAST_NONE, dataCount);
        AscendC::Cast(floatX, halfX, AscendC::RoundMode::CAST_NONE, dataCount);
    } else if constexpr (IsSameType<T, int4b_t>::value) {
        // int4
        LocalTensor<int4b_t> int4_xLocal = xLocal.template ReinterpretCast<int4b_t>();
        AscendC::Cast<half, AscendC::int4b_t>(halfX, int4_xLocal, AscendC::RoundMode::CAST_NONE, dataCount);
        AscendC::Cast<float, half>(floatX, halfX, AscendC::RoundMode::CAST_NONE, dataCount);
    }

    AscendC::Adds(floatX, floatX, oValue, dataCount);
    AscendC::Muls(floatX, floatX, sValue, dataCount);

    // cast and sd for y
    if constexpr (IsSameType<U, half>::value) {
        // fp16
        AscendC::Cast<half, float>(outLocal, floatX, AscendC::RoundMode::CAST_RINT, dataCount);
    } else if constexpr (IsSameType<U, bfloat16_t>::value) {
        // bf16
        AscendC::Cast<bfloat16_t, float>(outLocal, floatX, AscendC::RoundMode::CAST_RINT, dataCount);
    }

    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerTensorBase<T, T1, T2, U, SqrtMode>::CopyOutY(int64_t yLen,
                                                                                        int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.DeQue<U>();
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_->dim1, tilingData_->baseLen, tilingData_->baseN, yLen, copyParams);
    DataCopyPad<U>(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}
} // namespace AscendAntiQuantV2
#endif
