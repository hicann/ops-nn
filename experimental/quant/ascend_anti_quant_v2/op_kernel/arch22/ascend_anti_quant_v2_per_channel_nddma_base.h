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
 * \file ascend_anti_quant_v2_per_channel_nddma_base.h
 * \brief ascendantiquantv2 kernel
 */

#ifndef ASCEND_ANTI_QUANT_V2_PER_CHANNEL_NDDMA_BASE_H_
#define ASCEND_ANTI_QUANT_V2_PER_CHANNEL_NDDMA_BASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "ascend_anti_quant_v2_common.h"

namespace AscendAntiQuantV2 {
using namespace AscendC;
template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
class AscendAntiQuantV2PerChannelNddmaBase : public AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode> {
public:
    __aicore__ inline AscendAntiQuantV2PerChannelNddmaBase(const AscendAntiQuantV2TilingData* tilingData)
        : tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t offset, LocalTensor<float>& sLocal,
                                           LocalTensor<float>& oLocal);
    __aicore__ inline void CopyInScale(int64_t sLen, int64_t sInOffset);
    __aicore__ inline void CopyInOffset(int64_t sLen, int64_t sInOffset);
    __aicore__ inline void ParseCoreBlocks(const AscendAntiQuantV2TilingData* tilingData, int32_t blockIdx,
                                           int64_t& blockN, int64_t& blockLen);
    __aicore__ inline void BoardCastCust(int64_t repeatSize, int64_t repeatCount, LocalTensor<float>& dstLocal,
                                         LocalTensor<float>& srcLocal, LocalTensor<int32_t>& srcPatternLocal);
    __aicore__ inline void BoardCastCustEx(int64_t repeatSize, int64_t repeatCount, int64_t totalCount,
                                           LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal,
                                           LocalTensor<int32_t>& srcPatternLocal);
    __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset);
    __aicore__ inline void Compute(int64_t nRow, int64_t dataCount, LocalTensor<float>& sLocal,
                                   LocalTensor<float>& oLocal);

private:
    constexpr static int64_t INT4_NUMS_IN_INT8_SPACE = 2;
    using xCopyDtype = std::conditional_t<IsSameType<T, int4b_t>::value, uint8_t, T>;
    constexpr static int32_t bufferNum_ = 2;
    constexpr static int32_t fpNumPerUb_ = 32 / sizeof(float);
    TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueScale_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueOffset_;
    TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;

    TBuf<TPosition::VECCALC> fp32Buf;
    TBuf<TPosition::VECCALC> fp16Buf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> maskBuf;

    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<T1> scaleGm_;
    GlobalTensor<T2> offsetGm_;
    GlobalTensor<U> yGm_;

    LocalTensor<float> floats;
    LocalTensor<float> floato;

    LocalTensor<float> floatX;
    LocalTensor<float> floats_b;
    LocalTensor<float> floato_b;
    LocalTensor<int32_t> selectMask;

    const AscendAntiQuantV2TilingData* tilingData_;
    int32_t blockIdx_ = 0;
    int64_t gmXOffset_ = 0;
    int64_t gmSOffset_ = 0;
    int64_t blockN_ = 1;
    int64_t blockLen_ = 1;
    int64_t baseLenEx_ = 1;
    int64_t shape1 = 1;
};

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::Init(GM_ADDR x, GM_ADDR scale,
                                                                                          GM_ADDR offset, GM_ADDR y,
                                                                                          TPipe* pipeIn)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(scale));
    offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(offset));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));
    ParseCoreBlocks(tilingData_, blockIdx_, blockN_, blockLen_);
    // calc n size to alloc queue
    pipeIn->InitBuffer(inQueueX_, bufferNum_, tilingData_->baseLenEx * sizeof(xCopyDtype));
    pipeIn->InitBuffer(inQueueScale_, 1, tilingData_->baseLen * sizeof(T1));
    pipeIn->InitBuffer(inQueueOffset_, 1, tilingData_->baseLen * sizeof(T2));
    pipeIn->InitBuffer(outQueueY_, bufferNum_, tilingData_->baseLenEx * sizeof(U));
    pipeIn->InitBuffer(fp16Buf, tilingData_->baseLenEx * sizeof(half));
    pipeIn->InitBuffer(fp32Buf, 3 * tilingData_->baseLenEx * sizeof(float));
    pipeIn->InitBuffer(maskBuf, tilingData_->baseLenEx * sizeof(int32_t) / 32);
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        pipeIn->InitBuffer(tmpBuf, 2 * tilingData_->baseLen * sizeof(float));
    }
    shape1 = tilingData_->dim1;
    baseLenEx_ = tilingData_->baseLenEx;
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::Process()
{
    if (blockIdx_ >= tilingData_->numCore) {
        return;
    }
    gmXOffset_ = blockIdx_ * tilingData_->blockFactor * tilingData_->dim1;
    gmSOffset_ = 0;
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        this->floats = tmp;
        this->floato = tmp[tilingData_->baseLen];
    }
    LocalTensor<float> fp32Tmp = fp32Buf.Get<float>();
    this->floatX = fp32Tmp;
    this->floats_b = fp32Tmp[tilingData_->baseLenEx];
    this->floato_b = fp32Tmp[2 * tilingData_->baseLenEx];
    int64_t lenLoopTail = tilingData_->baseN * tilingData_->dim1;
    CopyInScale(tilingData_->dim1, gmSOffset_);
    CopyInOffset(tilingData_->dim1, gmSOffset_);
    LocalTensor<T1> sLocal = inQueueScale_.DeQue<T1>();
    LocalTensor<T2> oLocal = inQueueOffset_.DeQue<T2>();
    // ld and cast for scale
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        // bf16
        AscendC::Cast<float, bfloat16_t>(floats, sLocal, AscendC::RoundMode::CAST_NONE, tilingData_->dim1);
    }
    if constexpr (SqrtMode == TPL_SQRT_MODE) {
        if constexpr (IsSameType<T1, float>::value) {
            // fp32
            AscendC::Mul(sLocal, sLocal, sLocal, tilingData_->dim1);
        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
            AscendC::Mul(floats, floats, floats, tilingData_->dim1);
        }
    }
    // ld and cast for offset
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        AscendC::Cast<float, bfloat16_t>(floato, oLocal, AscendC::RoundMode::CAST_NONE, tilingData_->dim1);
    }

    // boardcast offset and scale
    this->selectMask = maskBuf.Get<int32_t>();
    Duplicate(this->selectMask, static_cast<int32_t>(0xFFFFFFFF), 128);

    if constexpr (IsSameType<T1, float>::value) {
        if (tilingData_->lcmN <= tilingData_->baseN) {
            BoardCastCustEx(tilingData_->dim1, tilingData_->lcmN, tilingData_->baseN, this->floats_b, sLocal,
                            this->selectMask);
            BoardCastCustEx(tilingData_->dim1, tilingData_->lcmN, tilingData_->baseN, this->floato_b, oLocal,
                            this->selectMask);
        } else {
            BoardCastCust(tilingData_->dim1, tilingData_->baseN, this->floats_b, sLocal, this->selectMask);
            BoardCastCust(tilingData_->dim1, tilingData_->baseN, this->floato_b, oLocal, this->selectMask);
        }
    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
        if (tilingData_->lcmN <= tilingData_->baseN) {
            BoardCastCustEx(tilingData_->dim1, tilingData_->lcmN, tilingData_->baseN, this->floats_b, this->floats,
                            this->selectMask);
            BoardCastCustEx(tilingData_->dim1, tilingData_->lcmN, tilingData_->baseN, this->floato_b, this->floato,
                            this->selectMask);
        } else {
            BoardCastCust(tilingData_->dim1, tilingData_->baseN, this->floats_b, this->floats, this->selectMask);
            BoardCastCust(tilingData_->dim1, tilingData_->baseN, this->floato_b, this->floato, this->selectMask);
        }
    }
    CopyXAndCompute(lenLoopTail, gmXOffset_, this->floats_b, this->floato_b);
    inQueueScale_.FreeTensor(sLocal);
    inQueueOffset_.FreeTensor(oLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::BoardCastCustEx(
    int64_t repeatSize, int64_t repeatCount, int64_t totalCount, LocalTensor<float>& dstLocal,
    LocalTensor<float>& srcLocal, LocalTensor<int32_t>& srcPatternLocal)
{
    BoardCastCust(repeatSize, repeatCount, dstLocal, srcLocal, srcPatternLocal);
    int64_t nLoopNum = totalCount / repeatCount;
    int64_t nLoopTail = totalCount % repeatCount;
    for (int64_t nIdx = 1; nIdx < nLoopNum; ++nIdx) {
        Adds(dstLocal[nIdx * repeatCount * repeatSize], dstLocal, static_cast<float>(0),
             static_cast<uint32_t>(repeatCount * repeatSize));
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (nLoopTail != 0) {
        Adds(dstLocal[nLoopNum * repeatCount * repeatSize], dstLocal, static_cast<float>(0),
             static_cast<uint32_t>(nLoopTail * repeatSize));
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::BoardCastCust(
    int64_t repeatSize, int64_t repeatCount, LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal,
    LocalTensor<int32_t>& srcPatternLocal)
{
    uint64_t rsvdCnt = 0;
    GatherMask(dstLocal, srcLocal, srcPatternLocal.template ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(repeatSize), {1, static_cast<uint16_t>(repeatCount), 0, 0}, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::ParseCoreBlocks(
    const AscendAntiQuantV2TilingData* tilingData, int32_t blockIdx, int64_t& blockN, int64_t& blockLen)
{
    if (tilingData->blockAxis == 0) {
        if (blockIdx == tilingData->numCore - 1) {
            blockN = tilingData->blockTailFactor;
        } else {
            blockN = tilingData->blockFactor;
        }
        blockLen = tilingData->dim1;
    } else if (tilingData->blockAxis == 1) {
        blockN = tilingData->dim0;
        if (blockIdx == tilingData->numCore - 1) {
            blockLen = tilingData->blockTailFactor;
        } else {
            blockLen = tilingData->blockFactor;
        }
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::CopyInScale(int64_t sLen,
                                                                                                 int64_t sInOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sLen * sizeof(T1);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
    LocalTensor<T1> sLocal = inQueueScale_.AllocTensor<T1>();
    DataCopyPad(sLocal, scaleGm_[sInOffset], copyParams, {false, 0, 0, 0});
    inQueueScale_.EnQue(sLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::CopyInOffset(int64_t sLen,
                                                                                                  int64_t sInOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sLen * sizeof(T2);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
    LocalTensor<T2> oLocal = inQueueOffset_.AllocTensor<T2>();
    DataCopyPad(oLocal, offsetGm_[sInOffset], copyParams, {false, 0, 0, 0});
    inQueueOffset_.EnQue(oLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::CopyXAndCompute(
    int64_t dataCount, int64_t offset, LocalTensor<float>& sFPLocal, LocalTensor<float>& oFpLocal)
{
    int64_t nLoopNum = blockN_ / tilingData_->baseN;
    int64_t nLoopTail = blockN_ % tilingData_->baseN;
    int64_t xOffset = offset;
    for (int64_t nIdx = 0; nIdx < nLoopNum; ++nIdx) {
        xOffset = offset + nIdx * dataCount;
        CopyInX(1, dataCount, xOffset);
        Compute(1, dataCount, sFPLocal, oFpLocal);
        CopyOutY(1, dataCount, xOffset);
    }
    if (nLoopTail != 0) {
        xOffset = offset + nLoopNum * dataCount;
        CopyInX(1, nLoopTail * tilingData_->dim1, xOffset);
        Compute(1, nLoopTail * tilingData_->dim1, sFPLocal, oFpLocal);
        CopyOutY(1, nLoopTail * tilingData_->dim1, xOffset);
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::CopyInX(int64_t xN, int64_t xLen,
                                                                                             int64_t xInOffset)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        xInOffset = xInOffset >> 1;
    }

    LocalTensor<uint8_t> xLocal = inQueueX_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};

    int64_t xLenReal = xLen;
    if constexpr (IsSameType<T, int4b_t>::value) {
        xLenReal = xLenReal / INT4_NUMS_IN_INT8_SPACE;
        copyParams.blockLen = xLenReal * sizeof(xCopyDtype);
    } else {
        copyParams.blockLen = xLenReal * sizeof(T);
    }
    copyParams.blockCount = xN;
    copyParams.srcStride = 0;
    if (baseLenEx_ > xLenReal) {
        copyParams.dstStride = (baseLenEx_ - xLenReal) * sizeof(xCopyDtype) / 32;
    } else {
        copyParams.dstStride = 0;
    }
    DataCopyPad<uint8_t>(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::Compute(
    int64_t nRow, int64_t dataCount, LocalTensor<float>& sFPLocal, LocalTensor<float>& oFpLocal)
{
    LocalTensor<xCopyDtype> xLocal = inQueueX_.DeQue<xCopyDtype>();
    LocalTensor<U> outLocal = outQueueY_.AllocTensor<U>();

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
    AscendC::Add(floatX, floatX, oFpLocal, dataCount);
    AscendC::Mul(floatX, floatX, sFPLocal, dataCount);
    // cast and sd for y
    AscendC::Cast(outLocal, floatX, AscendC::RoundMode::CAST_RINT, dataCount);
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaBase<T, T1, T2, U, SqrtMode>::CopyOutY(int64_t yN, int64_t yLen,
                                                                                              int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.DeQue<U>();
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_->dim1, tilingData_->baseLen, yN, yLen, copyParams);
    DataCopyPad<U>(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}
} // namespace AscendAntiQuantV2
#endif
