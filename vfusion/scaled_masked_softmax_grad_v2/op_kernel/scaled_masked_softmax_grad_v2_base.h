/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scaled_masked_softmax_grad_v2_base.h
 * \brief
 */

#ifndef SCALED_MASKED_SOFTMAX_GRAD_V2_BASE_H
#define SCALED_MASKED_SOFTMAX_GRAD_V2_BASE_H

#include "kernel_operator.h"

namespace ScaledMaskedSoftmaxGradV2 {
using namespace AscendC;

// 行数信息
struct LineInfo {
    uint64_t currentLine;
    uint64_t currentBatch;
    uint64_t currentChannel;
    uint64_t currentLineInMask;
    uint64_t currentLineInBatch;
    uint64_t currentLineInChannel;
};

enum class CopyType : uint8_t
{
    INPUT = 0,
    OUTPUT = 1,
    MASK_INPUT = 2
};

struct GmTensor {
    const GM_ADDR yGrad = nullptr;
    const GM_ADDR y = nullptr;
    const GM_ADDR mask = nullptr;
    const GM_ADDR xGrad = nullptr;
};

// 定义常量
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t SIZE_4 = 4;
constexpr uint64_t SIZE_2 = 2;
constexpr uint64_t BLK_STRIDE = 1;
constexpr uint64_t REP_STRIDE = 8;
constexpr uint64_t REP_LEN = 256;
constexpr uint64_t BLK_LEN = 32;
constexpr uint64_t MASK_LEN_B32 = 64;
constexpr uint64_t BLK_REP_TIMES = 8;
constexpr uint64_t ALIGNED_NUM = 64;
constexpr float MASK_VALUE = 0.0;
constexpr float DEFAULT_SCALE = 1.0;

template <typename T>
class ScaledMaskedSoftmaxGradV2Base
{
public:
    __aicore__ inline ScaledMaskedSoftmaxGradV2Base()
    {}
    __aicore__ inline void Init(
        const GmTensor* gmTensor, const ScaledMaskedSoftmaxGradV2TilingData& tilingData, TPipe* pipeIn);
    __aicore__ inline uint64_t CeilDiv(const uint64_t dividend, const uint64_t divisor);

protected:
    __aicore__ inline void SetDataCopyParams(const uint64_t& currentLineNum);
    __aicore__ inline void CopyIn(LocalTensor<T>& yGradLocal, LocalTensor<T>& yLocal, const uint64_t& offset);
    __aicore__ inline void CopyInMask(LocalTensor<bool>& maskLocal, const uint64_t& currentLoop);
    __aicore__ inline void CopyOut(LocalTensor<T>& outLocal, const uint64_t& offset);
    __aicore__ inline void CopyInMaskLines(
        LocalTensor<bool>& maskLocal, uint64_t& lineCnt, const uint64_t& curLineNum, const uint64_t& currentLineInMask);
    __aicore__ inline void CopyInMaskBlock(
        LocalTensor<bool>& maskLocal, uint64_t& lineCnt, const uint64_t& curLineNum, uint64_t repeatTimes,
        const uint64_t& currentLineInMask);
    __aicore__ inline void CopyInMask1NSD(
        LocalTensor<bool>& maskLocal, uint64_t& lineCnt, const LineInfo& start, const LineInfo& end);
    __aicore__ inline void CopyInMaskB1SD(
        LocalTensor<bool>& maskLocal, uint64_t& lineCnt, LineInfo& start, const LineInfo& end);
    __aicore__ inline void CopyInMask11SD(
        LocalTensor<bool>& maskLocal, uint64_t& lineCnt, const LineInfo& start, const LineInfo& end);
    __aicore__ inline void CalcLineInfo(LineInfo& info, const uint64_t& currentLoop, const uint64_t& curLineNum);

    TPipe* pipe;
    GlobalTensor<T> yGradGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xGradGm;
    GlobalTensor<bool> maskGm;
    TBuf<TPosition::VECCALC> yGradTmpBuffer;
    TBuf<TPosition::VECCALC> yTmpBuffer;
    TBuf<TPosition::VECCALC> sharedBuffer;

    uint64_t channel_;
    uint64_t seqLength_;
    uint64_t headDim_;
    uint64_t paddedHeadDim_;
    uint64_t maxLinePerLoop_;
    uint64_t maskMoveMode_;
    uint64_t usedCoreNum_;
    float scale_;

    uint64_t lineOffset;
    uint64_t loopTimes;
    uint64_t minLine;
    uint64_t linePerBatch;
    uint64_t paddingNum;
    uint64_t maskPaddingNum;
    uint64_t dupNum;
    uint64_t oneLineLen;
    uint64_t currentCoreIdx;
    uint64_t lineNum;
    uint64_t moveNum;
    uint64_t calcNum;
    uint64_t alignedHeadDim;
    uint64_t maxLineBytes;
    uint64_t maxLineBytesMask;
    uint64_t maxLineBytesB32;
    uint16_t stride;
    uint16_t maskStride;
    CopyType copyType;
    DataCopyParams copyParams;
    DataCopyExtParams copyExtParams;
    DataCopyPadExtParams<T> copyPadExtParams;
    DataCopyPadExtParams<bool> maskCopyPadExtParams;
};

template <typename T>
__aicore__ inline uint64_t ScaledMaskedSoftmaxGradV2Base<T>::CeilDiv(const uint64_t dividend, const uint64_t divisor)
{
    if (divisor == 0) {
        return divisor;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::Init(
    const GmTensor* gmTensor, const ScaledMaskedSoftmaxGradV2TilingData& tilingData, TPipe* pipeIn)
{
    channel_ = tilingData.channel;
    seqLength_ = tilingData.seqLength;
    headDim_ = tilingData.headDim;
    paddedHeadDim_ = tilingData.paddedHeadDim;
    maxLinePerLoop_ = tilingData.maxLinePerLoop;
    maskMoveMode_ = tilingData.maskMoveMode;
    usedCoreNum_ = tilingData.usedCoreNum;
    scale_ = tilingData.scale;
    pipe = pipeIn;

    currentCoreIdx = GetBlockIdx();
    uint64_t dataSize = sizeof(T);

    if (currentCoreIdx < tilingData.headCoreNum) {
        lineOffset = currentCoreIdx * tilingData.totalLinePerHeadCore;
        loopTimes = CeilDiv(tilingData.totalLinePerHeadCore, maxLinePerLoop_);
        minLine = tilingData.tailLinePerHeadCore;
    } else {
        lineOffset = tilingData.headCoreNum * tilingData.totalLinePerHeadCore +
                     (currentCoreIdx - tilingData.headCoreNum) * tilingData.totalLinePerTailCore;
        loopTimes = CeilDiv(tilingData.totalLinePerTailCore, maxLinePerLoop_);
        minLine = tilingData.tailLinePerTailCore;
    }

    uint64_t gmOffset = lineOffset * headDim_;
    linePerBatch = channel_ * seqLength_;
    paddingNum = (paddedHeadDim_ - headDim_) % (BLK_LEN / dataSize);
    maskPaddingNum = (paddedHeadDim_ - headDim_) % BLK_LEN;
    dupNum = paddedHeadDim_ - headDim_ - paddingNum;
    oneLineLen = headDim_ * dataSize;
    lineNum = maxLinePerLoop_;
    moveNum = lineNum * headDim_;
    calcNum = lineNum * paddedHeadDim_;
    alignedHeadDim = (headDim_ * dataSize + BLK_LEN - 1) / BLK_LEN * BLK_LEN / dataSize;
    maxLineBytes = maxLinePerLoop_ * paddedHeadDim_ * dataSize;
    maxLineBytesMask = maxLinePerLoop_ * paddedHeadDim_;
    maxLineBytesB32 = maxLinePerLoop_ * paddedHeadDim_ * SIZE_4;
    stride = static_cast<uint16_t>(dupNum * dataSize / BLK_LEN);
    maskStride = static_cast<uint16_t>((paddedHeadDim_ - headDim_ - maskPaddingNum) / BLK_LEN);
    copyPadExtParams = {true, 0, static_cast<uint8_t>(paddingNum), MASK_VALUE};
    maskCopyPadExtParams = {true, 0, static_cast<uint8_t>(maskPaddingNum), 1};

    yGradGm.SetGlobalBuffer((__gm__ T*)gmTensor->yGrad + gmOffset);
    yGm.SetGlobalBuffer((__gm__ T*)gmTensor->y + gmOffset);
    maskGm.SetGlobalBuffer((__gm__ bool*)gmTensor->mask);
    xGradGm.SetGlobalBuffer((__gm__ T*)gmTensor->xGrad + gmOffset);
    pipe->InitBuffer(sharedBuffer, tilingData.selectSize);
    if constexpr (!IsSameType<T, float>::value) {
        pipe->InitBuffer(yGradTmpBuffer, maxLineBytesB32);
        pipe->InitBuffer(yTmpBuffer, maxLineBytesB32);
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::SetDataCopyParams(const uint64_t& currentLineNum)
{
    copyExtParams.blockCount = static_cast<uint32_t>(currentLineNum);
    if (copyType == CopyType::INPUT) {
        copyExtParams.blockLen = static_cast<uint32_t>(oneLineLen);
        copyExtParams.srcStride = 0;
        copyExtParams.dstStride = stride;
    } else if (copyType == CopyType::OUTPUT) {
        copyExtParams.blockLen = static_cast<uint32_t>(oneLineLen);
        copyExtParams.srcStride = stride;
        copyExtParams.dstStride = 0;
    } else {
        copyExtParams.blockLen = static_cast<uint32_t>(headDim_);
        copyExtParams.srcStride = 0;
        copyExtParams.dstStride = maskStride;
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::CopyIn(
    LocalTensor<T>& yGradLocal, LocalTensor<T>& yLocal, const uint64_t& offset)
{
    if (headDim_ % ALIGNED_NUM == 0) {
        DataCopy(yGradLocal, yGradGm[offset], moveNum);
        DataCopy(yLocal, yGm[offset], moveNum);
    } else {
#if __CCE_AICORE__ >= 220
        copyType = CopyType::INPUT;
        SetDataCopyParams(lineNum);
        DataCopyPad(yGradLocal, yGradGm[offset], copyExtParams, copyPadExtParams);
        DataCopyPad(yLocal, yGm[offset], copyExtParams, copyPadExtParams);
#else
        copyParams.blockCount = static_cast<uint32_t>(lineNum);
        copyParams.blockLen = static_cast<uint32_t>(oneLineLen);
        copyParams.srcStride = 0;
        copyParams.dstStride = stride;
        DataCopy(yGradLocal, yGradGm[offset], copyParams);
        DataCopy(yLocal, yGm[offset], copyParams);
#endif
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::CopyInMask(
    LocalTensor<bool>& maskLocal, const uint64_t& currentLoop)
{
    copyType = CopyType::MASK_INPUT;
    uint64_t lineCnt = 0u;
    LineInfo start;
    LineInfo end;
    CalcLineInfo(start, currentLoop, 0);
    CalcLineInfo(end, currentLoop, lineNum);
    if (maskMoveMode_ == 0) {
        CopyInMaskLines(maskLocal, lineCnt, lineNum, start.currentLineInMask);
    } else if (maskMoveMode_ == 1) {
        CopyInMask1NSD(maskLocal, lineCnt, start, end);
    } else if (maskMoveMode_ == 2) {
        CopyInMaskB1SD(maskLocal, lineCnt, start, end);
    } else {
        CopyInMask11SD(maskLocal, lineCnt, start, end);
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::CopyOut(LocalTensor<T>& outLocal, const uint64_t& offset)
{
    copyType = CopyType::OUTPUT;
    if (headDim_ % ALIGNED_NUM == 0) {
        DataCopy(xGradGm[offset], outLocal, moveNum);
    } else {
#if __CCE_AICORE__ >= 220
        copyType = CopyType::OUTPUT;
        SetDataCopyParams(lineNum);
        DataCopyPad(xGradGm[offset], outLocal, copyExtParams);
#else
        copyParams.blockCount = static_cast<uint32_t>(lineNum);
        copyParams.blockLen = static_cast<uint32_t>(oneLineLen);
        copyParams.srcStride = stride;
        copyParams.dstStride = 0;
        DataCopy(xGradGm[offset], outLocal, copyParams);
#endif
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::CopyInMaskLines(
    LocalTensor<bool>& maskLocal, uint64_t& lineCnt, const uint64_t& curLineNum, const uint64_t& currentLineInMask)
{
    uint64_t maskGmOffset = currentLineInMask * headDim_;
    if (headDim_ % ALIGNED_NUM == 0) {
        DataCopy(maskLocal[lineCnt * paddedHeadDim_], maskGm[maskGmOffset], curLineNum * paddedHeadDim_);
    } else {
        uint64_t maskLocalOffset = lineCnt * paddedHeadDim_;
#if __CCE_AICORE__ >= 220
        copyType = CopyType::MASK_INPUT;
        SetDataCopyParams(curLineNum);
        DataCopyPad(maskLocal[maskLocalOffset], maskGm[maskGmOffset], copyExtParams, maskCopyPadExtParams);
#else
        copyParams.blockCount = static_cast<uint32_t>(curLineNum);
        copyParams.blockLen = static_cast<uint32_t>(headDim_);
        copyParams.srcStride = 0;
        copyParams.dstStride = maskStride;
        DataCopy(maskLocal[maskLocalOffset], maskGm[maskGmOffset], copyParams);
#endif
    }
    lineCnt += curLineNum;
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::CopyInMaskBlock(
    LocalTensor<bool>& maskLocal, uint64_t& lineCnt, const uint64_t& curLineNum, uint64_t repeatTimes,
    const uint64_t& currentLineInMask)
{
    uint64_t maskGmOffset = currentLineInMask * headDim_;
    if (headDim_ % ALIGNED_NUM == 0) {
        for (; repeatTimes > 0; --repeatTimes) {
            DataCopy(maskLocal[lineCnt * paddedHeadDim_], maskGm[maskGmOffset], curLineNum * paddedHeadDim_);
            lineCnt += curLineNum;
        }
    } else {
#if __CCE_AICORE__ >= 220
        copyType = CopyType::MASK_INPUT;
        SetDataCopyParams(curLineNum);
        for (; repeatTimes > 0; --repeatTimes) {
            uint64_t maskLocalOffset = lineCnt * paddedHeadDim_;
            DataCopyPad(maskLocal[maskLocalOffset], maskGm[maskGmOffset], copyExtParams, maskCopyPadExtParams);
            lineCnt += curLineNum;
        }
#else
        copyParams.blockCount = static_cast<uint32_t>(curLineNum);
        copyParams.blockLen = static_cast<uint32_t>(headDim_);
        copyParams.srcStride = 0;
        copyParams.dstStride = maskStride;
        for (; repeatTimes > 0; --repeatTimes) {
            uint64_t maskLocalOffset = lineCnt * paddedHeadDim_;
            DataCopy(maskLocal[maskLocalOffset], maskGm[maskGmOffset], copyParams);
            lineCnt += curLineNum;
        }
#endif
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::CopyInMask1NSD(
    LocalTensor<bool>& maskLocal, uint64_t& lineCnt, const LineInfo& start, const LineInfo& end)
{
    if (start.currentBatch == end.currentBatch) {
        CopyInMaskLines(maskLocal, lineCnt, lineNum, start.currentLineInMask);
    } else {
        // first batch
        CopyInMaskLines(maskLocal, lineCnt, linePerBatch - start.currentLineInBatch, start.currentLineInMask);
        // middle batch
        CopyInMaskBlock(maskLocal, lineCnt, linePerBatch, end.currentBatch - start.currentBatch - 1, 0);
        // last batch
        if (end.currentLineInBatch != 0) {
            CopyInMaskLines(maskLocal, lineCnt, end.currentLineInBatch, 0);
        }
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::CopyInMaskB1SD(
    LocalTensor<bool>& maskLocal, uint64_t& lineCnt, LineInfo& start, const LineInfo& end)
{
    if (start.currentChannel == end.currentChannel) {
        CopyInMaskLines(maskLocal, lineCnt, lineNum, start.currentLineInMask);
    } else {
        CopyInMaskLines(maskLocal, lineCnt, seqLength_ - start.currentLineInChannel, start.currentLineInMask);
        start.currentLineInMask -= start.currentLineInChannel;
        if (start.currentBatch == end.currentBatch) {
            CopyInMaskBlock(
                maskLocal, lineCnt, seqLength_, end.currentChannel - start.currentChannel - 1, start.currentLineInMask);
            if (end.currentLineInChannel != 0) {
                CopyInMaskLines(maskLocal, lineCnt, end.currentLineInChannel, start.currentLineInMask);
            }
        } else {
            // first batch
            CopyInMaskBlock(
                maskLocal, lineCnt, seqLength_, channel_ - start.currentChannel % channel_ - 1,
                start.currentLineInMask);
            start.currentLineInMask += seqLength_;
            // middle batch
            for (uint64_t i = 1; i < end.currentBatch - start.currentBatch; ++i) {
                CopyInMaskBlock(maskLocal, lineCnt, seqLength_, channel_, start.currentLineInMask);
                start.currentLineInMask += seqLength_;
            }
            // last batch
            CopyInMaskBlock(maskLocal, lineCnt, seqLength_, end.currentChannel % channel_, start.currentLineInMask);
            if (end.currentLineInChannel != 0) {
                CopyInMaskLines(maskLocal, lineCnt, end.currentLineInChannel, start.currentLineInMask);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::CopyInMask11SD(
    LocalTensor<bool>& maskLocal, uint64_t& lineCnt, const LineInfo& start, const LineInfo& end)
{
    if (start.currentChannel == end.currentChannel) {
        CopyInMaskLines(maskLocal, lineCnt, lineNum, start.currentLineInMask);
    } else {
        CopyInMaskLines(maskLocal, lineCnt, seqLength_ - start.currentLineInChannel, start.currentLineInMask);
        uint64_t repeatTimes = end.currentChannel - start.currentChannel - 1;
        CopyInMaskBlock(maskLocal, lineCnt, seqLength_, repeatTimes, 0);
        if (end.currentLineInChannel != 0) {
            CopyInMaskLines(maskLocal, lineCnt, end.currentLineInChannel, 0);
        }
    }
}

template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2Base<T>::CalcLineInfo(
    LineInfo& info, const uint64_t& currentLoop, const uint64_t& curLineNum)
{
    info.currentLine = lineOffset + currentLoop * maxLinePerLoop_ + curLineNum;
    info.currentBatch = info.currentLine / linePerBatch;
    info.currentChannel = info.currentLine / seqLength_;
    info.currentLineInMask = info.currentLine;
    info.currentLineInBatch = info.currentLine % linePerBatch;
    info.currentLineInChannel = info.currentLine % seqLength_;
    if (maskMoveMode_ == 1) {
        info.currentLineInMask = info.currentLine % linePerBatch;
    } else if (maskMoveMode_ == 2) {
        info.currentLineInMask = info.currentLine % seqLength_ + info.currentBatch * seqLength_;
    } else if (maskMoveMode_ == 3) {
        info.currentLineInMask = info.currentLine % seqLength_;
    }
}
} // namespace ScaledMaskedSoftmaxGradV2

#endif