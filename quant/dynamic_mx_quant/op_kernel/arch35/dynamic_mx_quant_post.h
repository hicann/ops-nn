/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_post.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_POST_H
#define DYNAMIC_MX_QUANT_POST_H
#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "dynamic_mx_quant_common.h"
#include "op_kernel/math_util.h"

namespace DynamicMxQuant {
using namespace AscendC;
constexpr int64_t NON_TAIL_AXIS_NON_ZERO = 0;
constexpr int64_t NON_TAIL_AXIS_ZERO_LARGE = 1;
constexpr int64_t NON_TAIL_AXIS_ZERO_MEDIUM = 2;
constexpr int64_t NON_TAIL_AXIS_ZERO_SMALL = 3;
constexpr int64_t OPT_MAX_COL = 16;

class DynamicMxQuantPost {
public:
    __aicore__ inline DynamicMxQuantPost(){};
    __aicore__ inline void Init(GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(int64_t wsGmOffset1, int64_t wsGmOffset2, int64_t blockLen, int64_t blockCount);
    __aicore__ inline void CopyInPaddingZero(
        int64_t wsGmOffset1, int64_t wsGmOffset2, int64_t axisSize1, int64_t axisSize2, int64_t axisSize3);
    __aicore__ inline void ComputeInterleave(uint32_t elementNum);
    __aicore__ inline void CopyOut(int64_t scaleGmOffset, int64_t blockLen, int64_t blockCount);
    __aicore__ inline void ComputeInterleaveVF(
        __local_mem__ uint8_t* LocalAddr1, __local_mem__ uint8_t* LocalAddr2, __local_mem__ uint8_t* ScaleAddr,
        uint32_t elementNum);

private:
    __aicore__ inline void TailAxisCompute();
    __aicore__ inline void TailAxisOptCompute();
    __aicore__ inline void NonTailAxisNonZeroCompute();
    __aicore__ inline void NonTailAxisZeroLargeCompute();
    __aicore__ inline void NonTailAxisZeroMediumCompute();
    __aicore__ inline void NonTailAxisZeroSmallCompute();
    __aicore__ inline void NonTailAxisCompute();

private:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> inQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue1_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue2_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scaleQueue;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    TBuf<QuePosition::VECCALC> offsetBuf_;
    TBuf<QuePosition::VECCALC> offsetTmpBuf_;
    GlobalTensor<uint8_t> mxScaleGm_;
    GlobalTensor<uint8_t> workspaceGm_;

    int64_t blockIdx_ = 0;
    int64_t isTailAxis_ = 0;
    int64_t mxScaleSize_ = 0;
    int64_t preAxisSize_ = 0;
    int64_t axisSize_ = 0;
    int64_t postAxisSize_ = 0;

    int64_t lastSize_ = 0;
    int64_t lastSizeAlign_ = 0;
    int64_t lastPadSize_ = 0;

    int64_t firstSize_ = 0;
    int64_t firstSizeAlign_ = 0;
    int64_t firstPadSize_ = 0;

    int64_t rowLoopBlockCount_ = 0; // 列循环次数
    int64_t rowLoopBlockSize_ = 0;  // 列循环块大小
    int64_t colLoopBlockCount_ = 0; // 行循环次数
    int64_t colLoopBlockSize_ = 0;  // 行循环块大小

    uint16_t rowPerLoop_ = 0; // 计算单次能放下的完整行数
    int64_t nonTailAxisTemplate = 0;
};

__aicore__ inline void DynamicMxQuantPost::Init(
    GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData)
{
    SyncAll();
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ > 0) {
        return;
    }
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    this->mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale));
    this->workspaceGm_.SetGlobalBuffer((__gm__ uint8_t*)(workspace));

    isTailAxis_ = tilingData->isTailAxis;
    mxScaleSize_ = tilingData->mxScaleSize;
    preAxisSize_ = tilingData->preAxisSize;
    axisSize_ = tilingData->blockSizeNumInAxis;
    postAxisSize_ = tilingData->postAxisSize;

    if (isTailAxis_ == 0) {
        firstSize_ = (axisSize_ + DIGIT_TWO - 1) / DIGIT_TWO * DIGIT_TWO * preAxisSize_;
        lastSize_ = postAxisSize_;
        lastSizeAlign_ = (lastSize_ + OUT_ELE_NUM_ONE_BLK_FP8 - 1) / OUT_ELE_NUM_ONE_BLK_FP8 * OUT_ELE_NUM_ONE_BLK_FP8;

        rowPerLoop_ = SCALE_BUFFER_SIZE / lastSize_;
        if (rowPerLoop_ >= 1) {
            // 可以放下完整的多行
            rowPerLoop_ = (rowPerLoop_ < (firstSize_ / DIGIT_TWO)) ? rowPerLoop_ : (firstSize_ / DIGIT_TWO);
            rowPerLoop_ = (rowPerLoop_ < MAX_MTE_BLOCK_COUNT) ? rowPerLoop_ : MAX_MTE_BLOCK_COUNT;
            rowLoopBlockCount_ =
                (firstSize_ / DIGIT_TWO + rowPerLoop_ - 1) / rowPerLoop_; // ceil(preAxisSize_ / rowPerLoop_)
            colLoopBlockCount_ = 1;
            colLoopBlockSize_ = lastSize_; // 每次处理完整行，无需列分块
        } else {
            // 无法放下完整的一行，需要按列分块
            rowPerLoop_ = 1;                       // 每次只能处理1行的部分
            colLoopBlockSize_ = SCALE_BUFFER_SIZE; // 每次处理的列数
            colLoopBlockCount_ =
                (lastSize_ + colLoopBlockSize_ - 1) / colLoopBlockSize_; // ceil(lastSizeAlign_ / colLoopBlockSize_)
            rowLoopBlockCount_ = firstSize_ / DIGIT_TWO;                 // 每行需要分多次处理
        }
        if (axisSize_ % DIGIT_TWO == 0) {
            nonTailAxisTemplate = NON_TAIL_AXIS_NON_ZERO;
        } else {
            if (rowPerLoop_ == 1) {
                nonTailAxisTemplate = NON_TAIL_AXIS_ZERO_LARGE;
            } else if (DIGIT_TWO * rowPerLoop_ < axisSize_) {
                nonTailAxisTemplate = NON_TAIL_AXIS_ZERO_MEDIUM;
            } else {
                nonTailAxisTemplate = NON_TAIL_AXIS_ZERO_SMALL;
            }
        }
    } else {
        lastSize_ = axisSize_ * postAxisSize_;
        lastSizeAlign_ = (axisSize_ * postAxisSize_ + OUT_ELE_NUM_ONE_BLK_FP8 - 1) / OUT_ELE_NUM_ONE_BLK_FP8 *
                         OUT_ELE_NUM_ONE_BLK_FP8;
        lastPadSize_ = (axisSize_ + DIGIT_TWO - 1) / DIGIT_TWO * DIGIT_TWO * postAxisSize_;

        rowPerLoop_ = SCALE_BUFFER_SIZE / lastSizeAlign_;
        if (rowPerLoop_ >= 1) {
            // 可以放下完整的多行
            rowPerLoop_ = (rowPerLoop_ < preAxisSize_) ? rowPerLoop_ : preAxisSize_;
            rowPerLoop_ = (rowPerLoop_ < MAX_MTE_BLOCK_COUNT) ? rowPerLoop_ : MAX_MTE_BLOCK_COUNT;
            rowLoopBlockCount_ = (preAxisSize_ + rowPerLoop_ - 1) / rowPerLoop_; // ceil(preAxisSize_ / rowPerLoop_)
            colLoopBlockCount_ = 1;
            colLoopBlockSize_ = lastSize_; // 每次处理完整行，无需列分块
        } else {
            // 无法放下完整的一行，需要按列分块
            rowPerLoop_ = 1;                       // 每次只能处理1行的部分
            colLoopBlockSize_ = SCALE_BUFFER_SIZE; // 每次处理的列数
            colLoopBlockCount_ = (lastSizeAlign_ + colLoopBlockSize_ - 1) /
                                 colLoopBlockSize_; // ceil(lastSizeAlign_ / colLoopBlockSize_)
            rowLoopBlockCount_ = preAxisSize_;
        }
    }
    InitOutput<uint8_t>(mxScaleGm_, mxScaleSize_, 0);
    PipeBarrier<PIPE_ALL>();
    if (isTailAxis_ && lastSize_ < OPT_MAX_COL) {
        this->pipe_.InitBuffer(this->inQueue1_, DB_BUFFER, SCALE_BUFFER_SIZE);
        this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, SCALE_BUFFER_SIZE);
        this->pipe_.InitBuffer(this->offsetBuf_, SCALE_BUFFER_SIZE * sizeof(int32_t));
        this->pipe_.InitBuffer(this->offsetTmpBuf_, SCALE_BUFFER_SIZE * sizeof(int32_t));
    } else if (isTailAxis_) {
        this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, SCALE_BUFFER_SIZE);
    } else {
        this->pipe_.InitBuffer(this->inQueue1_, DB_BUFFER, SCALE_BUFFER_SIZE);
        this->pipe_.InitBuffer(this->inQueue2_, DB_BUFFER, SCALE_BUFFER_SIZE);
        this->pipe_.InitBuffer(this->scaleQueue, DB_BUFFER, DIGIT_TWO * SCALE_BUFFER_SIZE);
    }
}

__aicore__ inline void DynamicMxQuantPost::Process()
{
    if (blockIdx_ > 0) {
        return;
    }
    if (isTailAxis_ && lastSize_ < OPT_MAX_COL) {
        TailAxisOptCompute();
    } else if (isTailAxis_) {
        TailAxisCompute();
    } else {
        NonTailAxisCompute();
    }
}

__aicore__ inline void DynamicMxQuantPost::TailAxisCompute()
{
    for (int64_t i = 0; i < rowLoopBlockCount_; ++i) {
        int64_t rowSize = (i == rowLoopBlockCount_ - 1) ? preAxisSize_ - i * rowPerLoop_ : rowPerLoop_;
        for (int64_t j = 0; j < colLoopBlockCount_; ++j) {
            int64_t colSize = (j == colLoopBlockCount_ - 1) ? lastSize_ - j * colLoopBlockSize_ : colLoopBlockSize_;
            int64_t wsGmOffset = i * rowPerLoop_ * lastSize_ + j * colLoopBlockSize_;
            int64_t scaleGmOffset = i * rowPerLoop_ * lastPadSize_ + j * colLoopBlockSize_;

            auto localBuf = inQueue_.AllocTensor<uint8_t>();
            DataCopyExtParams copyParams;
            DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
            copyParams.blockCount = rowSize;
            copyParams.blockLen = colSize;
            copyParams.dstStride = 0;
            copyParams.srcStride = 0;
            DataCopyPad<uint8_t>(localBuf, workspaceGm_[wsGmOffset], copyParams, padParams);
            inQueue_.EnQue(localBuf);

            copyParams.dstStride = lastPadSize_ - lastSize_;

            localBuf = inQueue_.DeQue<uint8_t>();
            DataCopyPad<uint8_t>(mxScaleGm_[scaleGmOffset], localBuf, copyParams);
            inQueue_.FreeTensor(localBuf);
        }
    }
}

__aicore__ inline void DynamicMxQuantPost::TailAxisOptCompute()
{
    LocalTensor<int32_t> offsetLocal = offsetBuf_.Get<int32_t>();
    LocalTensor<int32_t> offsetTmpLocal = offsetTmpBuf_.Get<int32_t>();

    AscendC::CreateVecIndex(offsetLocal, 0, rowPerLoop_ * colLoopBlockSize_);
    AscendC::Duplicate(offsetTmpLocal, (int32_t)(colLoopBlockSize_), rowPerLoop_ * colLoopBlockSize_);
    AscendC::Div<int32_t>(offsetTmpLocal, offsetLocal, offsetTmpLocal, rowPerLoop_ * colLoopBlockSize_);
    AscendC::Add<int32_t>(offsetLocal, offsetLocal, offsetTmpLocal, rowPerLoop_ * colLoopBlockSize_);

    for (int64_t i = 0; i < rowLoopBlockCount_; ++i) {
        int64_t rowSize = (i == rowLoopBlockCount_ - 1) ? preAxisSize_ - i * rowPerLoop_ : rowPerLoop_;
        for (int64_t j = 0; j < colLoopBlockCount_; ++j) {
            int64_t colSize = (j == colLoopBlockCount_ - 1) ? lastSize_ - j * colLoopBlockSize_ : colLoopBlockSize_;
            int64_t colSizePad = Ops::Base::CeilAlign(colSize, DIGIT_TWO);
            int64_t wsGmOffset = i * rowPerLoop_ * colSize + j * colLoopBlockSize_;
            int64_t scaleGmOffset = i * rowPerLoop_ * colSizePad + j * colLoopBlockSize_;

            auto inputLocal = inQueue1_.AllocTensor<uint8_t>();
            DataCopyExtParams copyParams;
            DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
            copyParams.blockCount = 1;
            copyParams.blockLen = colSize * rowSize;
            copyParams.dstStride = 0;
            copyParams.srcStride = 0;
            DataCopyPad<uint8_t>(inputLocal, workspaceGm_[wsGmOffset], copyParams, padParams);

            inQueue1_.EnQue(inputLocal);
            auto outputLocal = outQueue_.AllocTensor<uint8_t>();
            inputLocal = inQueue1_.DeQue<uint8_t>();
            auto offsetLocalU32 = offsetLocal.template ReinterpretCast<uint32_t>();
            AscendC::Duplicate(outputLocal, (uint8_t)(0), rowSize * colSizePad);
            AscendC::Scatter(outputLocal, inputLocal, offsetLocalU32, (uint32_t)0, rowSize * colSize);
            inQueue1_.FreeTensor(inputLocal);

            outQueue_.EnQue(outputLocal);
            outputLocal = outQueue_.DeQue<uint8_t>();
            copyParams.blockLen = rowSize * colSizePad;
            DataCopyPad<uint8_t>(mxScaleGm_[scaleGmOffset], outputLocal, copyParams);
            outQueue_.FreeTensor(outputLocal);
        }
    }
}

__aicore__ inline void DynamicMxQuantPost::NonTailAxisNonZeroCompute()
{
    for (int64_t i = 0; i < rowLoopBlockCount_; ++i) {
        int64_t rowSize = (i == rowLoopBlockCount_ - 1) ? firstSize_ / DIGIT_TWO - i * rowPerLoop_ : rowPerLoop_;
        for (int64_t j = 0; j < colLoopBlockCount_; ++j) {
            int64_t colSize = (j == colLoopBlockCount_ - 1) ? lastSize_ - j * colLoopBlockSize_ : colLoopBlockSize_;
            int64_t wsGmOffset1 = DIGIT_TWO * i * rowPerLoop_ * lastSize_ + j * colLoopBlockSize_;
            int64_t wsGmOffset2 = (DIGIT_TWO * i * rowPerLoop_ + 1) * lastSize_ + j * colLoopBlockSize_;
            int64_t scaleGmOffset = DIGIT_TWO * i * rowPerLoop_ * lastSize_ + 2 * j * colLoopBlockSize_;

            auto scaleOut = scaleQueue.AllocTensor<uint8_t>();
            scaleQueue.EnQue(scaleOut);
            CopyIn(wsGmOffset1, wsGmOffset2, colSize, rowSize);
            ComputeInterleave(colSize * rowSize);
            CopyOut(scaleGmOffset, colSize, rowSize * DIGIT_TWO);
        }
    }
}

__aicore__ inline void DynamicMxQuantPost::NonTailAxisZeroLargeCompute()
{
    int64_t rowLoopNum = (axisSize_ + 1) / DIGIT_TWO;
    for (int64_t i = 0; i < preAxisSize_; ++i) {
        for (int64_t j = 0; j < rowLoopNum; ++j) {
            for (int64_t k = 0; k < colLoopBlockCount_; ++k) {
                int64_t colSize = (k == colLoopBlockCount_ - 1) ? lastSize_ - k * colLoopBlockSize_ : colLoopBlockSize_;
                int64_t wsGmOffset1 = i * axisSize_ * lastSize_ + j * DIGIT_TWO * lastSize_ + k * colLoopBlockSize_;
                int64_t wsGmOffset2 = wsGmOffset1 + lastSize_;
                int64_t scaleGmOffset =
                    i * (axisSize_ + 1) * lastSize_ + j * DIGIT_TWO * lastSize_ + DIGIT_TWO * k * colLoopBlockSize_;
                if (j == rowLoopNum - 1) {
                    CopyInPaddingZero(wsGmOffset1, wsGmOffset2, colSize, 1, 1);
                } else {
                    CopyIn(wsGmOffset1, wsGmOffset2, colSize, 1);
                }
                auto scaleOut = scaleQueue.AllocTensor<uint8_t>();
                scaleQueue.EnQue(scaleOut);
                ComputeInterleave(colSize);
                CopyOut(scaleGmOffset, colSize, DIGIT_TWO);
            }
        }
    }
}

__aicore__ inline void DynamicMxQuantPost::NonTailAxisZeroMediumCompute()
{
    int64_t LoopNum = (axisSize_ + 1 + DIGIT_TWO * rowPerLoop_ - 1) / (DIGIT_TWO * rowPerLoop_);
    int64_t rowTailLoop_ = ((axisSize_ + 1) - (LoopNum - 1) * DIGIT_TWO * rowPerLoop_) / DIGIT_TWO;

    for (int64_t i = 0; i < preAxisSize_; ++i) {
        for (int64_t j = 0; j < LoopNum-1; ++j) {
            int64_t colSize = colLoopBlockSize_;
            int64_t wsGmOffset1 = i * axisSize_ * lastSize_ + 2 * j * rowPerLoop_ * lastSize_;
            int64_t wsGmOffset2 = wsGmOffset1 + lastSize_;
            int64_t scaleGmOffset = i * (axisSize_ + 1) * lastSize_ + 2 * j * rowPerLoop_ * lastSize_;
            CopyIn(wsGmOffset1, wsGmOffset2, colSize, rowPerLoop_);
            auto scaleOut = scaleQueue.AllocTensor<uint8_t>();
            scaleQueue.EnQue(scaleOut);
            ComputeInterleave(colSize * rowPerLoop_);
            CopyOut(scaleGmOffset, colSize * rowPerLoop_, DIGIT_TWO);
        }
        int64_t colSize = colLoopBlockSize_;
        int64_t wsGmOffset1 = i * axisSize_ * lastSize_ + 2 * (LoopNum-1) * rowPerLoop_ * lastSize_;
        int64_t wsGmOffset2 = wsGmOffset1 + lastSize_;
        int64_t scaleGmOffset = i * (axisSize_ + 1) * lastSize_ + 2 * (LoopNum-1) * rowPerLoop_ * lastSize_;      
        CopyInPaddingZero(wsGmOffset1, wsGmOffset2, colSize, rowTailLoop_, 1);       
        auto scaleOut = scaleQueue.AllocTensor<uint8_t>();
        scaleQueue.EnQue(scaleOut);
        ComputeInterleave(colSize * rowPerLoop_);
        CopyOut(scaleGmOffset, colSize * rowTailLoop_, DIGIT_TWO);
    }
}

__aicore__ inline void DynamicMxQuantPost::NonTailAxisZeroSmallCompute()
{
    int64_t rowPerLoopAlign_ = (rowPerLoop_ * DIGIT_TWO) / (axisSize_ + 1) * (axisSize_ + 1);
    int64_t LoopNum = (firstSize_ + rowPerLoopAlign_ - 1) / rowPerLoopAlign_;
    int64_t rowPerLoop = rowPerLoopAlign_ / (axisSize_ + 1) * axisSize_;

    for (int64_t i = 0; i < LoopNum; ++i) {
        int64_t Size = (i == LoopNum - 1) ? firstSize_ - i * rowPerLoopAlign_ : rowPerLoopAlign_;
        int64_t wsGmOffset1 = i * rowPerLoop * lastSize_;
        int64_t wsGmOffset2 = wsGmOffset1 + lastSize_;
        int64_t scaleGmOffset = i * rowPerLoopAlign_ * lastSize_;
        CopyInPaddingZero(wsGmOffset1, wsGmOffset2, lastSize_, (axisSize_ + 1) / DIGIT_TWO, Size / (axisSize_ + 1));

        auto scaleOut = scaleQueue.AllocTensor<uint8_t>();
        scaleQueue.EnQue(scaleOut);
        ComputeInterleave(lastSize_ * (Size / DIGIT_TWO));
        CopyOut(scaleGmOffset, lastSize_, Size);
    }
}

__aicore__ inline void DynamicMxQuantPost::NonTailAxisCompute()
{
    if (nonTailAxisTemplate == NON_TAIL_AXIS_NON_ZERO) {
        NonTailAxisNonZeroCompute();
    } else if (nonTailAxisTemplate == NON_TAIL_AXIS_ZERO_LARGE) {
        NonTailAxisZeroLargeCompute();
    } else if (nonTailAxisTemplate == NON_TAIL_AXIS_ZERO_MEDIUM) {
        NonTailAxisZeroMediumCompute();
    } else if (nonTailAxisTemplate == NON_TAIL_AXIS_ZERO_SMALL) {
        NonTailAxisZeroSmallCompute();
    }
}

__aicore__ inline void DynamicMxQuantPost::CopyIn(
    int64_t wsGmOffset1, int64_t wsGmOffset2, int64_t blockLen, int64_t blockCount)
{
    MultiCopyLoopInfo<2> loopInfo = {
        {1, static_cast<uint32_t>(2 * blockLen)},
        {1, static_cast<uint32_t>(blockLen)},
        {static_cast<uint32_t>(blockLen), static_cast<uint32_t>(blockCount)},
        {0, 0},
        {0, 0}};
    MultiCopyParams<uint8_t, 2> params = {loopInfo, 0};
    static constexpr MultiCopyConfig config = {false};
    auto localBuf1 = inQueue1_.AllocTensor<uint8_t>();
    DataCopy<uint8_t, 2, config>(localBuf1, workspaceGm_[wsGmOffset1], params);
    inQueue1_.EnQue(localBuf1);
    auto localBuf2 = inQueue2_.AllocTensor<uint8_t>();
    DataCopy<uint8_t, 2, config>(localBuf2, workspaceGm_[wsGmOffset2], params);
    inQueue2_.EnQue(localBuf2);
}

__aicore__ inline void DynamicMxQuantPost::CopyInPaddingZero(
    int64_t wsGmOffset1, int64_t wsGmOffset2, int64_t axisSize1, int64_t axisSize2, int64_t axisSize3)
{
    MultiCopyLoopInfo<3> loopInfo1 = {
        {1, static_cast<uint32_t>(DIGIT_TWO * axisSize1),
         static_cast<uint32_t>((DIGIT_TWO * axisSize2 - 1) * axisSize1)},
        {1, static_cast<uint32_t>(axisSize1), static_cast<uint32_t>(axisSize1 * axisSize2)},
        {static_cast<uint32_t>(axisSize1), static_cast<uint32_t>(axisSize2), static_cast<uint32_t>(axisSize3)},
        {0, 0, 0},
        {0, 0, 0}};
    MultiCopyLoopInfo<3> loopInfo2 = {
        {1, static_cast<uint32_t>(DIGIT_TWO * axisSize1),
         static_cast<uint32_t>((DIGIT_TWO * axisSize2 - 1) * axisSize1)},
        {1, static_cast<uint32_t>(axisSize1), static_cast<uint32_t>(axisSize1 * axisSize2)},
        {static_cast<uint32_t>(axisSize1), static_cast<uint32_t>(axisSize2 - 1), static_cast<uint32_t>(axisSize3)},
        {0, 0, 0},
        {0, 1, 0}};
    static constexpr MultiCopyConfig config = {false};
    MultiCopyParams<uint8_t, 3> params1 = {loopInfo1, 0};
    auto localBuf1 = inQueue1_.AllocTensor<uint8_t>();
    DataCopy<uint8_t, 3, config>(localBuf1, workspaceGm_[wsGmOffset1], params1);
    inQueue1_.EnQue(localBuf1);
    MultiCopyParams<uint8_t, 3> params2 = {loopInfo2, 0};
    auto localBuf2 = inQueue2_.AllocTensor<uint8_t>();
    if (axisSize2 > 1) {
        DataCopy<uint8_t, 3, config>(localBuf2, workspaceGm_[wsGmOffset2], params2);
    } else {
        Duplicate<uint8_t>(localBuf2, 0, static_cast<int32_t>(axisSize1 * axisSize3));
    }
    inQueue2_.EnQue(localBuf2);
}

__aicore__ inline void DynamicMxQuantPost::ComputeInterleave(uint32_t elementNum)
{
    auto InBuf1 = inQueue1_.DeQue<uint8_t>();
    auto InBuf2 = inQueue2_.DeQue<uint8_t>();
    auto OutBuf = scaleQueue.DeQue<uint8_t>();
    __local_mem__ uint8_t* InAddr1 = (__local_mem__ uint8_t*)InBuf1.GetPhyAddr();
    __local_mem__ uint8_t* InAddr2 = (__local_mem__ uint8_t*)InBuf2.GetPhyAddr();
    __local_mem__ uint8_t* OutAddr = (__local_mem__ uint8_t*)OutBuf.GetPhyAddr();

    ComputeInterleaveVF(InAddr1, InAddr2, OutAddr, elementNum);

    inQueue1_.FreeTensor(InBuf1);
    inQueue2_.FreeTensor(InBuf2);
    scaleQueue.EnQue(OutBuf);
}

__aicore__ inline void DynamicMxQuantPost::CopyOut(int64_t scaleGmOffset, int64_t blockLen, int64_t blockCount)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = blockLen * blockCount;
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};

    auto scale = scaleQueue.DeQue<uint8_t>();
    DataCopyPad(mxScaleGm_[scaleGmOffset], scale, copyParams);
    scaleQueue.FreeTensor(scale);
}

__aicore__ inline void DynamicMxQuantPost::ComputeInterleaveVF(
    __local_mem__ uint8_t* LocalAddr1, __local_mem__ uint8_t* LocalAddr2, __local_mem__ uint8_t* ScaleAddr,
    uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(uint8_t);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint8_t, MicroAPI::RegTraitNumOne> vreg0;
        AscendC::MicroAPI::RegTensor<uint8_t, MicroAPI::RegTraitNumOne> vreg1;
        AscendC::MicroAPI::RegTensor<uint8_t, MicroAPI::RegTraitNumOne> vreg3;
        AscendC::MicroAPI::RegTensor<uint8_t, MicroAPI::RegTraitNumOne> vreg4;

        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            AscendC::MicroAPI::DataCopy(vreg0, LocalAddr1 + i * VL);
            AscendC::MicroAPI::DataCopy(vreg1, LocalAddr2 + i * VL);
            AscendC::MicroAPI::Interleave(vreg3, vreg4, vreg0, vreg1);
            AscendC::MicroAPI::DataCopy(ScaleAddr + DIGIT_TWO * i * VL, vreg3, mask);
            AscendC::MicroAPI::DataCopy(ScaleAddr + (DIGIT_TWO * i + 1) * VL, vreg4, mask);
        }
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_POST_H
