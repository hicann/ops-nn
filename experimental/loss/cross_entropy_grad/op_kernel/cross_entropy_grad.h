/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cross_entropy_grad.h
 * \brief CrossEntropyGrad算子Kernel实现
 *
 * 计算公式：grad = softmax(logits) - one_hot(target)
 *
 * 支持两种计算模式：
 *   - 正常模式：每行完整计算 softmax
 *   - 列分割模式：当 rowLen 过大导致 UB 放不下完整行时，分多遍扫描
 */

#ifndef CROSS_ENTROPY_GRAD_H_
#define CROSS_ENTROPY_GRAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cross_entropy_grad_tiling_data.h"
#include "cross_entropy_grad_tiling_key.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_ROWS_PER_TILE = 256;

template <typename T, typename TargetT = int32_t>
class KernelCrossEntropyGrad {
public:
    __aicore__ inline KernelCrossEntropyGrad() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR target, GM_ADDR gradOutput,
                                 GM_ADDR workspace, const CrossEntropyGradTilingData* tilingData)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->rowLen = tilingData->rowLen;
        this->totalRows = tilingData->totalRows;
        this->tileRowNum = tilingData->tileRowNum;
        this->typeBytes = tilingData->typeBytes;
        this->colSplitMode = tilingData->colSplitMode;
        this->tileCols = tilingData->tileCols;
        this->numColPasses = tilingData->numColPasses;

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t bigCoreNum = tilingData->bigCoreNum;

        uint32_t startRow = 0;
        if (coreIdx < bigCoreNum) {
            this->coreRowNum = tilingData->bigCoreRowNum;
            this->tileNum = tilingData->bigCoreTileNum;
            this->tailRowNum = tilingData->bigTailRowNum;
            startRow = coreIdx * tilingData->bigCoreRowNum;
        } else {
            this->coreRowNum = tilingData->smallCoreRowNum;
            this->tileNum = tilingData->smallCoreTileNum;
            this->tailRowNum = tilingData->smallTailRowNum;
            startRow = bigCoreNum * tilingData->bigCoreRowNum +
                       (coreIdx - bigCoreNum) * tilingData->smallCoreRowNum;
        }

        inputGMLogits.SetGlobalBuffer((__gm__ T*)logits + startRow * rowLen, this->coreRowNum * rowLen);
        inputGMTarget.SetGlobalBuffer((__gm__ TargetT*)target + startRow, this->coreRowNum);
        outputGMGrad.SetGlobalBuffer((__gm__ T*)gradOutput + startRow * rowLen, this->coreRowNum * rowLen);

        this->rowLenAlign32 = ((rowLen * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / sizeof(T);

        if (colSplitMode) {
            this->tileColsAlign32 = ((tileCols * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / sizeof(T);
            uint32_t tileDataSize = tileRowNum * tileColsAlign32;

            pipe.InitBuffer(inputQueueLogits, BUFFER_NUM, tileDataSize * sizeof(T));
            pipe.InitBuffer(outputQueueGrad, BUFFER_NUM, tileDataSize * sizeof(T));

            uint32_t targetAlign32 = ((tileRowNum * sizeof(TargetT) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(inputQueueTarget, BUFFER_NUM, targetAlign32);

            uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
            uint32_t elementsPerRepeat = 256 / sizeof(T);
            uint32_t firstMaxRepeat = (tileColsAlign32 + elementsPerRepeat - 1) / elementsPerRepeat;
            uint32_t iter1OutputCount = firstMaxRepeat * 2;
            uint32_t workLocalElements = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
            uint32_t workLocalSize = ((workLocalElements * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

            pipe.InitBuffer(reduceTmpBuf, workLocalSize);
            pipe.InitBuffer(reduceWorkBuf, workLocalSize);

            uint32_t maxBufSize = ((tileRowNum * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(maxBuf, maxBufSize);
            uint32_t sumBufSize = ((tileRowNum * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(sumBuf, sumBufSize);
            pipe.InitBuffer(oneHotBuf, tileDataSize * sizeof(T));
        } else {
            uint32_t tileDataSize = tileRowNum * rowLenAlign32;

            pipe.InitBuffer(inputQueueLogits, BUFFER_NUM, tileDataSize * sizeof(T));
            pipe.InitBuffer(outputQueueGrad, BUFFER_NUM, tileDataSize * sizeof(T));

            uint32_t targetAlign32 = ((tileRowNum * sizeof(TargetT) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(inputQueueTarget, BUFFER_NUM, targetAlign32);

            uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
            uint32_t elementsPerRepeat = 256 / sizeof(T);
            uint32_t firstMaxRepeat = (rowLenAlign32 + elementsPerRepeat - 1) / elementsPerRepeat;
            uint32_t iter1OutputCount = firstMaxRepeat * 2;
            uint32_t workLocalElements = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
            uint32_t workLocalSize = ((workLocalElements * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

            pipe.InitBuffer(reduceTmpBuf, workLocalSize);
            pipe.InitBuffer(reduceWorkBuf, workLocalSize);

            uint32_t maxBufSize = ((tileRowNum * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(maxBuf, maxBufSize);
            uint32_t sumBufSize = ((tileRowNum * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(sumBuf, sumBufSize);
            pipe.InitBuffer(oneHotBuf, tileDataSize * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->coreRowNum == 0) {
            return;
        }

        if (colSplitMode) {
            ProcessColSplit();
        } else {
            ProcessNormal();
        }
    }

private:
    // ========================================================================
    // 正常模式
    // ========================================================================
    __aicore__ inline void ProcessNormal()
    {
        int32_t loopCount = this->tileNum;
        this->processRowNum = this->tileRowNum;

        for (int32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1 && this->tailRowNum > 0) {
                this->processRowNum = this->tailRowNum;
            }
            CopyInNormal(i);
            ComputeNormal(i);
            CopyOutNormal(i);
        }
    }

    __aicore__ inline void CopyInNormal(int32_t progress)
    {
        AscendC::LocalTensor<T> logitsLocal = inputQueueLogits.AllocTensor<T>();
        AscendC::LocalTensor<TargetT> targetLocal = inputQueueTarget.AllocTensor<TargetT>();

        uint32_t gmBase = progress * tileRowNum * rowLen;

        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(processRowNum),
            static_cast<uint32_t>(rowLen * sizeof(T)),
            0, 0, 0
        };
        uint8_t rightPadding = static_cast<uint8_t>(
            (BLOCK_SIZE - (rowLen * sizeof(T)) % BLOCK_SIZE) % BLOCK_SIZE / sizeof(T));
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, rightPadding, 0};
        AscendC::DataCopyPad(logitsLocal, inputGMLogits[gmBase], copyParams, padParams);

        uint32_t targetBase = progress * tileRowNum;
        for (uint32_t row = 0; row < processRowNum; row++) {
            targetLocal.SetValue(row, inputGMTarget.GetValue(targetBase + row));
        }

        inputQueueLogits.EnQue(logitsLocal);
        inputQueueTarget.EnQue(targetLocal);
    }

    __aicore__ inline void CopyOutNormal(int32_t progress)
    {
        AscendC::LocalTensor<T> gradLocal = outputQueueGrad.DeQue<T>();

        uint32_t gmBase = progress * tileRowNum * rowLen;

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(1),
                static_cast<uint32_t>(rowLen * sizeof(T)),
                0, 0, 0
            };
            AscendC::DataCopyPad(outputGMGrad[gmBase + row * rowLen],
                              gradLocal[row * rowLenAlign32], copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        outputQueueGrad.FreeTensor(gradLocal);
    }

    __aicore__ inline void ComputeNormal(int32_t progress)
    {
        AscendC::LocalTensor<T> logitsLocal = inputQueueLogits.DeQue<T>();
        AscendC::LocalTensor<TargetT> targetLocal = inputQueueTarget.DeQue<TargetT>();
        AscendC::LocalTensor<T> gradLocal = outputQueueGrad.AllocTensor<T>();

        AscendC::LocalTensor<T> reduceTmp = reduceTmpBuf.Get<T>();
        AscendC::LocalTensor<T> workLocal = reduceWorkBuf.Get<T>();
        AscendC::LocalTensor<T> maxVals = maxBuf.Get<T>();
        AscendC::LocalTensor<T> sumVals = sumBuf.Get<T>();
        AscendC::LocalTensor<T> oneHotTmp = oneHotBuf.Get<T>();

        // Step 1: 计算每行的最大值
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::ReduceMax<T>(reduceTmp, logitsLocal[row * rowLenAlign32], workLocal, rowLen, false);
            maxVals.SetValue(row, reduceTmp.GetValue(0));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 2: 减去最大值并计算 exp（逐行避免行间gap干扰）
        for (uint32_t row = 0; row < processRowNum; row++) {
            float maxVal_t = static_cast<float>(maxVals.GetValue(row));
            T maxVal = static_cast<T>(-maxVal_t);
            AscendC::Adds(logitsLocal[row * rowLenAlign32], logitsLocal[row * rowLenAlign32], maxVal, rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Exp(logitsLocal[row * rowLenAlign32], logitsLocal[row * rowLenAlign32], rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 3: 计算每行的 exp 和
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::ReduceSum<T>(reduceTmp, logitsLocal[row * rowLenAlign32], workLocal, rowLen);
            sumVals.SetValue(row, reduceTmp.GetValue(0));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 4: 归一化得到 softmax
        for (uint32_t row = 0; row < processRowNum; row++) {
            float sumVal = static_cast<float>(sumVals.GetValue(row));
            AscendC::Muls(logitsLocal[row * rowLenAlign32], logitsLocal[row * rowLenAlign32],
                          static_cast<T>(1.0f / sumVal), rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 5: 构建 one-hot 并计算梯度
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Duplicate(oneHotTmp[row * rowLenAlign32], (T)0.0, rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            TargetT targetIdx = targetLocal.GetValue(row);
            oneHotTmp.SetValue(row * rowLenAlign32 + targetIdx, static_cast<T>(1.0));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Sub(gradLocal[row * rowLenAlign32],
                         logitsLocal[row * rowLenAlign32],
                         oneHotTmp[row * rowLenAlign32], rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        outputQueueGrad.EnQue(gradLocal);
        inputQueueTarget.FreeTensor(targetLocal);
        inputQueueLogits.FreeTensor(logitsLocal);
    }

    // ========================================================================
    // 列分割模式
    // ========================================================================
    __aicore__ inline void ProcessColSplit()
    {
        int32_t loopCount = this->tileNum;
        this->processRowNum = this->tileRowNum;

        for (int32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1 && this->tailRowNum > 0) {
                this->processRowNum = this->tailRowNum;
            }

            // Phase 1: 扫描求每行的 global_max
            float globalMax[MAX_ROWS_PER_TILE];
            for (uint32_t r = 0; r < processRowNum; r++) globalMax[r] = -1e30f;

            for (uint32_t pass = 0; pass < numColPasses; pass++) {
                float localMax[MAX_ROWS_PER_TILE];
                CopyInColSplit(i, pass);
                ComputeColSplitScan(i, pass, localMax);
                for (uint32_t r = 0; r < processRowNum; r++) {
                    if (localMax[r] > globalMax[r]) globalMax[r] = localMax[r];
                }
            }

            // Phase 2: 用 global_max 扫描求 global_sum
            float globalSum[MAX_ROWS_PER_TILE];
            for (uint32_t r = 0; r < processRowNum; r++) globalSum[r] = 0.0f;

            for (uint32_t pass = 0; pass < numColPasses; pass++) {
                float localSum[MAX_ROWS_PER_TILE];
                CopyInColSplit(i, pass);
                ComputeColSplitSum(i, pass, globalMax, localSum);
                for (uint32_t r = 0; r < processRowNum; r++) {
                    globalSum[r] += localSum[r];
                }
            }

            // Phase 3: 用 global_max 和 global_sum 计算 softmax + grad
            for (uint32_t pass = 0; pass < numColPasses; pass++) {
                CopyInColSplit(i, pass);
                ComputeColSplitGrad(i, pass, globalMax, globalSum);
                CopyOutColSplit(i, pass);
            }
        }
    }

    __aicore__ inline void CopyInColSplit(int32_t progress, uint32_t pass)
    {
        AscendC::LocalTensor<T> logitsLocal = inputQueueLogits.AllocTensor<T>();
        AscendC::LocalTensor<TargetT> targetLocal = inputQueueTarget.AllocTensor<TargetT>();

        uint32_t colStart = pass * tileCols;
        uint32_t actualDataLen = tileCols;
        if ((colStart + tileCols) > rowLen) {
            actualDataLen = rowLen - colStart;
        }

        uint32_t copyLen = ((actualDataLen * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / sizeof(T);

        uint32_t gmBase = progress * tileRowNum * rowLen;
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::DataCopyExtParams cpLogits = {1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> ppLogits = {false, 0, 0, 0};
            AscendC::DataCopyPad(logitsLocal[row * tileColsAlign32],
                                  inputGMLogits[gmBase + row * rowLen + colStart], cpLogits, ppLogits);
        }

        // batch copy target indices from GM → UB (avoid GlobalTensor::GetValue)
        uint32_t targetBase = progress * tileRowNum;
        AscendC::DataCopyExtParams cpTarget = {1, static_cast<uint32_t>(processRowNum * sizeof(TargetT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<TargetT> ppTarget = {false, 0, 0, 0};
        AscendC::DataCopyPad(targetLocal, inputGMTarget[targetBase], cpTarget, ppTarget);

        inputQueueLogits.EnQue(logitsLocal);
        inputQueueTarget.EnQue(targetLocal);
    }

    __aicore__ inline void CopyOutColSplit(int32_t progress, uint32_t pass)
    {
        AscendC::LocalTensor<T> gradLocal = outputQueueGrad.DeQue<T>();

        uint32_t colStart = pass * tileCols;
        uint32_t actualDataLen = tileCols;
        if ((colStart + tileCols) > rowLen) {
            actualDataLen = rowLen - colStart;
        }

        uint32_t copyLen = ((actualDataLen * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / sizeof(T);

        uint32_t gmBase = progress * tileRowNum * rowLen;
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::DataCopyExtParams cpOut = {1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(outputGMGrad[gmBase + row * rowLen + colStart],
                                  gradLocal[row * tileColsAlign32], cpOut);
        }

        outputQueueGrad.FreeTensor(gradLocal);
    }

    __aicore__ inline void ComputeColSplitScan(int32_t progress, uint32_t pass, float* localMax)
    {
        AscendC::LocalTensor<T> logitsLocal = inputQueueLogits.DeQue<T>();
        AscendC::LocalTensor<TargetT> targetLocal = inputQueueTarget.DeQue<TargetT>();

        AscendC::LocalTensor<T> reduceTmp = reduceTmpBuf.Get<T>();
        AscendC::LocalTensor<T> workLocal = reduceWorkBuf.Get<T>();

        uint32_t colStart = pass * tileCols;
        uint32_t actualDataLen = tileCols;
        if ((colStart + tileCols) > rowLen) {
            actualDataLen = rowLen - colStart;
        }
        if (actualDataLen != tileColsAlign32) {
            T NEG_LARGE = static_cast<T>(-1e30);
            uint32_t padLen = tileColsAlign32 - actualDataLen;
            for (uint32_t row = 0; row < processRowNum; row++) {
                AscendC::Duplicate(logitsLocal[row * tileColsAlign32 + actualDataLen], NEG_LARGE, padLen);
            }
        }

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::ReduceMax<T>(reduceTmp, logitsLocal[row * tileColsAlign32], workLocal, tileColsAlign32, false);
            localMax[row] = static_cast<float>(reduceTmp.GetValue(0));
        }
        AscendC::PipeBarrier<PIPE_V>();

        inputQueueTarget.FreeTensor(targetLocal);
        inputQueueLogits.FreeTensor(logitsLocal);
    }

    __aicore__ inline void ComputeColSplitSum(int32_t progress, uint32_t pass,
                                               const float* globalMax, float* localSum)
    {
        AscendC::LocalTensor<T> logitsLocal = inputQueueLogits.DeQue<T>();
        AscendC::LocalTensor<TargetT> targetLocal = inputQueueTarget.DeQue<TargetT>();

        AscendC::LocalTensor<T> reduceTmp = reduceTmpBuf.Get<T>();
        AscendC::LocalTensor<T> workLocal = reduceWorkBuf.Get<T>();

        uint32_t colStart = pass * tileCols;
        uint32_t actualDataLen = tileCols;
        if ((colStart + tileCols) > rowLen) {
            actualDataLen = rowLen - colStart;
        }
        if (actualDataLen != tileColsAlign32) {
            T NEG_LARGE = static_cast<T>(-1e30);
            uint32_t padLen = tileColsAlign32 - actualDataLen;
            for (uint32_t row = 0; row < processRowNum; row++) {
                AscendC::Duplicate(logitsLocal[row * tileColsAlign32 + actualDataLen], NEG_LARGE, padLen);
            }
        }

        for (uint32_t row = 0; row < processRowNum; row++) {
            T negMax = static_cast<T>(-globalMax[row]);
            AscendC::Adds(logitsLocal[row * tileColsAlign32], logitsLocal[row * tileColsAlign32], negMax, tileColsAlign32);
        }
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t totalAlign = processRowNum * tileColsAlign32;
        AscendC::Exp(logitsLocal, logitsLocal, totalAlign);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::ReduceSum<T>(reduceTmp, logitsLocal[row * tileColsAlign32], workLocal, tileColsAlign32);
            localSum[row] = static_cast<float>(reduceTmp.GetValue(0));
        }
        AscendC::PipeBarrier<PIPE_V>();

        inputQueueTarget.FreeTensor(targetLocal);
        inputQueueLogits.FreeTensor(logitsLocal);
    }

    __aicore__ inline void ComputeColSplitGrad(int32_t progress, uint32_t pass,
                                                  const float* globalMax, const float* globalSum)
    {
        AscendC::LocalTensor<T> logitsLocal = inputQueueLogits.DeQue<T>();
        AscendC::LocalTensor<TargetT> targetLocal = inputQueueTarget.DeQue<TargetT>();

        AscendC::LocalTensor<T> gradLocal = outputQueueGrad.AllocTensor<T>();
        AscendC::LocalTensor<T> oneHotTmp = oneHotBuf.Get<T>();

        uint32_t colStart = pass * tileCols;
        uint32_t totalAlign = processRowNum * tileColsAlign32;

        for (uint32_t row = 0; row < processRowNum; row++) {
            T negMax = static_cast<T>(-globalMax[row]);
            AscendC::Adds(logitsLocal[row * tileColsAlign32], logitsLocal[row * tileColsAlign32], negMax, tileColsAlign32);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(logitsLocal, logitsLocal, totalAlign);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Muls(logitsLocal[row * tileColsAlign32], logitsLocal[row * tileColsAlign32],
                          static_cast<T>(1.0f / globalSum[row]), tileColsAlign32);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Duplicate(oneHotTmp, (T)0.0, totalAlign);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            TargetT targetIdx = targetLocal.GetValue(row);
            uint32_t targetCol = static_cast<uint32_t>(targetIdx);
            if (targetCol >= colStart && targetCol < colStart + tileCols) {
                oneHotTmp.SetValue(row * tileColsAlign32 + (targetCol - colStart), static_cast<T>(1.0));
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub(gradLocal, logitsLocal, oneHotTmp, totalAlign);
        AscendC::PipeBarrier<PIPE_V>();

        outputQueueGrad.EnQue(gradLocal);
        inputQueueTarget.FreeTensor(targetLocal);
        inputQueueLogits.FreeTensor(logitsLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inputQueueLogits;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inputQueueTarget;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outputQueueGrad;

    AscendC::GlobalTensor<T> inputGMLogits;
    AscendC::GlobalTensor<TargetT> inputGMTarget;
    AscendC::GlobalTensor<T> outputGMGrad;

    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oneHotBuf;

    uint32_t rowLen = 0;
    uint32_t rowLenAlign32 = 0;
    uint32_t totalRows = 0;
    uint32_t tileRowNum = 0;
    uint32_t coreRowNum = 0;
    uint32_t tileNum = 0;
    uint32_t tailRowNum = 0;
    uint32_t processRowNum = 0;
    uint32_t typeBytes = 0;

    uint32_t colSplitMode = 0;
    uint32_t tileCols = 0;
    uint32_t numColPasses = 0;
    uint32_t tileColsAlign32 = 0;
};

// bfloat16_t 特化：logits/grad IO用bf16，target用int32_t，计算用float32
template <>
class KernelCrossEntropyGrad<bfloat16_t, int32_t> {
public:
    __aicore__ inline KernelCrossEntropyGrad() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR target, GM_ADDR gradOutput,
                                 GM_ADDR workspace, const CrossEntropyGradTilingData* tilingData)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->rowLen = tilingData->rowLen;
        this->totalRows = tilingData->totalRows;
        this->tileRowNum = tilingData->tileRowNum;
        this->colSplitMode = tilingData->colSplitMode;
        this->tileCols = tilingData->tileCols;
        this->numColPasses = tilingData->numColPasses;

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t bigCoreNum = tilingData->bigCoreNum;

        uint32_t startRow = 0;
        if (coreIdx < bigCoreNum) {
            this->coreRowNum = tilingData->bigCoreRowNum;
            this->tileNum = tilingData->bigCoreTileNum;
            this->tailRowNum = tilingData->bigTailRowNum;
            startRow = coreIdx * tilingData->bigCoreRowNum;
        } else {
            this->coreRowNum = tilingData->smallCoreRowNum;
            this->tileNum = tilingData->smallCoreTileNum;
            this->tailRowNum = tilingData->smallTailRowNum;
            startRow = bigCoreNum * tilingData->bigCoreRowNum +
                       (coreIdx - bigCoreNum) * tilingData->smallCoreRowNum;
        }

        inputGMLogits.SetGlobalBuffer((__gm__ bfloat16_t*)logits + startRow * rowLen, this->coreRowNum * rowLen);
        inputGMTarget.SetGlobalBuffer((__gm__ int32_t*)target + startRow, this->coreRowNum);
        outputGMGrad.SetGlobalBuffer((__gm__ bfloat16_t*)gradOutput + startRow * rowLen, this->coreRowNum * rowLen);

        this->rowLenAlign32 = ((rowLen * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2;
        this->rowLenAlign32F32 = ((rowLen * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 4;

        if (colSplitMode) {
            this->tileColsAlign32 = ((tileCols * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2;
            this->tileColsAlign32F32 = ((tileCols * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 4;

            uint32_t tileDataSizeBf16 = tileRowNum * tileColsAlign32;
            uint32_t tileDataSizeF32 = tileRowNum * tileColsAlign32F32;

            pipe.InitBuffer(inputQueueLogits, BUFFER_NUM, tileDataSizeBf16 * sizeof(bfloat16_t));
            pipe.InitBuffer(outputQueueGrad, BUFFER_NUM, tileDataSizeBf16 * sizeof(bfloat16_t));

            uint32_t targetAlign32 = ((tileRowNum * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(inputQueueTarget, BUFFER_NUM, targetAlign32);

            uint32_t elementsPerBlock = BLOCK_SIZE / 4;
            uint32_t elementsPerRepeat = 256 / 4;
            uint32_t firstMaxRepeat = (tileColsAlign32F32 + elementsPerRepeat - 1) / elementsPerRepeat;
            uint32_t iter1OutputCount = firstMaxRepeat * 2;
            uint32_t workLocalElements = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
            uint32_t workLocalSize = ((workLocalElements * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

            pipe.InitBuffer(reduceTmpBuf, workLocalSize);
            pipe.InitBuffer(reduceWorkBuf, workLocalSize);

            uint32_t maxBufSize = ((tileRowNum * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(maxBuf, maxBufSize);
            pipe.InitBuffer(sumBuf, maxBufSize);
            pipe.InitBuffer(logitsFloatBuf, tileDataSizeF32 * sizeof(float));
            pipe.InitBuffer(gradFloatBuf, tileDataSizeF32 * sizeof(float));
        } else {
            uint32_t tileDataSizeBf16 = tileRowNum * rowLenAlign32;
            uint32_t tileDataSizeF32 = tileRowNum * rowLenAlign32F32;

            pipe.InitBuffer(inputQueueLogits, BUFFER_NUM, tileDataSizeBf16 * sizeof(bfloat16_t));
            pipe.InitBuffer(outputQueueGrad, BUFFER_NUM, tileDataSizeBf16 * sizeof(bfloat16_t));

            uint32_t targetAlign32 = ((tileRowNum * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(inputQueueTarget, BUFFER_NUM, targetAlign32);

            uint32_t elementsPerBlock = BLOCK_SIZE / 4;
            uint32_t elementsPerRepeat = 256 / 4;
            uint32_t firstMaxRepeat = (rowLenAlign32F32 + elementsPerRepeat - 1) / elementsPerRepeat;
            uint32_t iter1OutputCount = firstMaxRepeat * 2;
            uint32_t workLocalElements = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
            uint32_t workLocalSize = ((workLocalElements * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

            pipe.InitBuffer(reduceTmpBuf, workLocalSize);
            pipe.InitBuffer(reduceWorkBuf, workLocalSize);

            uint32_t maxBufSize = ((tileRowNum * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            pipe.InitBuffer(maxBuf, maxBufSize);
            pipe.InitBuffer(sumBuf, maxBufSize);
            pipe.InitBuffer(logitsFloatBuf, tileDataSizeF32 * sizeof(float));
            pipe.InitBuffer(gradFloatBuf, tileDataSizeF32 * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->coreRowNum == 0) return;
        if (colSplitMode) ProcessColSplit();
        else ProcessNormal();
    }

private:
    // ============== 正常模式 ==============
    __aicore__ inline void ProcessNormal()
    {
        int32_t loopCount = this->tileNum;
        this->processRowNum = this->tileRowNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1 && this->tailRowNum > 0)
                this->processRowNum = this->tailRowNum;
            CopyInNormal(i);
            ComputeNormal(i);
            CopyOutNormal(i);
        }
    }

    __aicore__ inline void CopyInNormal(int32_t progress)
    {
        AscendC::LocalTensor<bfloat16_t> logitsLocal = inputQueueLogits.AllocTensor<bfloat16_t>();
        AscendC::LocalTensor<int32_t> targetLocal = inputQueueTarget.AllocTensor<int32_t>();

        uint32_t gmBase = progress * tileRowNum * rowLen;
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(processRowNum),
            static_cast<uint32_t>(rowLen * 2),
            0, 0, 0
        };
        uint8_t rightPadding = static_cast<uint8_t>((BLOCK_SIZE - (rowLen * 2) % BLOCK_SIZE) % BLOCK_SIZE / 2);
        AscendC::DataCopyPadExtParams<bfloat16_t> padParams{ true, 0, rightPadding, 0 };
        AscendC::DataCopyPad(logitsLocal, inputGMLogits[gmBase], copyParams, padParams);

        uint32_t targetBase = progress * tileRowNum;
        for (uint32_t row = 0; row < processRowNum; row++)
            targetLocal.SetValue(row, inputGMTarget.GetValue(targetBase + row));

        inputQueueLogits.EnQue(logitsLocal);
        inputQueueTarget.EnQue(targetLocal);
    }

    __aicore__ inline void CopyOutNormal(int32_t progress)
    {
        AscendC::LocalTensor<bfloat16_t> gradLocal = outputQueueGrad.DeQue<bfloat16_t>();
        uint32_t gmBase = progress * tileRowNum * rowLen;
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::DataCopyExtParams copyParams{ 1, static_cast<uint32_t>(rowLen * 2), 0, 0, 0 };
            AscendC::DataCopyPad(outputGMGrad[gmBase + row * rowLen],
                              gradLocal[row * rowLenAlign32], copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        outputQueueGrad.FreeTensor(gradLocal);
    }

    __aicore__ inline void ComputeNormal(int32_t progress)
    {
        AscendC::LocalTensor<bfloat16_t> logitsLocal = inputQueueLogits.DeQue<bfloat16_t>();
        AscendC::LocalTensor<int32_t> targetLocal = inputQueueTarget.DeQue<int32_t>();
        AscendC::LocalTensor<bfloat16_t> gradLocal = outputQueueGrad.AllocTensor<bfloat16_t>();

        AscendC::LocalTensor<float> logitsFloat = logitsFloatBuf.Get<float>();
        AscendC::LocalTensor<float> gradFloat = gradFloatBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> workLocal = reduceWorkBuf.Get<float>();
        AscendC::LocalTensor<float> maxVals = maxBuf.Get<float>();
        AscendC::LocalTensor<float> sumVals = sumBuf.Get<float>();

        // 逐行 Cast bf16 logits → float
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Cast(logitsFloat[row * rowLenAlign32F32], logitsLocal[row * rowLenAlign32],
                           AscendC::RoundMode::CAST_NONE, rowLen);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 1: 计算每行的最大值
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::ReduceMax<float>(reduceTmp, logitsFloat[row * rowLenAlign32F32], workLocal, rowLen, false);
            maxVals.SetValue(row, reduceTmp.GetValue(0));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 2: 减去最大值并计算 exp
        for (uint32_t row = 0; row < processRowNum; row++) {
            float maxVal = maxVals.GetValue(row);
            AscendC::Adds(logitsFloat[row * rowLenAlign32F32], logitsFloat[row * rowLenAlign32F32], -maxVal, rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Exp(logitsFloat[row * rowLenAlign32F32], logitsFloat[row * rowLenAlign32F32], rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 3: 计算每行的 exp 和
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::ReduceSum<float>(reduceTmp, logitsFloat[row * rowLenAlign32F32], workLocal, rowLen);
            sumVals.SetValue(row, reduceTmp.GetValue(0));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 4: 归一化得到 softmax
        for (uint32_t row = 0; row < processRowNum; row++) {
            float sumVal = sumVals.GetValue(row);
            AscendC::Muls(logitsFloat[row * rowLenAlign32F32], logitsFloat[row * rowLenAlign32F32], 1.0f / sumVal, rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 5: 构建 one-hot 并计算梯度 (float)
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Duplicate(gradFloat[row * rowLenAlign32F32], 0.0f, rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            int32_t targetIdx = targetLocal.GetValue(row);
            gradFloat.SetValue(row * rowLenAlign32F32 + targetIdx, 1.0f);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Sub(gradFloat[row * rowLenAlign32F32],
                         logitsFloat[row * rowLenAlign32F32],
                         gradFloat[row * rowLenAlign32F32], rowLen);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();

        // 逐行 Cast float grad → bf16
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Cast(gradLocal[row * rowLenAlign32], gradFloat[row * rowLenAlign32F32],
                           AscendC::RoundMode::CAST_RINT, rowLen);
        }

        outputQueueGrad.EnQue(gradLocal);
        inputQueueTarget.FreeTensor(targetLocal);
        inputQueueLogits.FreeTensor(logitsLocal);
    }

    // ============== 列分割模式 ==============
    __aicore__ inline void ProcessColSplit()
    {
        int32_t loopCount = this->tileNum;
        this->processRowNum = this->tileRowNum;

        for (int32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1 && this->tailRowNum > 0)
                this->processRowNum = this->tailRowNum;

            float globalMax[MAX_ROWS_PER_TILE];
            float globalSum[MAX_ROWS_PER_TILE];
            for (uint32_t r = 0; r < processRowNum; r++) { globalMax[r] = -1e30f; globalSum[r] = 0.0f; }

            for (uint32_t pass = 0; pass < numColPasses; pass++) {
                float localMax[MAX_ROWS_PER_TILE];
                CopyInColSplit(i, pass);
                ComputeColSplitScan(i, pass, localMax);
                for (uint32_t r = 0; r < processRowNum; r++)
                    if (localMax[r] > globalMax[r]) globalMax[r] = localMax[r];
            }
            for (uint32_t pass = 0; pass < numColPasses; pass++) {
                float localSum[MAX_ROWS_PER_TILE];
                CopyInColSplit(i, pass);
                ComputeColSplitSum(i, pass, globalMax, localSum);
                for (uint32_t r = 0; r < processRowNum; r++) globalSum[r] += localSum[r];
            }
            for (uint32_t pass = 0; pass < numColPasses; pass++) {
                CopyInColSplit(i, pass);
                ComputeColSplitGrad(i, pass, globalMax, globalSum);
                CopyOutColSplit(i, pass);
            }
        }
    }

    __aicore__ inline void CopyInColSplit(int32_t progress, uint32_t pass)
    {
        AscendC::LocalTensor<bfloat16_t> logitsLocal = inputQueueLogits.AllocTensor<bfloat16_t>();
        AscendC::LocalTensor<int32_t> targetLocal = inputQueueTarget.AllocTensor<int32_t>();

        uint32_t colStart = pass * tileCols;
        uint32_t actualDataLen = (colStart + tileCols > rowLen) ? rowLen - colStart : tileCols;
        uint32_t copyLen = ((actualDataLen * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2;

        uint32_t gmBase = progress * tileRowNum * rowLen;
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::DataCopyExtParams cpLogits = {1, static_cast<uint32_t>(copyLen * sizeof(bfloat16_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<bfloat16_t> ppLogits = {false, 0, 0, 0};
            AscendC::DataCopyPad(logitsLocal[row * tileColsAlign32],
                                  inputGMLogits[gmBase + row * rowLen + colStart], cpLogits, ppLogits);
        }

        // batch copy target indices from GM → UB (avoid GlobalTensor::GetValue)
        uint32_t targetBase = progress * tileRowNum;
        AscendC::DataCopyExtParams cpTarget = {1, static_cast<uint32_t>(processRowNum * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> ppTarget = {false, 0, 0, 0};
        AscendC::DataCopyPad(targetLocal, inputGMTarget[targetBase], cpTarget, ppTarget);

        inputQueueLogits.EnQue(logitsLocal);
        inputQueueTarget.EnQue(targetLocal);
    }

    __aicore__ inline void CopyOutColSplit(int32_t progress, uint32_t pass)
    {
        AscendC::LocalTensor<bfloat16_t> gradLocal = outputQueueGrad.DeQue<bfloat16_t>();

        uint32_t colStart = pass * tileCols;
        uint32_t actualDataLen = (colStart + tileCols > rowLen) ? rowLen - colStart : tileCols;
        uint32_t copyLen = ((actualDataLen * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / 2;

        uint32_t gmBase = progress * tileRowNum * rowLen;
        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::DataCopyExtParams cpOut = {1, static_cast<uint32_t>(copyLen * sizeof(bfloat16_t)), 0, 0, 0};
            AscendC::DataCopyPad(outputGMGrad[gmBase + row * rowLen + colStart],
                                  gradLocal[row * tileColsAlign32], cpOut);
        }

        outputQueueGrad.FreeTensor(gradLocal);
    }

    __aicore__ inline void ComputeColSplitScan(int32_t progress, uint32_t pass, float* localMax)
    {
        AscendC::LocalTensor<bfloat16_t> logitsLocal = inputQueueLogits.DeQue<bfloat16_t>();
        AscendC::LocalTensor<int32_t> targetLocal = inputQueueTarget.DeQue<int32_t>();

        AscendC::LocalTensor<float> logitsFloat = logitsFloatBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> workLocal = reduceWorkBuf.Get<float>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Cast(logitsFloat[row * tileColsAlign32F32], logitsLocal[row * tileColsAlign32],
                           AscendC::RoundMode::CAST_NONE, tileColsAlign32);
        }

        uint32_t colStart = pass * tileCols;
        uint32_t actualDataLen = (colStart + tileCols > rowLen) ? rowLen - colStart : tileCols;
        if (actualDataLen != tileColsAlign32F32) {
            float NEG_LARGE = -1e30f;
            uint32_t padLen = tileColsAlign32F32 - actualDataLen;
            for (uint32_t row = 0; row < processRowNum; row++)
                AscendC::Duplicate(logitsFloat[row * tileColsAlign32F32 + actualDataLen], NEG_LARGE, padLen);
        }

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::ReduceMax<float>(reduceTmp, logitsFloat[row * tileColsAlign32F32], workLocal, tileColsAlign32F32, false);
            localMax[row] = reduceTmp.GetValue(0);
        }
        AscendC::PipeBarrier<PIPE_V>();

        inputQueueTarget.FreeTensor(targetLocal);
        inputQueueLogits.FreeTensor(logitsLocal);
    }

    __aicore__ inline void ComputeColSplitSum(int32_t progress, uint32_t pass,
                                               const float* globalMax, float* localSum)
    {
        AscendC::LocalTensor<bfloat16_t> logitsLocal = inputQueueLogits.DeQue<bfloat16_t>();
        AscendC::LocalTensor<int32_t> targetLocal = inputQueueTarget.DeQue<int32_t>();

        AscendC::LocalTensor<float> logitsFloat = logitsFloatBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> workLocal = reduceWorkBuf.Get<float>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Cast(logitsFloat[row * tileColsAlign32F32], logitsLocal[row * tileColsAlign32],
                           AscendC::RoundMode::CAST_NONE, tileColsAlign32);
        }

        uint32_t colStart = pass * tileCols;
        uint32_t actualDataLen = (colStart + tileCols > rowLen) ? rowLen - colStart : tileCols;
        if (actualDataLen != tileColsAlign32F32) {
            float NEG_LARGE = -1e30f;
            uint32_t padLen = tileColsAlign32F32 - actualDataLen;
            for (uint32_t row = 0; row < processRowNum; row++)
                AscendC::Duplicate(logitsFloat[row * tileColsAlign32F32 + actualDataLen], NEG_LARGE, padLen);
        }

        for (uint32_t row = 0; row < processRowNum; row++)
            AscendC::Adds(logitsFloat[row * tileColsAlign32F32], logitsFloat[row * tileColsAlign32F32], -globalMax[row], tileColsAlign32F32);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t totalAlign = processRowNum * tileColsAlign32F32;
        AscendC::Exp(logitsFloat, logitsFloat, totalAlign);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::ReduceSum<float>(reduceTmp, logitsFloat[row * tileColsAlign32F32], workLocal, tileColsAlign32F32);
            localSum[row] = reduceTmp.GetValue(0);
        }
        AscendC::PipeBarrier<PIPE_V>();

        inputQueueTarget.FreeTensor(targetLocal);
        inputQueueLogits.FreeTensor(logitsLocal);
    }

    __aicore__ inline void ComputeColSplitGrad(int32_t progress, uint32_t pass,
                                                  const float* globalMax, const float* globalSum)
    {
        AscendC::LocalTensor<bfloat16_t> logitsLocal = inputQueueLogits.DeQue<bfloat16_t>();
        AscendC::LocalTensor<int32_t> targetLocal = inputQueueTarget.DeQue<int32_t>();
        AscendC::LocalTensor<bfloat16_t> gradLocal = outputQueueGrad.AllocTensor<bfloat16_t>();

        AscendC::LocalTensor<float> logitsFloat = logitsFloatBuf.Get<float>();
        AscendC::LocalTensor<float> gradFloat = gradFloatBuf.Get<float>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Cast(logitsFloat[row * tileColsAlign32F32], logitsLocal[row * tileColsAlign32],
                           AscendC::RoundMode::CAST_NONE, tileColsAlign32);
        }

        uint32_t colStart = pass * tileCols;
        uint32_t totalAlign = processRowNum * tileColsAlign32F32;

        for (uint32_t row = 0; row < processRowNum; row++)
            AscendC::Adds(logitsFloat[row * tileColsAlign32F32], logitsFloat[row * tileColsAlign32F32], -globalMax[row], tileColsAlign32F32);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(logitsFloat, logitsFloat, totalAlign);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++)
            AscendC::Muls(logitsFloat[row * tileColsAlign32F32], logitsFloat[row * tileColsAlign32F32], 1.0f / globalSum[row], tileColsAlign32F32);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Duplicate(gradFloat, 0.0f, totalAlign);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < processRowNum; row++) {
            int32_t targetIdx = targetLocal.GetValue(row);
            uint32_t targetCol = static_cast<uint32_t>(targetIdx);
            if (targetCol >= colStart && targetCol < colStart + tileCols)
                gradFloat.SetValue(row * tileColsAlign32F32 + (targetCol - colStart), 1.0f);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub(gradFloat, logitsFloat, gradFloat, totalAlign);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < processRowNum; row++) {
            AscendC::Cast(gradLocal[row * tileColsAlign32], gradFloat[row * tileColsAlign32F32],
                           AscendC::RoundMode::CAST_RINT, tileCols);
        }

        outputQueueGrad.EnQue(gradLocal);
        inputQueueTarget.FreeTensor(targetLocal);
        inputQueueLogits.FreeTensor(logitsLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inputQueueLogits;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inputQueueTarget;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outputQueueGrad;

    AscendC::GlobalTensor<bfloat16_t> inputGMLogits;
    AscendC::GlobalTensor<int32_t> inputGMTarget;
    AscendC::GlobalTensor<bfloat16_t> outputGMGrad;

    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> logitsFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gradFloatBuf;

    uint32_t rowLen = 0;
    uint32_t rowLenAlign32 = 0;
    uint32_t rowLenAlign32F32 = 0;
    uint32_t totalRows = 0;
    uint32_t tileRowNum = 0;
    uint32_t coreRowNum = 0;
    uint32_t tileNum = 0;
    uint32_t tailRowNum = 0;
    uint32_t processRowNum = 0;
    uint32_t colSplitMode = 0;
    uint32_t tileCols = 0;
    uint32_t numColPasses = 0;
    uint32_t tileColsAlign32 = 0;
    uint32_t tileColsAlign32F32 = 0;
};

#endif // CROSS_ENTROPY_GRAD_H_
