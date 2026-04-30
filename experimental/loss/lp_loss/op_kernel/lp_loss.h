/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LP_LOSS_H
#define LP_LOSS_H

#include "lp_loss_tiling_data.h"
#include "kernel_operator.h"
#include "lp_loss_tiling_key.h"

namespace MyLpLoss {
using namespace AscendC;

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM, uint32_t IS_REDUCE>
class KernelLpLoss {
public:
    __aicore__ inline KernelLpLoss()
    {}

    __aicore__ inline void InitNone(
        GM_ADDR predict, GM_ADDR label, GM_ADDR y, const LpLossTilingData* tilingData, AscendC::TPipe* pipeIn)
    {

        InitCommon(predict, label, tilingData, pipeIn);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + this->globalBufferIndex, this->coreDataNum);

        pipe->InitBuffer(inQueuePredict, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe->InitBuffer(inQueueLabel, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        if constexpr (!AscendC::IsSameType<TYPE_X, float>::value) {
            pipe->InitBuffer(calcBufA, this->tileDataNum * sizeof(float) * 2);
        }
    }

    __aicore__ inline void InitReduce(
        GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR usrWorkspace, const LpLossTilingData* tilingData,
        AscendC::TPipe* pipeIn)
    {


        InitCommon(predict, label, tilingData, pipeIn);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y, 1);

        const uint64_t syncAllBytes = this->coreNum * this->blockSize;
        __gm__ uint8_t* usrWorkspaceBytes = (__gm__ uint8_t*)usrWorkspace;
        __gm__ float* reduceWorkspace = (__gm__ float*)(usrWorkspaceBytes + syncAllBytes);
        usrWorkspaceBase.SetGlobalBuffer(reduceWorkspace, this->coreNum);

        this->totalReduceCount = static_cast<uint32_t>(this->coreNum);
        this->totalReduceAlignedCount =
            static_cast<uint32_t>(AlignUpElems(this->totalReduceCount, this->blockSize / sizeof(float)));
        pipe->InitBuffer(reduceAccumBuf, sizeof(float) * (this->blockSize / sizeof(float)));
        pipe->InitBuffer(crossCoreReduceBuf, sizeof(float) * this->totalReduceAlignedCount);
        pipe->InitBuffer(reductionOutBuf, this->blockSize);

        uint64_t alignElemsX = this->blockSize / sizeof(TYPE_X);
        uint64_t queueAlignSizeX = AlignUpElems(this->tileDataNum, alignElemsX) * sizeof(TYPE_X);
        pipe->InitBuffer(inQueuePredict, BUFFER_NUM, queueAlignSizeX);
        pipe->InitBuffer(inQueueLabel, BUFFER_NUM, queueAlignSizeX);

        uint64_t alignElemsF = this->blockSize / sizeof(float);
        uint64_t fAlignSize = AlignUpElems(this->tileDataNum, alignElemsF) * sizeof(float);
        pipe->InitBuffer(calcBufA, fAlignSize);
        if constexpr (!AscendC::IsSameType<TYPE_X, float>::value) {
            pipe->InitBuffer(calcBufB, fAlignSize);
        }
    }

    __aicore__ inline void SetProcessDataNum(int32_t index, int32_t loopCount)
    {
        if (index == loopCount - 1) {
            this->processDataNum = this->tailDataNum;
        } else {
            this->processDataNum = this->tileDataNum;
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (IS_REDUCE == 0) {
            ProcessNone();
        } else {
            ProcessReduce();
        }
    }

private:
    __aicore__ inline static uint64_t AlignUpElems(uint64_t x, uint64_t align)
    {
        if (align == 0) {
            return x;
        }

        return (x + align - 1) / align * align;
    }

    __aicore__ inline static bool IsBlockAlignedBytes(uint64_t elemNum, uint64_t elemBytes)
    {
        constexpr uint64_t blockSize = ONE_BLK_SIZE;
        return ((elemNum * elemBytes) % blockSize) == 0;
    }

    __aicore__ inline void InitCommon(
        GM_ADDR predict, GM_ADDR label, const LpLossTilingData* tilingData, AscendC::TPipe* pipeIn)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        ASSERT(tilingData != nullptr && "tilingData can not be nullptr!");
        ASSERT(pipeIn != nullptr && "pipeIn can not be nullptr!");

        this->pipe = pipeIn;
        this->reduction = tilingData->reduction;
        this->totalNum = tilingData->totalNum;
        this->coreNum = tilingData->coreNum;
        this->blockSize = tilingData->blockSize;
        this->meanFactor = tilingData->meanFactor;

        uint64_t blockIdx = AscendC::GetBlockIdx();
        this->globalBufferIndex = tilingData->bigCoreDataNum * blockIdx;
        this->tileDataNum = tilingData->tileDataNum;

        if (blockIdx < tilingData->tailBlockNum) {
            this->coreDataNum = tilingData->bigCoreDataNum;
            this->tileNum = tilingData->finalBigTileNum;
            this->tailDataNum = tilingData->bigTailDataNum;
        } else {
            this->coreDataNum = tilingData->smallCoreDataNum;
            this->tileNum = tilingData->finalSmallTileNum;
            this->tailDataNum = tilingData->smallTailDataNum;
            this->globalBufferIndex -=
                (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (blockIdx - tilingData->tailBlockNum);
        }

        if constexpr (IS_REDUCE != 0) {
            if (this->globalBufferIndex >= this->totalNum) {
                this->coreDataNum = 0;
            } else if (this->globalBufferIndex + this->coreDataNum > this->totalNum) {
                this->coreDataNum = this->totalNum - this->globalBufferIndex;
            }

            if (this->coreDataNum > 0) {
                this->tileNum = (this->coreDataNum + this->tileDataNum - 1) / this->tileDataNum;
                this->tailDataNum = this->coreDataNum - (this->tileNum - 1) * this->tileDataNum;
            } else {
                this->tileNum = 0;
                this->tailDataNum = 0;
            }
        }

        predictGm.SetGlobalBuffer((__gm__ TYPE_X*)predict + this->globalBufferIndex, this->coreDataNum);
        labelGm.SetGlobalBuffer((__gm__ TYPE_X*)label + this->globalBufferIndex, this->coreDataNum);
    }

    __aicore__ inline void ProcessNone()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyInNone(i);
            ComputeNone();
            CopyOut(i);
        }

        this->processDataNum = this->tailDataNum;
        CopyInNone(loopCount - 1);
        ComputeNone();
        CopyOut(loopCount - 1);
    }

    __aicore__ inline void ProcessReduce()
    {
        AscendC::LocalTensor<float> coreAccum = reduceAccumBuf.template Get<float>();
        coreAccum.SetValue(0, 0.0f);
        AscendC::PipeBarrier<PIPE_ALL>();

        int32_t loopCount = this->tileNum;
        if (this->coreDataNum > 0 && loopCount > 0) {
            SetProcessDataNum(0, loopCount);
            CopyInReduce(0);

            for (int32_t i = 1; i < loopCount; ++i) {
                SetProcessDataNum(i, loopCount);
                CopyInReduce(i);

                SetProcessDataNum(i - 1, loopCount);
                ComputeReduce();
            }

            SetProcessDataNum(loopCount - 1, loopCount);
            ComputeReduce();
        }

        GlobalReduction();
    }

    __aicore__ inline void AccumulateTileSum(AscendC::LocalTensor<float>& floatRes, uint64_t processLen)
    {
        AscendC::ReduceSum<float>(floatRes, floatRes, floatRes, processLen);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::LocalTensor<float> coreAccum = reduceAccumBuf.template Get<float>();
        AscendC::Add(coreAccum, coreAccum, floatRes, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInReduce(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> predictLocal = inQueuePredict.template AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> labelLocal = inQueueLabel.template AllocTensor<TYPE_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(TYPE_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(predictLocal, predictGm[progress * this->tileDataNum], copyParams, padParams);
        AscendC::DataCopyPad(labelLocal, labelGm[progress * this->tileDataNum], copyParams, padParams);
        inQueuePredict.EnQue(predictLocal);
        inQueueLabel.EnQue(labelLocal);
    }

    __aicore__ inline void CopyInNone(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> predictLocal = inQueuePredict.template AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> labelLocal = inQueueLabel.template AllocTensor<TYPE_X>();

        AscendC::DataCopy(predictLocal, predictGm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(labelLocal, labelGm[progress * this->tileDataNum], this->processDataNum);

        inQueuePredict.EnQue(predictLocal);
        inQueueLabel.EnQue(labelLocal);
    }

    __aicore__ inline void ComputeNone()
    {
        AscendC::LocalTensor<TYPE_X> predictLocal = inQueuePredict.template DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> labelLocal = inQueueLabel.template DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.template AllocTensor<TYPE_Y>();

        if constexpr (!AscendC::IsSameType<TYPE_X, float>::value) {
            AscendC::LocalTensor<float> tmpBuffer = calcBufA.template Get<float>();
            AscendC::LocalTensor<float> val1 = tmpBuffer;
            AscendC::LocalTensor<float> val2 = tmpBuffer[this->tileDataNum];

            AscendC::Cast(val1, predictLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(val2, labelLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            AscendC::Sub(val1, val1, val2, this->processDataNum);
            AscendC::Abs(val1, val1, this->processDataNum);

            if constexpr (AscendC::IsSameType<TYPE_Y, bfloat16_t>::value) {
                AscendC::Cast(yLocal, val1, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            } else {
                AscendC::Cast(yLocal, val1, AscendC::RoundMode::CAST_ROUND, this->processDataNum);
            }
        } else {
            AscendC::Sub(yLocal, predictLocal, labelLocal, this->processDataNum);
            AscendC::Abs(yLocal, yLocal, this->processDataNum);
        }

        outQueueY.template EnQue<TYPE_Y>(yLocal);
        inQueuePredict.FreeTensor(predictLocal);
        inQueueLabel.FreeTensor(labelLocal);
    }

    __aicore__ inline void ComputeReduce()
    {
        AscendC::LocalTensor<TYPE_X> predictLocal = inQueuePredict.template DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> labelLocal = inQueueLabel.template DeQue<TYPE_X>();

        AscendC::LocalTensor<float> floatRes;

        if constexpr (!AscendC::IsSameType<TYPE_X, float>::value) {
            AscendC::LocalTensor<float> val1 = calcBufA.template Get<float>();
            AscendC::LocalTensor<float> val2 = calcBufB.template Get<float>();

            AscendC::Cast(val1, predictLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(val2, labelLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Sub(val1, val1, val2, this->processDataNum);
            AscendC::Abs(val1, val1, this->processDataNum);
            floatRes = val1;
        } else {
            floatRes = calcBufA.template Get<float>();
            AscendC::Sub(floatRes, predictLocal, labelLocal, this->processDataNum);
            AscendC::Abs(floatRes, floatRes, this->processDataNum);
        }

        AccumulateTileSum(floatRes, this->processDataNum);

        inQueuePredict.FreeTensor(predictLocal);
        inQueueLabel.FreeTensor(labelLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.template DeQue<TYPE_Y>();

        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);

        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void GlobalReduction()
    {
        AscendC::LocalTensor<float> coreAccum = reduceAccumBuf.template Get<float>();
        AscendC::DataCopyExtParams wsWriteParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(usrWorkspaceBase[AscendC::GetBlockIdx()], coreAccum, wsWriteParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::SyncAll();

        if (AscendC::GetBlockIdx() == 0) {
            AscendC::LocalTensor<float> allCoreSums = crossCoreReduceBuf.template Get<float>();
            AscendC::DataCopyExtParams wsReadParams{
                1, static_cast<uint32_t>(this->totalReduceCount * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> wsReadPadParams{true, 0, 0, 0};
            AscendC::DataCopyPad(allCoreSums, usrWorkspaceBase, wsReadParams, wsReadPadParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::ReduceSum<float>(allCoreSums, allCoreSums, allCoreSums, this->totalReduceCount);
            AscendC::PipeBarrier<PIPE_ALL>();

            if (this->reduction == 1) { // reduction mode: Mean
                AscendC::Muls(allCoreSums, allCoreSums, this->meanFactor, 1);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            if constexpr (AscendC::IsSameType<TYPE_Y, float>::value) {
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(TYPE_Y)), 0, 0, 0};
                AscendC::DataCopyPad(yGm[0], allCoreSums, copyParams);
            } else {
                AscendC::LocalTensor<TYPE_Y> outVal = reductionOutBuf.template Get<TYPE_Y>();
                if constexpr (AscendC::IsSameType<TYPE_Y, bfloat16_t>::value) {
                    AscendC::Cast(outVal, allCoreSums, AscendC::RoundMode::CAST_RINT, 1);
                } else {
                    AscendC::Cast(outVal, allCoreSums, AscendC::RoundMode::CAST_ROUND, 1);
                }
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(TYPE_Y)), 0, 0, 0};
                AscendC::DataCopyPad(yGm[0], outVal, copyParams);
            }
        }
    }

private:
    AscendC::TPipe* pipe = nullptr;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuePredict, inQueueLabel;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // --- 两个分支共用的 TBuf 隔离机制 ---
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBufA;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBufB;

    // --- Reduce分支专用的Buffer ---
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceAccumBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> crossCoreReduceBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reductionOutBuf;

    AscendC::GlobalTensor<TYPE_X> predictGm, labelGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    AscendC::GlobalTensor<float> usrWorkspaceBase;

    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    uint64_t totalNum = 0;
    uint64_t coreNum = 0;
    uint64_t blockSize = 0;
    uint64_t globalBufferIndex = 0;
    uint32_t totalReduceCount = 0;
    uint32_t totalReduceAlignedCount = 0;
    float meanFactor = 0.0f;
    uint64_t reduction = 0;
};
} // namespace MyLpLoss
#endif // LP_LOSS_H
