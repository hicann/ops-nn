/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_v3_common.h
 * \brief
 */
#ifndef __BATCH_NORM_GRAD_V3_BASE_SPLITLOAD_H__
#define __BATCH_NORM_GRAD_V3_BASE_SPLITLOAD_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace BatchNormGradV3 {
using namespace AscendC;

template <typename T1, typename T2, typename T3, BNGV3SplitMode SPLIT_MODE = BNGV3SplitMode::R0_SPLIT_MODE>
class BatchNormGradV3SplitLoadBase : public BatchNormGradV3Base<T1, T2, T3> {
public:
    __aicore__ inline BatchNormGradV3SplitLoadBase()
    {}

    __aicore__ inline void Init(GMStruct gmStruct, TPipe* pipeIn)
    {
        this->InitSplitLoad(gmStruct, pipeIn);
    }

protected:
    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> reduceQueue;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> reduceInputQueue;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanQueue;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> varQueue;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> weightQueue;

    TBuf<QuePosition::VECCALC> meanBrcbQueue;
    TBuf<QuePosition::VECCALC> varBrcbQueue;
    TBuf<QuePosition::VECCALC> weightBrcbQueue;

    LocalTensor<float> dyTensor;
    LocalTensor<float> xTensor;
    LocalTensor<float> reduceUbTensor;
    LocalTensor<float> reduceInputUbTensor;

    LocalTensor<float> weightTensor;
    LocalTensor<float> varTensor;
    LocalTensor<float> meanTensor;

    LocalTensor<float> weightBrcbTensor;
    LocalTensor<float> varBrcbTensor;
    LocalTensor<float> meanBrcbTensor;

    float rVar;

    float var;
    float mean;
    float weight;
    int64_t bUbLoop;
    int64_t bUbBlock;
    int64_t b0UbBlock;
    int64_t bUbBlockTail;
    int64_t b0UbBlockTail;
    int64_t bUbR1Block;
    int64_t groupCore;
    int64_t groupBlockNum;
    int64_t groupBlockNumTail;
    int64_t morehalfChannel;
    int64_t groupBlockNumHalf;
    int64_t groupBlockNumHalfTail;
    int64_t coreNum;
    int64_t bUbTailLoop;
    int64_t b0UbTailBlockTail;
    int64_t bUbLoopHalf = 0;
    int64_t b0UbBlockTailHalf = 0;
    int64_t bUbTailLoopHalf = 0;
    int64_t b0UbTailBlockTailHalf = 0;
    int64_t moreMultiChannel = 0;
    int64_t bUbLoopMulti = 0;
    int64_t b0UbBlockTailMulti = 0;

    int64_t usedGroupCore;
    int64_t usedGroupBlockNum;
    int64_t usedGroupBlockNumTail;
    int64_t usedBUbLoop;
    int64_t usedB0UbBlockTail;
    int64_t usedB0UbTailBlockTail;

    int64_t bUbBlockAlign;
    int64_t bUbBlockAlignTail;

    int64_t wsReduceOffset;
    int64_t wsReduceInputOffset;

    int64_t bUbContinueLoop;
    int64_t bUbContinueTailLoop;
    int64_t bUbBlockContinue;
    int64_t bUbBlockContinueTail;

    int64_t MAX_CHANNEL_SIZE = 1024;

    float dyReduceValue;
    float dyMulInputReduceValue;

protected:
    __aicore__ inline void initSplitLoadTiling(const BatchNormGradV3SplitLoadTilingData* tilingData_)
    {
        this->b1Dim = tilingData_->b1Dim;
        this->aDim = tilingData_->aDim;
        this->b0Dim = tilingData_->b0Dim;
        this->bAlign = tilingData_->bAlign;
        this->coreChannelNum = tilingData_->coreChannelNum;
        this->coreChannelNumTail = tilingData_->coreChannelNumTail;
        this->cUbBlock = tilingData_->bUbBlock;
        this->needCoreNum = tilingData_->needCoreNum;
        this->epsilon = tilingData_->epsilon;
        this->b1DimAlign = GetAlignValue<T1>(this->b1Dim);
        this->bAlign = this->b1DimAlign * this->b0Dim;

        this->blockIdx = GetBlockIdx();
        if (this->blockIdx > this->needCoreNum) {
            return;
        }

        this->totalChannel = this->coreChannelNum;
        if (this->blockIdx == this->needCoreNum - 1) {
            this->totalChannel = this->coreChannelNumTail;
        }
    }

    __aicore__ inline void InitSplitLoad(GMStruct gmStruct, TPipe* pipe_)
    {
        this->dyGm_.SetGlobalBuffer((__gm__ T1*)gmStruct.dy);
        this->xGm_.SetGlobalBuffer((__gm__ T1*)gmStruct.x);
        this->weightGm_.SetGlobalBuffer((__gm__ T2*)gmStruct.weight);
        this->varGm_.SetGlobalBuffer((__gm__ T3*)gmStruct.var);
        this->meanGm_.SetGlobalBuffer((__gm__ T3*)gmStruct.mean);
        this->dxGm_.SetGlobalBuffer((__gm__ T1*)gmStruct.dx);
        this->dWeightGm_.SetGlobalBuffer((__gm__ T2*)gmStruct.dWeight);
        this->dBiasGm_.SetGlobalBuffer((__gm__ T2*)gmStruct.dBias);
        this->workSpaceGm_.SetGlobalBuffer((__gm__ float*)gmStruct.usrWorkspace);
    }

    __aicore__ inline void InitSplitLoadMode(GMStruct gmStruct, TPipe* pipe_)
    {
        int64_t xShapeLen = this->cUbBlock;
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, xShapeLen * sizeof(float));
        pipe_->InitBuffer(reduceQueue, BUFFER_NUM, xShapeLen * sizeof(float));
        pipe_->InitBuffer(reduceInputQueue, BUFFER_NUM, xShapeLen * sizeof(float));
        pipe_->InitBuffer(inputQueue, BUFFER_NUM, xShapeLen * sizeof(float));

        int64_t tmpCLength = GetAlignValue<T2>(this->totalChannel);
        tmpCLength = tmpCLength > MAX_CHANNEL_SIZE ? MAX_CHANNEL_SIZE : tmpCLength;
        if (tmpCLength > 0) {
            pipe_->InitBuffer(meanQueue, BUFFER_NUM, tmpCLength * sizeof(float));
            pipe_->InitBuffer(varQueue, BUFFER_NUM, tmpCLength * sizeof(float));
            pipe_->InitBuffer(weightQueue, BUFFER_NUM, tmpCLength * sizeof(float));
        }

        int64_t totalSize = xShapeLen * 4 * sizeof(float) + tmpCLength * 3 * sizeof(float);
    }

    __aicore__ inline void CopyInXDyR0SplitMode(int64_t dyGmOffset)
    {
        LocalTensor<float> dyLocal = dyQueue_.AllocTensor<float>();
        LocalTensor<float> xLocal = inputQueue.AllocTensor<float>();

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = this->cBlockLength * sizeof(T1);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = true;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = this->cBlockLengthT1Align - this->cBlockLength;
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            dataCopyPadExtParams.paddingValue = ToBfloat16(0);
        } else {
            dataCopyPadExtParams.paddingValue = (T1)0;
        }
        LoadOneTensor(this->dyGm_[dyGmOffset], dyLocal, copyInParams, dataCopyPadExtParams, this->cBlockLength);
        LoadOneTensor(this->xGm_[dyGmOffset], xLocal, copyInParams, dataCopyPadExtParams, this->cBlockLength);
        inputQueue.EnQue(xLocal);
        dyQueue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInXDyR1SplitMode(int64_t dyGmOffset)
    {
        LocalTensor<float> dyLocal = dyQueue_.AllocTensor<float>();
        LocalTensor<float> xLocal = inputQueue.AllocTensor<float>();

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = bUbR1Block;
        copyInParams.blockLen = this->b1Dim * sizeof(T1);
        copyInParams.srcStride = this->b1Dim * (this->aDim - 1) * sizeof(T1);
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = this->b1DimAlign - this->b1Dim;
        dataCopyPadExtParams.isPad = true;

        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            dataCopyPadExtParams.paddingValue = ToBfloat16(0);
        } else {
            dataCopyPadExtParams.paddingValue = (T1)0;
        }
        LoadOneTensor(this->dyGm_[dyGmOffset], dyLocal, copyInParams, dataCopyPadExtParams, this->cBlockLengthT1Align);
        LoadOneTensor(this->xGm_[dyGmOffset], xLocal, copyInParams, dataCopyPadExtParams, this->cBlockLengthT1Align);

        inputQueue.EnQue(xLocal);
        dyQueue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInWeightMeanVar(int64_t offset, int64_t index)
    {
        int64_t mod = index % MAX_CHANNEL_SIZE;
        if (mod == 0) {
            weightTensor = weightQueue.AllocTensor<float>();
            meanTensor = meanQueue.AllocTensor<float>();
            varTensor = varQueue.AllocTensor<float>();

            int64_t loopId = index / MAX_CHANNEL_SIZE;
            offset = offset + loopId * MAX_CHANNEL_SIZE;
            int64_t copyNum = MAX_CHANNEL_SIZE;
            if (loopId == CeilDiv(this->totalChannel, MAX_CHANNEL_SIZE) - 1) {
                copyNum = this->totalChannel % MAX_CHANNEL_SIZE;
            }

            DataCopyExtParams extParam;
            extParam.blockCount = 1;
            extParam.blockLen = copyNum * sizeof(T2);
            DataCopyPadExtParams<T2> padExtParam;
            padExtParam.isPad = false;
            LoadOneTensor(this->weightGm_[offset], weightTensor, extParam, padExtParam, copyNum);

            // var
            extParam.blockLen = copyNum * sizeof(T3);
            DataCopyPadExtParams<T3> padExtParam2;
            padExtParam2.isPad = false;
            LoadOneTensor(this->meanGm_[offset], meanTensor, extParam, padExtParam2, copyNum);
            LoadOneTensor(this->varGm_[offset], varTensor, extParam, padExtParam2, copyNum);

            weightQueue.EnQue(weightTensor);
            meanQueue.EnQue(meanTensor);
            varQueue.EnQue(varTensor);

            weightTensor = weightQueue.DeQue<float>();
            meanTensor = meanQueue.DeQue<float>();
            varTensor = varQueue.DeQue<float>();
            TPipeSetWaitFlag<HardEvent::MTE2_V>();
            Adds(varTensor, varTensor, this->epsilon, copyNum);
            PipeBarrier<PIPE_V>();
            Sqrt(varTensor, varTensor, copyNum);
            PipeBarrier<PIPE_V>();
        }
        TPipeSetWaitFlag<HardEvent::V_S>();
        weight = weightTensor.GetValue(mod);
        mean = meanTensor.GetValue(mod);
        var = varTensor.GetValue(mod);
        rVar = var == 0 ? 0 : 1 / var;
    }

    __aicore__ inline void computeDBias()
    {
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceUbTensor, reduceUbTensor, reduceUbTensor, this->cUbBlock);
        PipeBarrier<PIPE_V>();
        reduceQueue.EnQue(reduceUbTensor);
    }

    __aicore__ inline void computeDWeight()
    {
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceInputUbTensor, reduceInputUbTensor, reduceInputUbTensor, this->cUbBlock);
        PipeBarrier<PIPE_V>();
        reduceInputQueue.EnQue(reduceInputUbTensor);
    }

    __aicore__ inline void CopyOutDBias(int64_t offset, int64_t index)
    {
        reduceUbTensor = reduceQueue.DeQue<float>();

        int64_t loopId = index / MAX_CHANNEL_SIZE;
        int64_t mod = index % MAX_CHANNEL_SIZE;
        int64_t copyNum = MAX_CHANNEL_SIZE;
        if (loopId == CeilDiv(this->totalChannel, MAX_CHANNEL_SIZE) - 1) {
            copyNum = this->totalChannel % MAX_CHANNEL_SIZE;
        }
        meanTensor.SetValue(mod, reduceUbTensor.GetValue(0));
        if (mod == copyNum - 1) {
            offset = offset + loopId * MAX_CHANNEL_SIZE;
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = copyNum * sizeof(T2);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            TPipeSetWaitFlag<HardEvent::S_MTE3>();
            StoreOneTensor(this->dBiasGm_[offset], meanTensor, copyInParams, copyNum);
            TPipeSetWaitFlag<HardEvent::MTE3_V>();

            meanQueue.FreeTensor(meanTensor);
            weightQueue.FreeTensor(weightTensor);
        }
    }

    __aicore__ inline void CopyOutDWeight(int64_t offset, int64_t index)
    {
        reduceInputUbTensor = reduceInputQueue.DeQue<float>();

        int64_t loopId = index / MAX_CHANNEL_SIZE;
        int64_t mod = index % MAX_CHANNEL_SIZE;
        int64_t copyNum = MAX_CHANNEL_SIZE;
        if (loopId == CeilDiv(this->totalChannel, MAX_CHANNEL_SIZE) - 1) {
            copyNum = this->totalChannel % MAX_CHANNEL_SIZE;
        }
        varTensor.SetValue(mod, reduceInputUbTensor.GetValue(0) * rVar);

        if (mod == copyNum - 1) {
            offset = offset + loopId * MAX_CHANNEL_SIZE;
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = copyNum * sizeof(T2);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            TPipeSetWaitFlag<HardEvent::S_MTE3>();
            StoreOneTensor(this->dWeightGm_[offset], varTensor, copyInParams, copyNum);
            TPipeSetWaitFlag<HardEvent::MTE3_V>();
            varQueue.FreeTensor(varTensor);
        }
    }

    __aicore__ inline void CopyOutDxR0SplitMode(int64_t dxGmOffset)
    {
        LocalTensor<float> dx = dyQueue_.DeQue<float>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = this->cBlockLength * sizeof(T1);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        StoreOneTensor(this->dxGm_[dxGmOffset], dx, copyInParams, this->cBlockLengthT1Align);

        dyQueue_.FreeTensor(dx);
    }

    __aicore__ inline void CopyOutDxR1SplitMode(int64_t dxGmOffset)
    {
        LocalTensor<float> dx = dyQueue_.DeQue<float>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = bUbR1Block;
        copyInParams.blockLen = this->b1Dim * sizeof(T1);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = this->b1Dim * (this->aDim - 1) * sizeof(T1);

        StoreOneTensor(this->dxGm_[dxGmOffset], dx, copyInParams, this->cBlockLengthT1Align);

        dyQueue_.FreeTensor(dx);
    }

    __aicore__ inline void ProcessLoopCommon(int64_t offset, int64_t channelIndex)
    {
        this->CopyInWeightMeanVar(offset, channelIndex);

        this->reduceUbTensor = this->reduceQueue.template AllocTensor<float>();
        this->reduceInputUbTensor = this->reduceInputQueue.template AllocTensor<float>();
        Duplicate(this->reduceUbTensor, static_cast<float>(0), this->cUbBlock);
        Duplicate(this->reduceInputUbTensor, static_cast<float>(0), this->cUbBlock);
    }

    __aicore__ inline void ComputeAndCopyOutBiasAndWeight(int64_t offset, int64_t channelIndex)
    {
        this->computeDBias();
        this->computeDWeight();

        this->CopyOutDBias(offset, channelIndex);
        this->CopyOutDWeight(offset, channelIndex);
    }

    /**
     * 由[c]广播为[c, hw]。
     */
    __aicore__ inline void BroadCastWeightMeanVar(WeightMeanVarStruct& weightMeanVarStruct)
    {
        Adds(weightMeanVarStruct.varTensor, weightMeanVarStruct.varTensor, this->epsilon, this->aDim);
        PipeBarrier<PIPE_V>();
        Sqrt(weightMeanVarStruct.varTensor, weightMeanVarStruct.varTensor, this->aDim);
        PipeBarrier<PIPE_V>();
        uint32_t srcShape[2] = {static_cast<uint32_t>(this->aDim), 1};
        uint32_t dstShape[2] = {static_cast<uint32_t>(this->aDim), static_cast<uint32_t>(this->b1DimAlign)};
        BroadCast<float, TWO, 1, false>(
            weightMeanVarStruct.weightBrcbTensor, weightMeanVarStruct.weightTensor, dstShape, srcShape);
        BroadCast<float, TWO, 1, false>(
            weightMeanVarStruct.varBrcbTensor, weightMeanVarStruct.varTensor, dstShape, srcShape);
        BroadCast<float, TWO, 1, false>(
            weightMeanVarStruct.meanBrcbTensor, weightMeanVarStruct.meanTensor, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void calcTwoTensorRepeatOperate(
        void (*func)(
            const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, const uint8_t,
            const BinaryRepeatParams&),
        void (*func2)(const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, const int32_t&),
        LocalTensor<float> dstTensor, LocalTensor<float> src1Tensor, LocalTensor<float> src2Tensor, int64_t nLength,
        int64_t lLength, int64_t totalLength)
    {
        if (lLength < ELEM_PER_REP_FP32 * UINT8_MAX_NUM) {
            int64_t repeatTime = lLength / ELEM_PER_REP_FP32;
            int64_t lLengthTail = lLength % ELEM_PER_REP_FP32;

            int64_t nLoopTime = nLength / UINT8_MAX_NUM;
            uint8_t repStride = lLength / B32_BLOCK_ALIGN_NUM;
            for (int64_t j = 0; j < repeatTime; j++) {
                for (int64_t i = 0; i < nLoopTime; i++) {
                    func(
                        dstTensor[i * UINT8_MAX_NUM * lLength + j * ELEM_PER_REP_FP32],
                        src1Tensor[i * UINT8_MAX_NUM * lLength + j * ELEM_PER_REP_FP32],
                        src2Tensor[j * ELEM_PER_REP_FP32], ELEM_PER_REP_FP32, UINT8_MAX_NUM,
                        {1, 1, 1, repStride, repStride, 0});
                }
                if (nLength % UINT8_MAX_NUM > 0) {
                    func(
                        dstTensor[nLoopTime * UINT8_MAX_NUM * lLength + j * ELEM_PER_REP_FP32],
                        src1Tensor[nLoopTime * UINT8_MAX_NUM * lLength + j * ELEM_PER_REP_FP32],
                        src2Tensor[j * ELEM_PER_REP_FP32], ELEM_PER_REP_FP32, nLength % UINT8_MAX_NUM,
                        {1, 1, 1, repStride, repStride, 0});
                }
            }
            if (lLengthTail != 0) {
                for (int64_t i = 0; i < nLoopTime; i++) {
                    func(
                        dstTensor[i * UINT8_MAX_NUM * lLength + repeatTime * ELEM_PER_REP_FP32],
                        src1Tensor[i * UINT8_MAX_NUM * lLength + repeatTime * ELEM_PER_REP_FP32],
                        src2Tensor[repeatTime * ELEM_PER_REP_FP32], lLengthTail, UINT8_MAX_NUM,
                        {1, 1, 1, repStride, repStride, 0});
                }
                if (nLength % UINT8_MAX_NUM > 0) {
                    func(
                        dstTensor[nLoopTime * UINT8_MAX_NUM * lLength + repeatTime * ELEM_PER_REP_FP32],
                        src1Tensor[nLoopTime * UINT8_MAX_NUM * lLength + repeatTime * ELEM_PER_REP_FP32],
                        src2Tensor[repeatTime * ELEM_PER_REP_FP32], lLengthTail, nLength % UINT8_MAX_NUM,
                        {1, 1, 1, repStride, repStride, 0});
                }
            }
        } else {
            for (int64_t i = 0; i < nLength; i++) {
                func2(dstTensor[i * lLength], src1Tensor[i * lLength], src2Tensor, totalLength);
            }
        }
    }

    __aicore__ inline void CopyInWeightMeanVarContinue()
    {
        LocalTensor<float> weightLocal = this->weightQueue.template AllocTensor<float>();
        LocalTensor<float> meanLocal = this->meanQueue.template AllocTensor<float>();
        LocalTensor<float> varLocal = this->varQueue.template AllocTensor<float>();
        this->varBrcbTensor = this->varBrcbQueue.template Get<float>();
        this->meanBrcbTensor = this->meanBrcbQueue.template Get<float>();
        this->weightBrcbTensor = this->weightBrcbQueue.template Get<float>();

        DataCopyExtParams extParam;
        extParam.blockCount = 1;
        extParam.blockLen = this->aDim * sizeof(T2);
        DataCopyPadExtParams<T2> padExtParam;
        padExtParam.isPad = false;
        LoadOneTensor(this->weightGm_, weightLocal, extParam, padExtParam, this->aDim);

        // var
        extParam.blockLen = this->aDim * sizeof(T3);
        DataCopyPadExtParams<T3> padExtParam2;
        padExtParam2.isPad = false;
        LoadOneTensor(this->meanGm_, meanLocal, extParam, padExtParam2, this->aDim);
        LoadOneTensor(this->varGm_, varLocal, extParam, padExtParam2, this->aDim);

        this->weightQueue.template EnQue(weightLocal);
        this->meanQueue.template EnQue(meanLocal);
        this->varQueue.template EnQue(varLocal);

        this->weightTensor = this->weightQueue.template DeQue<float>();
        this->meanTensor = this->meanQueue.template DeQue<float>();
        this->varTensor = this->varQueue.template DeQue<float>();

        TPipeSetWaitFlag<HardEvent::MTE2_V>();
        WeightMeanVarStruct weightMeanVarStruct{this->weightTensor,     this->meanTensor,     this->varTensor,
                                                this->weightBrcbTensor, this->meanBrcbTensor, this->varBrcbTensor};
        this->BroadCastWeightMeanVar(weightMeanVarStruct);
    }

    __aicore__ inline void ComputeInLineInfer()
    {
        this->xTensor = this->inputQueue.template DeQue<float>();
        this->dyTensor = this->dyQueue_.template DeQue<float>();
        Adds(this->xTensor, this->xTensor, -this->mean, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        Mul(this->xTensor, this->xTensor, this->dyTensor, this->cBlockLength);
        Add(this->reduceUbTensor, this->dyTensor, this->reduceUbTensor, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        Add(this->reduceInputUbTensor, this->xTensor, this->reduceInputUbTensor, this->cBlockLength);
        Muls(this->dyTensor, this->dyTensor, this->rVar, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        Muls(this->dyTensor, this->dyTensor, this->weight, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        this->inputQueue.template FreeTensor(this->xTensor);
        this->dyQueue_.template EnQue(this->dyTensor);
    }

    __aicore__ inline void ComputeInLineTrain()
    {
        this->xTensor = this->inputQueue.template DeQue<float>();
        this->dyTensor = this->dyQueue_.template DeQue<float>();
        Adds(this->xTensor, this->xTensor, -this->mean, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        Add(this->reduceUbTensor, this->dyTensor, this->reduceUbTensor, this->cBlockLength);
        Mul(this->xTensor, this->xTensor, this->dyTensor, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        Add(this->reduceInputUbTensor, this->xTensor, this->reduceInputUbTensor, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        this->inputQueue.template FreeTensor(this->xTensor);
        this->dyQueue_.template FreeTensor(this->dyTensor);
    }

    __aicore__ inline void ComputeDxTrain()
    {
        this->dyTensor = this->dyQueue_.template DeQue<float>();
        this->xTensor = this->inputQueue.template DeQue<float>();
        Adds(this->dyTensor, this->dyTensor, -dyReduceValue, this->cBlockLength);
        Adds(this->xTensor, this->xTensor, -this->mean, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        Muls(this->xTensor, this->xTensor, dyMulInputReduceValue, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        Sub(this->dyTensor, this->dyTensor, this->xTensor, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        Muls(this->dyTensor, this->dyTensor, this->rVar, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        Muls(this->dyTensor, this->dyTensor, this->weight, this->cBlockLength);
        PipeBarrier<PIPE_V>();
        this->dyQueue_.template EnQue(this->dyTensor);
        this->inputQueue.template FreeTensor(this->xTensor);
    }
};
} // namespace BatchNormGradV3
#endif
