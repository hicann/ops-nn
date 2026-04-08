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
 * \file ada_layer_norm_grad_workspace.h
 * \brief
 */

#ifndef ADA_LAYER_NORM_GRAD_WORKSPACE
#define ADA_LAYER_NORM_GRAD_WORKSPACE

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "ada_layer_norm_grad_determinstic_compute.h"

namespace AdaLayerNormGrad {
using namespace AscendC;
template <typename T, typename U, bool isDeterministic>
class AdaLayerNormGradWorkspace {
public:
__aicore__ inline AdaLayerNormGradWorkspace(){};

__aicore__ inline void CalculateBlockLengths(
    const AdaLayerNormGradTilingDataWorkspace* tilingData, int64_t& formerBlockLength, int64_t& curRowsNum, int64_t& curBlockBatchRowNum, int64_t& blockBatchOffset,
    int64_t& blockStartBatch, int64_t& curBlockLength, int64_t& curBlockBatchLength) {
    formerBlockLength = tilingData->blockFormer * tilingData->col;
    curRowsNum = (GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail;
    curBlockBatchRowNum = curRowsNum / seq;
    blockBatchOffset = tilingData->blockFormer / seq;
    blockStartBatch = blockBatchOffset * tilingData->col;
    curBlockLength = curRowsNum * tilingData->col;
    curBlockBatchLength = curBlockBatchRowNum * tilingData->col;
}

__aicore__ inline void InitGmTensors(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR scale, GM_ADDR gamma, GM_ADDR beta, GM_ADDR pdX, GM_ADDR pdScale, GM_ADDR pdShift,
    const AdaLayerNormGradTilingDataWorkspace* tilingData, int64_t formerBlockLength, int64_t curRowsNum, int64_t curBlockLength,
    int64_t blockStartBatch, int64_t curBlockBatchLength) {
    // 初始化GM输入
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + formerBlockLength * GetBlockIdx(), curBlockLength);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + formerBlockLength * GetBlockIdx(), curBlockLength);
    rstdInTensorGM.SetGlobalBuffer((__gm__ float*)rstd + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
    scaleInTensorGM.SetGlobalBuffer((__gm__ T*)scale + blockStartBatch * GetBlockIdx(), curBlockBatchLength);
    gammaInTensorGM.SetGlobalBuffer((__gm__ U*)gamma, tilingData->col);
    betaInTensorGM.SetGlobalBuffer((__gm__ U*)beta, tilingData->col);
    
    // 初始化GM输出
    pdXOutTensorGM.SetGlobalBuffer((__gm__ T*)pdX + formerBlockLength * GetBlockIdx(), curBlockLength);
    pdScaleOutTensorGM.SetGlobalBuffer((__gm__ T*)pdScale + blockStartBatch * GetBlockIdx(), curBlockBatchLength);
    pdShiftOutTensorGM.SetGlobalBuffer((__gm__ T*)pdShift + blockStartBatch * GetBlockIdx(), curBlockBatchLength);
}

__aicore__ inline void InitWorkspace(const AdaLayerNormGradTilingDataWorkspace* tilingData, 
                                     GM_ADDR workspace, int64_t wsLenPerBlock, int64_t curBlockBatchRowNum) {
    colLen = tilingData->col;
    int64_t wsBase = wsLenPerBlock * GetBlockIdx();
    
    dGammaWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase, (2 * curBlockBatchRowNum + 2) * colAlign);
    dBetaWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + colAlign, colAlign);
    dScaleWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + 2 * colAlign, curBlockBatchRowNum * colAlign);
    dShiftWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + (curBlockBatchRowNum + 2) * colAlign, curBlockBatchRowNum * colAlign);
    mul1WorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + (2 * curBlockBatchRowNum + 2) * colAlign, colAlign);
    mul3WorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + (2 * curBlockBatchRowNum + 3) * colAlign, colAlign);
    
    // 清空工作空间
    InitOutput<float>(dGammaWorkspaceGM, (2 * curBlockBatchRowNum + 2) * colAlign, 0.0);
}

__aicore__ inline void InitQueues(TPipe& pipe, const AdaLayerNormGradTilingDataWorkspace* tilingData) {
    int64_t bufferSize = tilingData->ubFormer * sizeof(float);
    pipe.InitBuffer(queIn0, 1, bufferSize);
    pipe.InitBuffer(queIn1, 1, bufferSize);
    pipe.InitBuffer(queIn2, 1, bufferSize);
    pipe.InitBuffer(queIn3, 1, bufferSize);
    pipe.InitBuffer(queOut4, 1, bufferSize);
    pipe.InitBuffer(queOut5, 1, bufferSize);
}

__aicore__ inline void Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR scale, GM_ADDR gamma, GM_ADDR beta, GM_ADDR pdX, GM_ADDR pdScale, GM_ADDR pdShift, 
    GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace, const AdaLayerNormGradTilingDataWorkspace* tilingData, TPipe& pipeIn) {
    // 初始化基础变量
    pipe = pipeIn;
    seq = tilingData->seq;
    colAlign = tilingData->colAlignV;
    
    // 计算块长度相关参数
    int64_t formerBlockLength, curRowsNum, curBlockBatchRowNum, blockBatchOffset, blockStartBatch, curBlockLength, curBlockBatchLength;
    CalculateBlockLengths(tilingData, formerBlockLength, curRowsNum, curBlockBatchRowNum, blockBatchOffset, blockStartBatch, curBlockLength, curBlockBatchLength);
    
    // 计算工作空间长度并初始化基础GM
    int64_t wsLenPerBlock = colAlign * (WORKSPACE_NUM + 2 * blockBatchOffset);
    pdGammaOutTensorGM.SetGlobalBuffer((__gm__ U*)pdGamma, tilingData->col);
    pdBetaOutTensorGM.SetGlobalBuffer((__gm__ U*)pdBeta, tilingData->col);
    workspaceGMOri.SetGlobalBuffer((__gm__ float*)workspace, wsLenPerBlock * (tilingData->blockNum - 1) + (tilingData->blockTail / seq) * colAlign);

    if (GetBlockIdx() < tilingData->blockNum) {
        // 初始化GM输入输出
        InitGmTensors(dy, x, rstd, mean, scale, gamma, beta, pdX, pdScale, pdShift, tilingData,
                     formerBlockLength, curRowsNum, curBlockLength, blockStartBatch, curBlockBatchLength);
        
        // 初始化工作空间
        InitWorkspace(tilingData, workspace, wsLenPerBlock, curBlockBatchRowNum);

        // 非确定性模式下的初始化
        if constexpr (!isDeterministic) {
            if (GetBlockIdx() == 0) {
                InitOutput<T>(pdScaleOutTensorGM, curBlockBatchLength, 0.0);
                InitOutput<T>(pdShiftOutTensorGM, curBlockBatchLength, 0.0);
                InitOutput<U>(pdGammaOutTensorGM, tilingData->col, 0.0);
                InitOutput<U>(pdBetaOutTensorGM, tilingData->col, 0.0);
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
        }
        
        PipeBarrier<PIPE_ALL>();
        // 初始化队列
        InitQueues(pipe, tilingData);
        coff_ = static_cast<float>(1.0) / static_cast<float>(tilingData->col);
    } else if (!isDeterministic) {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
    }
}

    __aicore__ inline void Process(const AdaLayerNormGradTilingDataWorkspace* tilingData)
    {
        if (GetBlockIdx() < tilingData->blockNum) {
            int64_t rowCount = (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->blockTail : tilingData->blockFormer;
            for (int64_t rowIndex = 0; rowIndex < rowCount; rowIndex++) {
                rowOfBatch = rowIndex / seq;

                float reduce2 = 0.0f;
                float reduce3 = 0.0f;
                float meanIn = meanInTensorGM.GetValue(rowIndex);
                float rstdIn = rstdInTensorGM.GetValue(rowIndex);

                // step 1. calc reduce2 reduce3 mul_1 mul3
                for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
                    CopyInPhase0(rowIndex, ubIndex, tilingData->ubFormer, tilingData->ubFormer);
                    ComputePhase0(ubIndex, tilingData->ubFormer, tilingData->ubFormer, meanIn, rstdIn);
                    LocalTensor<float> tmp0 = queIn0.AllocTensor<float>();
                    LocalTensor<float> tmp1 = queIn1.AllocTensor<float>();
                    reduce2 += ReducePhase0(buffer4, tmp0, tilingData->ubFormer);
                    reduce3 += ReducePhase0(buffer5, tmp1, tilingData->ubFormer);
                    FreeTensor();

                    queIn0.FreeTensor(tmp0);
                    queIn1.FreeTensor(tmp1);
                }
                CopyInPhase0(rowIndex, tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail);
                ComputePhase0(tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail, meanIn, rstdIn);
                LocalTensor<float> tmp0 = queIn0.AllocTensor<float>();
                LocalTensor<float> tmp1 = queIn1.AllocTensor<float>();
                reduce2 += ReducePhase0(buffer4, tmp0, tilingData->ubTail);
                reduce3 += ReducePhase0(buffer5, tmp1, tilingData->ubTail);
                FreeTensor();
                queIn0.FreeTensor(tmp0);
                queIn1.FreeTensor(tmp1);
                reduce2 = reduce2 * coff_;
                reduce3 = reduce3 * coff_;

                // step 2. calc dx
                PipeBarrier<PIPE_ALL>();
                for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
                    CopyInPhase1(ubIndex, tilingData->ubFormer, tilingData->ubFormer);
                    ComputePhase1(rowIndex, ubIndex, tilingData->ubFormer, tilingData->ubFormer, reduce2, reduce3, rstdIn);
                }
                CopyInPhase1(tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail);
                ComputePhase1(rowIndex, tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail, reduce2, reduce3, rstdIn);
            }
            CopyOutdScaleWithdShift(tilingData);
        }

        // step3. calc pgamma and pbeta form workspace
        doLastStep(tilingData);
    }


    __aicore__ inline void CopyInPhase0(int64_t rowIndex, int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        //copy in dy with x
        buffer0 = queIn0.AllocTensor<float>();
        buffer1 = queIn1.AllocTensor<float>();
        buffer2 = queIn2.AllocTensor<float>();
        buffer3 = queIn3.AllocTensor<float>();

        DataCopyParams intriParamsT = {1, static_cast<uint16_t>(calcNum * sizeof(T)), 0, 0};
        DataCopyParams intriParamsU = {1, static_cast<uint16_t>(calcNum * sizeof(U)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};
        int64_t colOffset = ubIndex * ubFormer;
        int64_t rowOffset = rowIndex * colLen + colOffset;
        int64_t batchOffset = rowOfBatch * colLen + colOffset;

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer0, dyInTensorGM[rowOffset], intriParamsT, padParams);
            DataCopyPad(buffer1, xInTensorGM[rowOffset], intriParamsT, padParams);
            DataCopyPad(buffer2, scaleInTensorGM[batchOffset], intriParamsT, padParams);
        } else {
            DataCopyPad(buffer0.ReinterpretCast<T>()[ubFormer], dyInTensorGM[rowOffset], intriParamsT, padParams);//buffer0 dy
            DataCopyPad(buffer1.ReinterpretCast<T>()[ubFormer], xInTensorGM[rowOffset], intriParamsT, padParams);//buffer1 x
            DataCopyPad(buffer2.ReinterpretCast<T>()[ubFormer], scaleInTensorGM[batchOffset], intriParamsT, padParams);
        }
        queIn0.EnQue(buffer0);
        queIn1.EnQue(buffer1);
        queIn2.EnQue(buffer2);

        if constexpr (IsSameType<U, float>::value) {
            DataCopyPad(buffer3, gammaInTensorGM[colOffset], intriParamsU, padParams);
        } else {
            DataCopyPad(buffer3.ReinterpretCast<U>()[ubFormer], gammaInTensorGM[colOffset], intriParamsU, padParams);
        }
        queIn3.EnQue(buffer3);
    }


    __aicore__ inline void ComputePhase0Cast(int64_t ubFormer, int64_t calcNum)
    {
        buffer0 = queIn0.DeQue<float>();
        buffer1 = queIn1.DeQue<float>();
        buffer2 = queIn2.DeQue<float>();
        buffer3 = queIn3.DeQue<float>();

        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer0, buffer0.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
            Cast(buffer1, buffer1.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
            Cast(buffer2, buffer2.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
        }
        
        if constexpr (!IsSameType<U, float>::value) {
            Cast(buffer3, buffer3.ReinterpretCast<U>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
        }
    }


    __aicore__ inline void ComputePhase0Part1(int64_t ubIndex, int64_t ubFormer, int64_t calcNum, float meanIn, float rstdIn)
    {
        Adds(buffer4, buffer0, 0.0f, calcNum);//dy+0
        PipeBarrier<PIPE_V>();

        Adds(buffer2, buffer2, 1.0f, calcNum);//1+scale
        PipeBarrier<PIPE_V>();
        
        Adds(buffer1, buffer1, -meanIn, calcNum);//x-mean
        PipeBarrier<PIPE_V>();

        queOut4.EnQue(buffer4);
        buffer4 = queOut4.DeQue<float>();
        
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};
        
        SetAtomicAdd<float>();
        DataCopyPad(dShiftWorkspaceGM[rowOfBatch * colAlign + ubIndex * ubFormer], buffer4, intriParams); 
        SetAtomicNone();

        PipeBarrier<PIPE_V>();
        Muls(buffer5, buffer1, rstdIn, calcNum);//(x-mean)*rstd

        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event0);
        WaitFlag<HardEvent::MTE3_V>(event0);

        PipeBarrier<PIPE_V>();
        Mul(buffer4, buffer0, buffer2, calcNum);// dy*(1+scale)
        PipeBarrier<PIPE_V>();

        queOut5.EnQue(buffer5);
        buffer5 = queOut5.DeQue<float>();
        DataCopyPad(mul1WorkspaceGM[ubIndex * ubFormer], buffer5, intriParams); // workspace mul_1=(x-mean)*rstd
        
        queOut4.EnQue(buffer4);
        buffer4 = queOut4.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopyPad(dBetaWorkspaceGM[ubIndex * ubFormer], buffer4, intriParams); // workspace dbeta=(1+scale)*dy
        SetAtomicNone();
        
        PipeBarrier<PIPE_V>();
        Mul(buffer2, buffer4, buffer3, calcNum);//(1+scale)*dy*gamma
        PipeBarrier<PIPE_V>();
    }


    __aicore__ inline void ComputePhase0Part2(int64_t ubIndex, int64_t calcNum)
    {
        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event1);
        WaitFlag<HardEvent::MTE3_V>(event1);

        Mul(buffer4, buffer5, buffer4, calcNum);//(x-mean)*rstd * (1+scale)*dy
        PipeBarrier<PIPE_V>();

        Mul(buffer1, buffer3, buffer5, calcNum);//gamma*(x-mean)*rstd
        PipeBarrier<PIPE_V>();
    }


    __aicore__ inline void ComputePhase0Part3(int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};
        
        event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event2);
        WaitFlag<HardEvent::V_MTE3>(event2);
        queOut4.EnQue(buffer4);
        buffer4 = queOut4.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyPad(dGammaWorkspaceGM[ubIndex * ubFormer], buffer4, intriParams); // workspace dgamma
        SetAtomicNone();
        queIn3.FreeTensor(buffer3);

        CopyInBeta(ubIndex, ubFormer, calcNum);

        PipeBarrier<PIPE_V>();
        Add(buffer1, buffer3, buffer1, calcNum);//gamma*(x-mean)*rstd+beta
        PipeBarrier<PIPE_V>();

        event_t event3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event3);
        WaitFlag<HardEvent::MTE3_V>(event3);
        Mul(buffer4, buffer0, buffer1, calcNum);//dy*(gamma*(x-mean)*rstd+beta)
        PipeBarrier<PIPE_V>();

        event_t event4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event4);
        WaitFlag<HardEvent::V_MTE3>(event4);
        queOut4.EnQue(buffer4);
        buffer4 = queOut4.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopyPad(dScaleWorkspaceGM[rowOfBatch * colAlign + ubIndex * ubFormer], buffer4, intriParams); // workspace dscale
        SetAtomicNone();

        PipeBarrier<PIPE_V>();
        Mul(buffer5, buffer2, buffer5, calcNum);//(1+scale)*dy*gamma*(x-mean)*rstd
        PipeBarrier<PIPE_V>();

        event_t event5 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event5);
        WaitFlag<HardEvent::MTE3_V>(event5);
        Adds(buffer4, buffer2, 0.0f, calcNum);//(1+scale)*dy*gamma
        PipeBarrier<PIPE_ALL>();

        DataCopyPad(mul3WorkspaceGM[ubIndex * ubFormer], buffer4, intriParams); 

        queIn0.FreeTensor(buffer0);
        queIn1.FreeTensor(buffer1);
        queIn2.FreeTensor(buffer2);
        queIn3.FreeTensor(buffer3);
    }

    __aicore__ inline void ComputePhase0(int64_t ubIndex, int64_t ubFormer, int64_t calcNum, float meanIn, float rstdIn)
    {
        buffer4 = queOut4.AllocTensor<float>();
        buffer5 = queOut5.AllocTensor<float>();

        ComputePhase0Cast(ubFormer, calcNum);
        
        ComputePhase0Part1(ubIndex, ubFormer, calcNum, meanIn, rstdIn);
        
        ComputePhase0Part2(ubIndex, calcNum);
        
        ComputePhase0Part3(ubIndex, ubFormer, calcNum);
    }

    __aicore__ inline void FreeTensor()
    {
        queOut4.FreeTensor(buffer4);
        queOut5.FreeTensor(buffer5);
    }

        __aicore__ inline void CopyInBeta(int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        //copy in dy with x
        buffer3 = queIn3.AllocTensor<float>();

        DataCopyParams intriParamsU = {1, static_cast<uint16_t>(calcNum * sizeof(U)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};
        int64_t colOffset = ubIndex * ubFormer;

        if constexpr (IsSameType<U, float>::value) {
            DataCopyPad(buffer3, betaInTensorGM[colOffset], intriParamsU, padParams);
        } else {
            DataCopyPad(buffer3.ReinterpretCast<U>()[ubFormer], betaInTensorGM[colOffset], intriParamsU, padParams);
        }

        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event1);
        WaitFlag<HardEvent::MTE2_V>(event1);
        
        if constexpr (!IsSameType<U, float>::value) {
            Cast(buffer3, buffer3.ReinterpretCast<U>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline float ReducePhase0(const LocalTensor<float>& src, const LocalTensor<float>& tmp, int64_t reduceNum)
    {
        ReduceSum(tmp, src, tmp, reduceNum);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float value = tmp.GetValue(0);
        return value;
    }

    __aicore__ inline void CopyInPhase1(int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        buffer0 = queIn0.AllocTensor<float>();
        buffer1 = queIn1.AllocTensor<float>();
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams padParams = {false, 0, 0, 0.0f};
        DataCopyPad(buffer0, mul1WorkspaceGM[ubIndex * ubFormer], intriParams, padParams);//取出(x-mean)*rstd
        queIn0.EnQue(buffer0);
        DataCopyPad(buffer1, mul3WorkspaceGM[ubIndex * ubFormer], intriParams, padParams);//取出dy*gamma*(1+scale)
        queIn1.EnQue(buffer1);
    }

    __aicore__ inline void ComputePhase1(
        int64_t rowIndex, int64_t ubIndex, int64_t ubFormer, int64_t calcNum, float reduce2, float reduce3,
        float rstdIn)
    {
        buffer4 = queOut4.AllocTensor<float>();
        buffer0 = queIn0.DeQue<float>();
        buffer1 = queIn1.DeQue<float>();
        Muls(buffer0, buffer0, reduce3, calcNum);//(x-mean)*rstd*(1/N)*∑(dy*gamma*(1+scale)*(x-mean)*rstd)
        PipeBarrier<PIPE_V>();
        Adds(buffer1, buffer1, -reduce2, calcNum);//dy*gamma*(1+scale)-(x-mean)*rstd*(1/N)*∑(dy*gamma*(1+scale)*(x-mean)*rstd)
        PipeBarrier<PIPE_V>();
        Sub(buffer1, buffer1, buffer0, calcNum);//dy*gamma-(1/N)∑dy*gamma*(1+scale)-(x-mean)*rstd*(1/N)*∑(dy*gamma*(1+scale)*(x-mean)*rstd))
        PipeBarrier<PIPE_V>();


        if constexpr (IsSameType<T, float>::value) {
            Muls(buffer4, buffer1, rstdIn, calcNum);//rstd*(dy*gamma-(1/N)∑dy*gamma*(1+scale)-(x-mean)*rstd*(1/N)*∑(dy*gamma*(1+scale)*(x-mean)*rstd)))  type=float32
        } else {
            Muls(buffer0, buffer1, rstdIn, calcNum);//  type=half
            PipeBarrier<PIPE_V>();
            Cast(buffer4.ReinterpretCast<T>(), buffer0,  RoundMode::CAST_RINT, calcNum);
            
        }
        queOut4.EnQue(buffer4);
        buffer4 = queOut4.DeQue<float>();
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(pdXOutTensorGM[rowIndex * colLen + ubIndex * ubFormer], buffer4.ReinterpretCast<T>(), intriParams);
        queIn0.FreeTensor(buffer0);
        queIn1.FreeTensor(buffer1);
        queOut4.FreeTensor(buffer4);
    }

        __aicore__ inline void CopyOutdScaleWithdShift(const AdaLayerNormGradTilingDataWorkspace* tilingData)
    {
        int64_t BatchRowsNum =
            ((GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail) / seq;
        for (int64_t BatchIndex = 0; BatchIndex < BatchRowsNum; BatchIndex++){

        for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
                CopyOutSWT(ubIndex, BatchIndex, tilingData->ubFormer, tilingData->ubFormer);
            }
            CopyOutSWT(tilingData->ubLoop - 1, BatchIndex, tilingData->ubFormer, tilingData->ubTail);
        }
    }


        __aicore__ inline void CopyOutSWT(int64_t ubIndex, int64_t BatchIndex, int64_t ubFormer, int64_t calcNum)
    {
        buffer0 = queIn0.AllocTensor<float>();
        buffer1 = queIn1.AllocTensor<float>();
        buffer4 = queOut4.AllocTensor<float>();
        buffer5 = queOut5.AllocTensor<float>();

        DataCopyExtParams intriParamsIn = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams padParams = {false, 0, 0, 0.0f};
        int64_t offset = ubIndex * ubFormer;

        DataCopyPad(buffer0, dScaleWorkspaceGM[BatchIndex * colAlign + offset], intriParamsIn, padParams);
        queIn0.EnQue(buffer0);
        DataCopyPad(buffer1, dShiftWorkspaceGM[BatchIndex * colAlign + offset], intriParamsIn, padParams);
        queIn1.EnQue(buffer1);

        
        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event1);
        WaitFlag<HardEvent::MTE2_V>(event1);
        buffer0 = queIn0.DeQue<float>();
        buffer1 = queIn1.DeQue<float>();

        Adds(buffer4, buffer0, 0.0f, calcNum);
        PipeBarrier<PIPE_V>();
        Adds(buffer5, buffer1, 0.0f, calcNum);
        PipeBarrier<PIPE_V>();
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer4.ReinterpretCast<T>(), buffer4,  RoundMode::CAST_RINT, calcNum);
            PipeBarrier<PIPE_V>();
            Cast(buffer5.ReinterpretCast<T>(), buffer5,  RoundMode::CAST_RINT, calcNum);
            PipeBarrier<PIPE_V>();
        }

        event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event2);
        WaitFlag<HardEvent::V_MTE3>(event2);
        DataCopyExtParams intriParamsOut = {1, static_cast<uint32_t>(calcNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(pdScaleOutTensorGM[BatchIndex * colLen + offset], buffer4.ReinterpretCast<T>(), intriParamsOut);
        DataCopyPad(pdShiftOutTensorGM[BatchIndex * colLen + offset], buffer5.ReinterpretCast<T>(), intriParamsOut);

        queIn0.FreeTensor(buffer0);
        queIn1.FreeTensor(buffer1);
        queOut4.FreeTensor(buffer4);
        queOut5.FreeTensor(buffer5);

    }

    __aicore__ inline void doLastStep(const AdaLayerNormGradTilingDataWorkspace* tilingData)
    {
        if constexpr (isDeterministic) {
            pipe.Reset();
            SyncAll();
            FinalProcessDeterministicNew(tilingData);
        } else if (GetBlockIdx() < tilingData->blockNum) {
            PipeBarrier<PIPE_ALL>();
            CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
            for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
                FinalProcess(ubIndex, tilingData->ubFormer, tilingData->ubFormer, tilingData->blockNum);
            }
            FinalProcess(tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail, tilingData->blockNum);
        } else {
            CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
        }
    }

    __aicore__ inline void FinalProcess(int64_t ubIndex, int64_t ubFormer, int64_t calcNum, int64_t blockNum)
    {
            buffer0 = queIn0.AllocTensor<float>();
            buffer1 = queIn1.AllocTensor<float>();
            buffer4 = queOut4.AllocTensor<float>();
            buffer5 = queOut5.AllocTensor<float>();

            queIn0.FreeTensor(buffer0);
            queIn1.FreeTensor(buffer1);
            queOut4.FreeTensor(buffer4);
            queOut5.FreeTensor(buffer5);
    }

    __aicore__ inline void FinalProcessDeterministicNew(const AdaLayerNormGradTilingDataWorkspace* tilingData)
    {
        pipe.Reset();
        SyncAll();
        AdaLayerNormGradDeterminsticCompute<U> op;
        // 这里beta gamma参数倒着传是因为workspace和singleread gamma和beta在workspace里的存储顺序是反的
        int64_t curWorkspaceRowsNum = (2 * tilingData->blockFormer) / seq + WORKSPACE_NUM;
        op.initBuffer(pipe, pdGammaOutTensorGM, pdBetaOutTensorGM, workspaceGMOri, curWorkspaceRowsNum);
        op.FinalProcessDeterministic(tilingData->colAlignV, tilingData->blockNum, tilingData->col);
    }

private:
    constexpr static int64_t WORKSPACE_NUM = 4;
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;

    TPipe pipe;

    TQue<QuePosition::VECIN, 1> queIn0;
    TQue<QuePosition::VECIN, 1> queIn1;
    TQue<QuePosition::VECIN, 1> queIn2;
    TQue<QuePosition::VECIN, 1> queIn3;
    TQue<QuePosition::VECOUT, 1> queOut4;
    TQue<QuePosition::VECOUT, 1> queOut5;

    LocalTensor<float> buffer0;
    LocalTensor<float> buffer1;
    LocalTensor<float> buffer2;
    LocalTensor<float> buffer3;
    LocalTensor<float> buffer4;
    LocalTensor<float> buffer5;


    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<T> scaleInTensorGM;
    GlobalTensor<U> gammaInTensorGM;
    GlobalTensor<U> betaInTensorGM;
    GlobalTensor<T> pdXOutTensorGM;
    GlobalTensor<T> pdScaleOutTensorGM;
    GlobalTensor<T> pdShiftOutTensorGM;
    GlobalTensor<U> pdGammaOutTensorGM;
    GlobalTensor<U> pdBetaOutTensorGM;
    GlobalTensor<float> dScaleWorkspaceGM;
    GlobalTensor<float> dShiftWorkspaceGM;
    GlobalTensor<float> dBetaWorkspaceGM;
    GlobalTensor<float> dGammaWorkspaceGM;
    GlobalTensor<float> mul1WorkspaceGM;
    GlobalTensor<float> mul3WorkspaceGM;
    GlobalTensor<float> workspaceGMOri;

    int64_t colLen;
    int64_t colAlign;
    int64_t seq;
    int64_t rowOfBatch;
    float coff_;
};
} // namespace AdaLayerNormGrad
#endif