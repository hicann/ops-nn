/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ada_layer_norm_grad_bs_workspace.h
 * \brief
 */


#ifndef ADA_LAYER_NORM_GRAD_MERGE_BS_WORKSPACE
#define ADA_LAYER_NORM_GRAD_MERGE_BS_WORKSPACE

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "ada_layer_norm_grad_determinstic_compute.h"

namespace AdaLayerNormGrad {
using namespace AscendC;
template <typename T, typename U, bool isDeterministic>
class AdaLayerNormGradMergeBSWorkspace {
public:
    __aicore__ inline AdaLayerNormGradMergeBSWorkspace(){};
__aicore__ inline void CalculateBlockLengths(
    const AdaLayerNormGradTilingDataMergeBSWorkspace* tilingData, int64_t& formerBlockLength, int64_t& curRowsNum, int64_t& curBlockLength) {
    formerBlockLength = tilingData->blockFormer * tilingData->col;
    curRowsNum = (GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail;
    curBlockLength = curRowsNum * tilingData->col;
}

__aicore__ inline void InitGmTensors(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR scale, GM_ADDR gamma, GM_ADDR beta, GM_ADDR pdX, GM_ADDR pdScale, GM_ADDR pdShift,
    const AdaLayerNormGradTilingDataMergeBSWorkspace* tilingData, int64_t formerBlockLength, int64_t curRowsNum, int64_t curBlockLength) {
    // 初始化GM输入
    sBatchIdx = (GetBlockIdx() * tilingData->blockFormer) / tilingData->seq;
    int64_t eBatchIdx = (GetBlockIdx() * tilingData->blockFormer + curRowsNum) / tilingData->seq;
    int64_t curBlockScaleLength = (eBatchIdx - sBatchIdx + 1) * tilingData->col;
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + formerBlockLength * GetBlockIdx(), curBlockLength);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + formerBlockLength * GetBlockIdx(), curBlockLength);
    rstdInTensorGM.SetGlobalBuffer((__gm__ float*)rstd + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
    scaleInTensorGM.SetGlobalBuffer((__gm__ T*)scale + sBatchIdx * colLen, curBlockScaleLength);
    gammaInTensorGM.SetGlobalBuffer((__gm__ U*)gamma, tilingData->col);
    betaInTensorGM.SetGlobalBuffer((__gm__ U*)beta, tilingData->col);
    
    // 初始化GM输出
    pdXOutTensorGM.SetGlobalBuffer((__gm__ T*)pdX + formerBlockLength * GetBlockIdx(), curBlockLength);
}

__aicore__ inline void InitWorkspace(const AdaLayerNormGradTilingDataMergeBSWorkspace* tilingData, 
                                     GM_ADDR workspace, int64_t wsLenPerBlock) {
    int64_t wsBase = wsLenPerBlock * GetBlockIdx();
    
    dGammaWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase, wsLenPerBlock);
    dBetaWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + colAlign, colAlign);
    dScaleWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + 2 * colAlign, tilingData->batch * colAlign);
    dShiftWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + (tilingData->batch + 2) * colAlign, tilingData->batch * colAlign);
    mul1WorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + (2 * tilingData->batch + 2) * colAlign, colAlign);
    mul3WorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + wsBase + (2 * tilingData->batch + 3) * colAlign, colAlign);
    
    // 清空工作空间
    InitOutput<float>(dGammaWorkspaceGM, (2 * tilingData->batch + 2) * colAlign, 0.0);
}

__aicore__ inline void InitQueues(TPipe& pipe, const AdaLayerNormGradTilingDataMergeBSWorkspace* tilingData) {
    int64_t sizeOfBuffer = tilingData->ubFormer * sizeof(float);
    pipe.InitBuffer(queIn0_, 1, sizeOfBuffer);
    pipe.InitBuffer(queIn1_, 1, sizeOfBuffer);
    pipe.InitBuffer(queIn2_, 1, sizeOfBuffer);
    pipe.InitBuffer(queIn3_, 1, sizeOfBuffer);
    pipe.InitBuffer(queOut4_, 1, sizeOfBuffer);
    pipe.InitBuffer(queOut5_, 1, sizeOfBuffer);
    coff = static_cast<float>(1.0) / static_cast<float>(tilingData->col);
}

__aicore__ inline void Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR scale, GM_ADDR gamma, GM_ADDR beta, GM_ADDR pdX, GM_ADDR pdScale, GM_ADDR pdShift, 
    GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace, const AdaLayerNormGradTilingDataMergeBSWorkspace* tilingData, TPipe& pipeIn) {
    // 初始化基础变量
    pipe = pipeIn;
    seq = tilingData->seq;
    colAlign = tilingData->colAlignV;
    colLen = tilingData->col;
    
    // 计算块长度相关参数
    int64_t formerBlockLength, curRowsNum, curBlockLength;
    CalculateBlockLengths(tilingData, formerBlockLength, curRowsNum, curBlockLength);
    
    // 计算工作空间长度并初始化基础GM
    int64_t wsLenPerBlock = colAlign * (WORKSPACE_NUM + 2 * tilingData->batch);
    pdGammaOutTensorGM.SetGlobalBuffer((__gm__ U*)pdGamma, colLen);
    pdBetaOutTensorGM.SetGlobalBuffer((__gm__ U*)pdBeta, colLen);
    pdScaleOutTensorGM.SetGlobalBuffer((__gm__ T*)pdScale, tilingData->batch * colLen);
    pdShiftOutTensorGM.SetGlobalBuffer((__gm__ T*)pdShift, tilingData->batch * colLen);
    workspaceGMOri.SetGlobalBuffer((__gm__ float*)workspace, wsLenPerBlock * tilingData->blockNum);

    if (GetBlockIdx() < tilingData->blockNum) {
        // 初始化GM输入输出
        InitGmTensors(dy, x, rstd, mean, scale, gamma, beta, pdX, pdScale, pdShift, tilingData,
                     formerBlockLength, curRowsNum, curBlockLength);
        
        // 初始化工作空间
        InitWorkspace(tilingData, workspace, wsLenPerBlock);

        // 非确定性模式下的初始化
        if constexpr (!isDeterministic) {
            if (GetBlockIdx() == 0) {
                InitOutput<U>(pdGammaOutTensorGM, colLen, 0.0);
                InitOutput<U>(pdBetaOutTensorGM, colLen, 0.0);
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
        }
        
        PipeBarrier<PIPE_ALL>();
        // 初始化buffer
        InitQueues(pipe, tilingData);
    } else if (!isDeterministic) {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
    }
}

    __aicore__ inline void Process(const AdaLayerNormGradTilingDataMergeBSWorkspace* tilingData)
    {
        if (GetBlockIdx() < tilingData->blockNum) {
            int64_t rowCount =
                (GetBlockIdx() == tilingData->blockNum - 1) ? tilingData->blockTail : tilingData->blockFormer;
            int64_t blockSrow = GetBlockIdx() * tilingData->blockFormer;

            for (int64_t rowIndex = 0; rowIndex < rowCount; rowIndex++) {
                rowOfBatch = (rowIndex + blockSrow) / seq;
                float reduce2 = 0.0f;
                float reduce3 = 0.0f;
                float rstdIn = rstdInTensorGM.GetValue(rowIndex);
                float meanIn = meanInTensorGM.GetValue(rowIndex);

                // step 1. calc reduce2 reduce3 mul1 mul3
                for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
                    CopyInPhase0(rowIndex, ubIndex, tilingData->ubFormer, tilingData->ubFormer);
                    ComputePhase0(ubIndex, tilingData->ubFormer, tilingData->ubFormer, meanIn, rstdIn);
                    LocalTensor<float> temp0 = queIn0_.AllocTensor<float>();
                    LocalTensor<float> temp1 = queIn1_.AllocTensor<float>();
                    reduce2 += ReducePhase0(buffer4_, temp0, tilingData->ubFormer);
                    reduce3 += ReducePhase0(buffer5_, temp1, tilingData->ubFormer);
                    FreeTensor();

                    queIn0_.FreeTensor(temp0);
                    queIn1_.FreeTensor(temp1);
                }
                CopyInPhase0(rowIndex, tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail);
                ComputePhase0(tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail, meanIn, rstdIn);
                LocalTensor<float> temp0 = queIn0_.AllocTensor<float>();
                LocalTensor<float> temp1 = queIn1_.AllocTensor<float>();
                reduce2 += ReducePhase0(buffer4_, temp0, tilingData->ubTail);
                reduce3 += ReducePhase0(buffer5_, temp1, tilingData->ubTail);
                FreeTensor();
                queIn0_.FreeTensor(temp0);
                queIn1_.FreeTensor(temp1);
                reduce2 = reduce2 * coff;
                reduce3 = reduce3 * coff;

                // step 2. calc dx
                PipeBarrier<PIPE_ALL>();
                for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
                    CopyInPhase1(ubIndex, tilingData->ubFormer, tilingData->ubFormer);
                    ComputePhase1(
                        rowIndex, ubIndex, tilingData->ubFormer, tilingData->ubFormer, reduce2, reduce3, rstdIn);
                }
                CopyInPhase1(tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail);
                ComputePhase1(
                    rowIndex, tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail, reduce2, reduce3,
                    rstdIn);
            }
        }
        // step3. calc pgamma and pbeta form workspace
        doLastStep(tilingData);
    }

    __aicore__ inline void CopyInPhase0(int64_t rowIndex, int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        // copy in dy with x
        buffer0_ = queIn0_.AllocTensor<float>();
        buffer1_ = queIn1_.AllocTensor<float>();
        buffer2_ = queIn2_.AllocTensor<float>();
        buffer3_ = queIn3_.AllocTensor<float>();

        DataCopyParams intriParamsT = {1, static_cast<uint16_t>(calcNum * sizeof(T)), 0, 0};
        DataCopyParams intriParamsU = {1, static_cast<uint16_t>(calcNum * sizeof(U)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};
        int64_t colOffset = ubIndex * ubFormer;
        int64_t rowOffset = rowIndex * colLen + colOffset;
        int64_t batchOffset = (rowOfBatch - sBatchIdx) * colLen + colOffset;

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer0_, dyInTensorGM[rowOffset], intriParamsT, padParams);
            DataCopyPad(buffer1_, xInTensorGM[rowOffset], intriParamsT, padParams);
            DataCopyPad(buffer2_, scaleInTensorGM[batchOffset], intriParamsT, padParams);
        } else {
            DataCopyPad(
                buffer0_.ReinterpretCast<T>()[ubFormer], dyInTensorGM[rowOffset], intriParamsT, padParams); // buffer0_ dy
            DataCopyPad(
                buffer1_.ReinterpretCast<T>()[ubFormer], xInTensorGM[rowOffset], intriParamsT, padParams); // buffer1_ x
            DataCopyPad(buffer2_.ReinterpretCast<T>()[ubFormer], scaleInTensorGM[batchOffset], intriParamsT, padParams);
        }
        queIn0_.EnQue(buffer0_);
        queIn1_.EnQue(buffer1_);
        queIn2_.EnQue(buffer2_);

        if constexpr (IsSameType<U, float>::value) {
            DataCopyPad(buffer3_, gammaInTensorGM[colOffset], intriParamsU, padParams);
        } else {
            DataCopyPad(buffer3_.ReinterpretCast<U>()[ubFormer], gammaInTensorGM[colOffset], intriParamsU, padParams);
        }
        queIn3_.EnQue(buffer3_);
    }

    __aicore__ inline void ComputePhase0Cast(int64_t ubFormer, int64_t calcNum)
    {
        buffer0_ = queIn0_.DeQue<float>();
        buffer1_ = queIn1_.DeQue<float>();
        buffer2_ = queIn2_.DeQue<float>();
        buffer3_ = queIn3_.DeQue<float>();

        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer0_, buffer0_.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
            Cast(buffer1_, buffer1_.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
            Cast(buffer2_, buffer2_.ReinterpretCast<T>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
        }

        if constexpr (!IsSameType<U, float>::value) {
            Cast(buffer3_, buffer3_.ReinterpretCast<U>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputePhase0Part1(
        int64_t ubIndex, int64_t ubFormer, int64_t calcNum, float meanIn, float rstdIn)
    {
        Adds(buffer4_, buffer0_, 0.0f, calcNum); // dy+0
        PipeBarrier<PIPE_V>();

        Adds(buffer2_, buffer2_, 1.0f, calcNum); // 1+scale
        PipeBarrier<PIPE_V>();

        Adds(buffer1_, buffer1_, -meanIn, calcNum); // x-mean
        PipeBarrier<PIPE_V>();

        queOut4_.EnQue(buffer4_);
        buffer4_ = queOut4_.DeQue<float>();

        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};

        SetAtomicAdd<float>();
        DataCopyPad(dShiftWorkspaceGM[rowOfBatch * colAlign + ubIndex * ubFormer], buffer4_, intriParams);
        SetAtomicNone();

        PipeBarrier<PIPE_V>();
        Muls(buffer5_, buffer1_, rstdIn, calcNum); //(x-mean)*rstd

        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event0);
        WaitFlag<HardEvent::MTE3_V>(event0);

        PipeBarrier<PIPE_V>();
        Mul(buffer4_, buffer0_, buffer2_, calcNum); // dy*(1+scale)
        PipeBarrier<PIPE_V>();

        queOut5_.EnQue(buffer5_);
        buffer5_ = queOut5_.DeQue<float>();
        DataCopyPad(mul1WorkspaceGM[ubIndex * ubFormer], buffer5_, intriParams); // workspace mul_1=(x-mean)*rstd

        queOut4_.EnQue(buffer4_);
        buffer4_ = queOut4_.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopyPad(dBetaWorkspaceGM[ubIndex * ubFormer], buffer4_, intriParams); // workspace dbeta=(1+scale)*dy
        SetAtomicNone();

        PipeBarrier<PIPE_V>();
        Mul(buffer2_, buffer4_, buffer3_, calcNum); //(1+scale)*dy*gamma
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputePhase0Part2(int64_t ubIndex, int64_t calcNum)
    {
        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event1);
        WaitFlag<HardEvent::MTE3_V>(event1);

        Mul(buffer4_, buffer5_, buffer4_, calcNum); //(x-mean)*rstd * (1+scale)*dy
        PipeBarrier<PIPE_V>();

        Mul(buffer1_, buffer3_, buffer5_, calcNum); // gamma*(x-mean)*rstd
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputePhase0Part3(int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};

        event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event2);
        WaitFlag<HardEvent::V_MTE3>(event2);
        queOut4_.EnQue(buffer4_);
        buffer4_ = queOut4_.DeQue<float>();

        SetAtomicAdd<float>();
        DataCopyPad(dGammaWorkspaceGM[ubIndex * ubFormer], buffer4_, intriParams); // workspace dgamma
        SetAtomicNone();
        queIn3_.FreeTensor(buffer3_);

        CopyInBeta(ubIndex, ubFormer, calcNum);

        PipeBarrier<PIPE_V>();
        Add(buffer1_, buffer3_, buffer1_, calcNum); // gamma*(x-mean)*rstd+beta
        PipeBarrier<PIPE_V>();

        event_t event3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event3);
        WaitFlag<HardEvent::MTE3_V>(event3);
        Mul(buffer4_, buffer0_, buffer1_, calcNum); // dy*(gamma*(x-mean)*rstd+beta)
        PipeBarrier<PIPE_V>();

        event_t event4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event4);
        WaitFlag<HardEvent::V_MTE3>(event4);
        queOut4_.EnQue(buffer4_);
        buffer4_ = queOut4_.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopyPad(
            dScaleWorkspaceGM[rowOfBatch * colAlign + ubIndex * ubFormer], buffer4_, intriParams); // workspace dscale
        SetAtomicNone();

        PipeBarrier<PIPE_V>();
        Mul(buffer5_, buffer2_, buffer5_, calcNum); //(1+scale)*dy*gamma*(x-mean)*rstd
        PipeBarrier<PIPE_V>();

        event_t event5 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event5);
        WaitFlag<HardEvent::MTE3_V>(event5);
        Adds(buffer4_, buffer2_, 0.0f, calcNum); //(1+scale)*dy*gamma
        PipeBarrier<PIPE_ALL>();

        DataCopyPad(mul3WorkspaceGM[ubIndex * ubFormer], buffer4_, intriParams);

        queIn0_.FreeTensor(buffer0_);
        queIn1_.FreeTensor(buffer1_);
        queIn2_.FreeTensor(buffer2_);
        queIn3_.FreeTensor(buffer3_);
    }

    __aicore__ inline void ComputePhase0(int64_t ubIndex, int64_t ubFormer, int64_t calcNum, float meanIn, float rstdIn)
    {
        buffer4_ = queOut4_.AllocTensor<float>();
        buffer5_ = queOut5_.AllocTensor<float>();

        ComputePhase0Cast(ubFormer, calcNum);

        ComputePhase0Part1(ubIndex, ubFormer, calcNum, meanIn, rstdIn);

        ComputePhase0Part2(ubIndex, calcNum);

        ComputePhase0Part3(ubIndex, ubFormer, calcNum);
    }

    __aicore__ inline void FreeTensor()
    {
        queOut4_.FreeTensor(buffer4_);
        queOut5_.FreeTensor(buffer5_);
    }

    __aicore__ inline void CopyInBeta(int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        // copy in dy with x
        buffer3_ = queIn3_.AllocTensor<float>();

        DataCopyParams intriParamsU = {1, static_cast<uint16_t>(calcNum * sizeof(U)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};
        int64_t colOffset = ubIndex * ubFormer;

        if constexpr (IsSameType<U, float>::value) {
            DataCopyPad(buffer3_, betaInTensorGM[colOffset], intriParamsU, padParams);
        } else {
            DataCopyPad(buffer3_.ReinterpretCast<U>()[ubFormer], betaInTensorGM[colOffset], intriParamsU, padParams);
        }

        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event1);
        WaitFlag<HardEvent::MTE2_V>(event1);

        if constexpr (!IsSameType<U, float>::value) {
            Cast(buffer3_, buffer3_.ReinterpretCast<U>()[ubFormer], RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline float ReducePhase0(
        const LocalTensor<float>& src, const LocalTensor<float>& tmp, int64_t reduceNum)
    {
        ReduceSum(tmp, src, tmp, reduceNum);
        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event0);
        WaitFlag<HardEvent::V_S>(event0);
        float value = tmp.GetValue(0);
        return value;
    }

    __aicore__ inline void CopyInPhase1(int64_t ubIndex, int64_t ubFormer, int64_t calcNum)
    {
        buffer0_ = queIn0_.AllocTensor<float>();
        buffer1_ = queIn1_.AllocTensor<float>();
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams padParams = {false, 0, 0, 0.0f};
        DataCopyPad(buffer0_, mul1WorkspaceGM[ubIndex * ubFormer], intriParams, padParams); // 取出(x-mean)*rstd
        queIn0_.EnQue(buffer0_);
        DataCopyPad(buffer1_, mul3WorkspaceGM[ubIndex * ubFormer], intriParams, padParams); // 取出dy*gamma*(1+scale)
        queIn1_.EnQue(buffer1_);
    }

    __aicore__ inline void ComputePhase1(
        int64_t rowIndex, int64_t ubIndex, int64_t ubFormer, int64_t calcNum, float reduce2, float reduce3,
        float rstdIn)
    {
        buffer4_ = queOut4_.AllocTensor<float>();
        buffer0_ = queIn0_.DeQue<float>();
        buffer1_ = queIn1_.DeQue<float>();
        Muls(buffer0_, buffer0_, reduce3, calcNum); //(x-mean)*rstd*(1/N)*∑(dy*gamma*(1+scale)*(x-mean)*rstd)
        PipeBarrier<PIPE_V>();
        Adds(buffer1_, buffer1_, -reduce2, calcNum);// dy*gamma*(1+scale)-(1/N)∑dy*gamma*(1+scale)
        PipeBarrier<PIPE_V>();
        // dy*gamma-(1/N)∑dy*gamma*(1+scale)-(x-mean)*rstd*(1/N)*∑(dy*gamma*(1+scale)*(x-mean)*rstd))
        Sub(buffer1_, buffer1_, buffer0_, calcNum);
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, float>::value) {
            Muls(
                buffer4_, buffer1_, rstdIn,
                calcNum); // rstd*(dy*gamma-(1/N)∑dy*gamma*(1+scale)-(x-mean)*rstd*(1/N)*∑(dy*gamma*(1+scale)*(x-mean)*rstd)))
                          // type=float32
        } else {
            Muls(buffer0_, buffer1_, rstdIn, calcNum); //  type=half
            PipeBarrier<PIPE_V>();
            Cast(buffer4_.ReinterpretCast<T>(), buffer0_, RoundMode::CAST_RINT, calcNum);
        }
        queOut4_.EnQue(buffer4_);
        buffer4_ = queOut4_.DeQue<float>();
        DataCopyExtParams intriParams = {1, static_cast<uint32_t>(calcNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(pdXOutTensorGM[rowIndex * colLen + ubIndex * ubFormer], buffer4_.ReinterpretCast<T>(), intriParams);
        queIn0_.FreeTensor(buffer0_);
        queIn1_.FreeTensor(buffer1_);
        queOut4_.FreeTensor(buffer4_);
    }

    __aicore__ inline void doLastStep(const AdaLayerNormGradTilingDataMergeBSWorkspace* tilingData)
    {
        if constexpr (isDeterministic) {
            pipe.Reset();
            SyncAll();
            FinalProcessDeterministicNew(tilingData);
            pipe.Reset();
            SyncAll();
            SWTFinalProcessDeterministic(tilingData);
        } else if (GetBlockIdx() < tilingData->blockNum) {
            PipeBarrier<PIPE_ALL>();
            CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
            for (int64_t ubIndex = 0; ubIndex < tilingData->ubLoop - 1; ubIndex++) {
                FinalProcessOfGB(ubIndex, tilingData->ubFormer, tilingData->ubFormer, tilingData->blockNum);
            }
            FinalProcessOfGB(tilingData->ubLoop - 1, tilingData->ubFormer, tilingData->ubTail, tilingData->blockNum);
        } else {
            CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
        }
    }

    __aicore__ inline void FinalProcessOfGB(int64_t ubIndex, int64_t ubFormer, int64_t calcNum, int64_t blockNum)
    {
        buffer0_ = queIn0_.AllocTensor<float>();
        buffer1_ = queIn1_.AllocTensor<float>();
        buffer4_ = queOut4_.AllocTensor<float>();
        buffer5_ = queOut5_.AllocTensor<float>();

        queIn0_.FreeTensor(buffer0_);
        queIn1_.FreeTensor(buffer1_);
        queOut4_.FreeTensor(buffer4_);
        queOut5_.FreeTensor(buffer5_);
    }

    __aicore__ inline void FinalProcessDeterministicNew(const AdaLayerNormGradTilingDataMergeBSWorkspace* tilingData)
    {
        pipe.Reset();
        SyncAll();
        AdaLayerNormGradDeterminsticCompute<U> op;
        int64_t curWorkspaceRowsNum = 2 * tilingData->batch + WORKSPACE_NUM;
        op.initBuffer(pipe, pdGammaOutTensorGM, pdBetaOutTensorGM, workspaceGMOri, curWorkspaceRowsNum);
        op.FinalProcessDeterministic(tilingData->colAlignV, tilingData->blockNum, tilingData->col);
    }

    __aicore__ inline void SWTFinalProcessDeterministic(const AdaLayerNormGradTilingDataMergeBSWorkspace* tilingData)
    {
        int64_t curWorkspaceRowsNum = 2 * tilingData->batch + WORKSPACE_NUM;

        AdaLayerNormGradDeterminsticCompute<T> op;
        // initBuffer 依然按照原来的方式调用，传入初始的基地址
        op.initBuffer(pipe, pdScaleOutTensorGM, pdShiftOutTensorGM, workspaceGMOri, curWorkspaceRowsNum);

        // 调用新写的 Batch 处理函数
        op.BatchFinalProcessDeterministic(
            tilingData->batch, tilingData->colAlignV, tilingData->blockNum, tilingData->col, pdScaleOutTensorGM,
            pdShiftOutTensorGM);
    }

private:
    constexpr static int64_t WORKSPACE_NUM = 4;
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;

    TPipe pipe;

    TQue<QuePosition::VECIN, 1> queIn0_;
    TQue<QuePosition::VECIN, 1> queIn1_;
    TQue<QuePosition::VECIN, 1> queIn2_;
    TQue<QuePosition::VECIN, 1> queIn3_;
    TQue<QuePosition::VECOUT, 1> queOut4_;
    TQue<QuePosition::VECOUT, 1> queOut5_;

    LocalTensor<float> buffer0_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;
    LocalTensor<float> buffer4_;
    LocalTensor<float> buffer5_;

    GlobalTensor<float> dScaleWorkspaceGM;
    GlobalTensor<float> dShiftWorkspaceGM;
    GlobalTensor<float> dBetaWorkspaceGM;
    GlobalTensor<float> dGammaWorkspaceGM;
    GlobalTensor<float> mul1WorkspaceGM;
    GlobalTensor<float> mul3WorkspaceGM;
    GlobalTensor<float> workspaceGMOri;
    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<T> scaleInTensorGM;
    GlobalTensor<U> gammaInTensorGM;
    GlobalTensor<U> betaInTensorGM;
    GlobalTensor<T> pdXOutTensorGM;
    GlobalTensor<T> pdScaleOutTensorGM;
    GlobalTensor<T> pdShiftOutTensorGM;
    GlobalTensor<U> pdGammaOutTensorGM;
    GlobalTensor<U> pdBetaOutTensorGM;

    int64_t colLen;
    int64_t colAlign;
    int64_t seq;
    int64_t rowOfBatch;
    int64_t sBatchIdx;
    float coff;
};
} // namespace AdaLayerNormGrad
#endif