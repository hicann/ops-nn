/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_v3_transpose_gamma_beta_impl.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_TRANSPOSE_GAMMA_BETA_IMPL_
#define LAYER_NORM_GRAD_V3_TRANSPOSE_GAMMA_BETA_IMPL_
#include "layer_norm_grad_v3_api.h"
#include "layer_norm_grad_v3_base.h"

namespace LayerNormGradV3 {
using namespace AscendC;

template <typename T, typename PD_GAMMA_TYPE>
class LayerNormGradV3TransposeGammaBeta : public LayerNormGradV3Base {
public:
    __aicore__ inline LayerNormGradV3TransposeGammaBeta() : LayerNormGradV3Base(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR pdGamma, GM_ADDR pdBeta,
                                GM_ADDR workspace, const LayerNormGradV3TilingDataTransposeRegBase* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInData(const int64_t mi, const int64_t mfactor, LocalTensor<float>& dyOut,
                                      LocalTensor<float>& xOut, LocalTensor<float>& meanOut,
                                      LocalTensor<float>& rstdOut);
    __aicore__ inline void CopyOutData();
    __aicore__ inline void Stage1();
    __aicore__ inline void Stage2();
    __aicore__ inline void ComputeDyMulXNormMain(const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
                                                 const LocalTensor<float>& rstdTensor,
                                                 const LocalTensor<float>& meanTensor, const int64_t rowSize,
                                                 const int64_t colSize, const int64_t stride);
    __aicore__ inline void ComputeDyMulXNormFold(const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
                                                 const LocalTensor<float>& rstdTensor,
                                                 const LocalTensor<float>& meanTensor, LocalTensor<float>& xMainTensor,
                                                 const int64_t rowSize, const int64_t colSize, const int64_t stride);

private:
    const LayerNormGradV3TilingDataTransposeRegBase* __restrict td_;
    TPipe* pipe_;

    constexpr static int64_t ONE_BUFFER = 1;
    constexpr static int64_t DOUBLE_BUFFER = 2;
    constexpr static int64_t TRIPLE_BUFFER = 3;

    int64_t N = 0;
    int64_t NAlign = 0;
    int64_t MAlign = 0;
    int64_t currentLoop = 0;
    int64_t currentMTail = 0;
    int64_t basicBlockLoop = 0;
    int64_t mainFoldCount = 0;
    int64_t resultCacheID = 0;

    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<float> dbetaWorkspaceGM;
    GlobalTensor<float> dgammaWorkspaceGM;
    GlobalTensor<PD_GAMMA_TYPE> dgammaOutTensorGM;
    GlobalTensor<PD_GAMMA_TYPE> dbetaOutTensorGM;
    GM_ADDR savedWorkspaceAddr = nullptr;
    GM_ADDR savedPdGammaAddr = nullptr;
    GM_ADDR savedPdBetaAddr = nullptr;

    TQue<QuePosition::VECIN, ONE_BUFFER> inQueueDy;
    TQue<QuePosition::VECIN, ONE_BUFFER> inQueueX;
    TQue<QuePosition::VECIN, ONE_BUFFER> inQueueMean;
    TQue<QuePosition::VECIN, ONE_BUFFER> inQueueRstd;
    TQue<QuePosition::VECOUT, ONE_BUFFER> outQueueDgamma;
    TQue<QuePosition::VECOUT, ONE_BUFFER> outQueueDbeta;

    TBuf<> dbetaCacheBuffer;
    TBuf<> dgammaCacheBuffer;
    TBuf<> reduceSumBuffer;
    TBuf<> tmpDataBuffer;

    LocalTensor<float> dyMain_;
    LocalTensor<float> xMain_;
    LocalTensor<float> dbetaCache;
    LocalTensor<float> dgammaCache;
};

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3TransposeGammaBeta<T, PD_GAMMA_TYPE>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace,
    const LayerNormGradV3TilingDataTransposeRegBase* tilingData, TPipe* pipeIn)
{
    td_ = tilingData;
    N = td_->col;
    NAlign = td_->gammaBetaNAlign;
    MAlign = td_->gammaBetaMAlign;

    int64_t usedCoreNum = td_->gammaBetaUsedCoreNum;
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx >= usedCoreNum) {
        return;
    }

    // 主核每核mPerCore行，尾核mTailCore行
    int64_t mPerCore = td_->gammaBetaMPerCore;
    int64_t mTailCore = td_->gammaBetaMTailCore;

    int64_t isTail = (blockIdx == usedCoreNum - 1) ? 1 : 0;
    int64_t oneCoreM = isTail ? mTailCore : mPerCore;
    // basicBlock: 二分累加的主块轮数; resultCacheID: 二分树最终结果所在cache层
    basicBlockLoop = isTail ? td_->gammaBetaTailCoreBasicBlock : td_->gammaBetaMainCoreBasicBlock;
    resultCacheID = isTail ? td_->gammaBetaTailResultCacheID : td_->gammaBetaMainResultCacheID;

    // 总数据块数量
    currentLoop = oneCoreM / MAlign;              // 整块的块数
    currentMTail = oneCoreM % MAlign;             // 尾块的大小
    mainFoldCount = currentLoop - basicBlockLoop; // 二分折叠块数

    int64_t dyShape = oneCoreM * N;
    int64_t gmOffsetM = blockIdx * mPerCore;
    int64_t gmOffsetMN = gmOffsetM * N;
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + gmOffsetMN, dyShape);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + gmOffsetMN, dyShape);
    rstdInTensorGM.SetGlobalBuffer((__gm__ float*)rstd + gmOffsetM, oneCoreM);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean + gmOffsetM, oneCoreM);

    // workspace分两段: 前半段dbeta, 后半段dgamma, 每核占NAlign个元素
    savedWorkspaceAddr = workspace;
    savedPdGammaAddr = pdGamma;
    savedPdBetaAddr = pdBeta;
    int64_t workspaceOffset = blockIdx * NAlign;
    dbetaWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffset, NAlign);
    dgammaWorkspaceGM.SetGlobalBuffer((__gm__ float*)workspace + usedCoreNum * NAlign + workspaceOffset, NAlign);

    pipe_ = pipeIn;

    int64_t dyBufSize = NAlign * MAlign * sizeof(float);
    int64_t paramBufSize = MAlign * sizeof(float);
    int64_t outBufSize = NAlign * sizeof(float);
    pipe_->InitBuffer(inQueueDy, TRIPLE_BUFFER, dyBufSize);
    pipe_->InitBuffer(inQueueX, TRIPLE_BUFFER, dyBufSize);
    pipe_->InitBuffer(inQueueMean, TRIPLE_BUFFER, paramBufSize);
    pipe_->InitBuffer(inQueueRstd, TRIPLE_BUFFER, paramBufSize);
    pipe_->InitBuffer(outQueueDgamma, DOUBLE_BUFFER, outBufSize);
    pipe_->InitBuffer(outQueueDbeta, DOUBLE_BUFFER, outBufSize);

    int64_t cacheBufferCount = td_->gammaBetaCacheBufferCount;
    int64_t cacheBufSize = cacheBufferCount * NAlign * sizeof(float);
    pipe_->InitBuffer(dbetaCacheBuffer, cacheBufSize);
    pipe_->InitBuffer(dgammaCacheBuffer, cacheBufSize);
    pipe_->InitBuffer(reduceSumBuffer, DOUBLE_BUFFER * NAlign * sizeof(float));
    pipe_->InitBuffer(tmpDataBuffer, dyBufSize);
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3TransposeGammaBeta<T, PD_GAMMA_TYPE>::Process()
{
    if (GetBlockIdx() < td_->gammaBetaUsedCoreNum) {
        // stage 1: 核内累加
        Stage1();
    }

    SyncAll();

    if (GetBlockIdx() != 0) {
        return;
    }

    Stage2();
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3TransposeGammaBeta<T, PD_GAMMA_TYPE>::Stage1()
{
    dbetaCache = dbetaCacheBuffer.Get<float>();
    dgammaCache = dgammaCacheBuffer.Get<float>();

    int64_t pdgammaIsRequire = td_->pdgammaIsRequire;
    int64_t pdbetaIsRequire = td_->pdbetaIsRequire;
    int64_t loopCnt = basicBlockLoop ? basicBlockLoop : 1;

    for (int64_t i = 0; i < loopCnt; ++i) {
        int64_t curMfactor = (currentLoop != 0) ? MAlign : currentMTail;

        // 1. 搬运主块数据
        LocalTensor<float> mean;
        LocalTensor<float> rstd;
        CopyInData(i, curMfactor, dyMain_, xMain_, mean, rstd);

        // 2. 主块计算: dgamma VF计算 dy_mul_x_norm (结果放回x)
        if (pdgammaIsRequire) {
            ComputeDyMulXNormMain(dyMain_, xMain_, rstd, mean, NAlign, curMfactor, MAlign);
        }

        // 3. 折叠块: 将非2的幂部分累加到主块上
        if (basicBlockLoop != 0 &&
            ((i < mainFoldCount) || (i == mainFoldCount && currentMTail > 0 && currentLoop != 0))) {
            int64_t foldMfactor = (i < mainFoldCount) ? MAlign : currentMTail;
            int64_t foldIdx = i + basicBlockLoop; // 折叠块紧跟主块之后

            LocalTensor<float> foldDy;
            LocalTensor<float> foldX;
            LocalTensor<float> foldMean;
            LocalTensor<float> foldRstd;
            CopyInData(foldIdx, foldMfactor, foldDy, foldX, foldMean, foldRstd);

            // dbeta: dyFold加到dyMain
            if (pdbetaIsRequire) {
                VectorAdd(dyMain_, dyMain_, foldDy, NAlign, foldMfactor, MAlign);
            }
            // dgamma: dy*xNorm 并累加到xMain
            if (pdgammaIsRequire) {
                ComputeDyMulXNormFold(foldDy, foldX, foldRstd, foldMean, xMain_, NAlign, foldMfactor, MAlign);
            }

            inQueueMean.FreeTensor(foldMean);
            inQueueRstd.FreeTensor(foldRstd);
            inQueueDy.FreeTensor(foldDy);
            inQueueX.FreeTensor(foldX);
        }

        // 4. 求和环节: LastReduceSum + UpdateCache
        int64_t cacheID = GetCacheID(i);
        LocalTensor<float> reduceSumTensor = reduceSumBuffer.Get<float>();
        LocalTensor<float> dbetaSum = reduceSumTensor;
        LocalTensor<float> dgammaSum = reduceSumTensor[NAlign];
        LocalTensor<float> tmpData = tmpDataBuffer.Get<float>();

        if (pdbetaIsRequire) {
            LastReduceSum(dbetaSum, dyMain_, tmpData, NAlign, curMfactor, MAlign);
            UpdateCache(dbetaCache, dbetaSum, cacheID, NAlign, NAlign);
        }
        if (pdgammaIsRequire) {
            LastReduceSum(dgammaSum, xMain_, tmpData, NAlign, curMfactor, MAlign);
            UpdateCache(dgammaCache, dgammaSum, cacheID, NAlign, NAlign);
        }

        inQueueMean.FreeTensor(mean);
        inQueueRstd.FreeTensor(rstd);
        inQueueDy.FreeTensor(dyMain_);
        inQueueX.FreeTensor(xMain_);
    }

    // 结果搬出到GM workspace
    CopyOutData();
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3TransposeGammaBeta<T, PD_GAMMA_TYPE>::Stage2()
{
    // stage 2: core 0 做核间 ReduceSum RA
    int64_t usedCoreNum = td_->gammaBetaUsedCoreNum;
    dbetaWorkspaceGM.SetGlobalBuffer((__gm__ float*)savedWorkspaceAddr, usedCoreNum * NAlign);
    dgammaWorkspaceGM.SetGlobalBuffer((__gm__ float*)savedWorkspaceAddr + usedCoreNum * NAlign, usedCoreNum * NAlign);
    dgammaOutTensorGM.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)savedPdGammaAddr, N);
    dbetaOutTensorGM.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)savedPdBetaAddr, N);

    LocalTensor<float> tmpData = tmpDataBuffer.Get<float>();
    LocalTensor<float> reduceSumTensor = reduceSumBuffer.Get<float>();

    // dbeta: 从workspace搬运到x buffer, ReduceSum RA, cast, CopyOut
    if (td_->pdbetaIsRequire) {
        LocalTensor<float> dbetaData = inQueueX.template AllocTensor<float>();
        CopyIn(dbetaData, dbetaWorkspaceGM, usedCoreNum * NAlign);
        inQueueX.EnQue(dbetaData);
        dbetaData = inQueueX.template DeQue<float>();

        uint32_t srcShape[2] = {static_cast<uint32_t>(usedCoreNum), static_cast<uint32_t>(NAlign)};
        LocalTensor<PD_GAMMA_TYPE> dbetaOut = outQueueDbeta.template AllocTensor<PD_GAMMA_TYPE>();
        if constexpr (IsSameType<PD_GAMMA_TYPE, float>::value) {
            // 输出即float, ReduceSum直接写入dbetaOut
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(
                dbetaOut, dbetaData, tmpData.template ReinterpretCast<uint8_t>(), srcShape, false);
        } else {
            // 输出非float: 先归约到reduceSumTensor, 再cast到目标dtype并写入dbetaOut
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(
                reduceSumTensor, dbetaData, tmpData.template ReinterpretCast<uint8_t>(), srcShape, false);
            CopyUB2UBWithCast<PD_GAMMA_TYPE>(dbetaOut, reduceSumTensor, N);
        }
        inQueueX.FreeTensor(dbetaData);
        outQueueDbeta.EnQue(dbetaOut);
        dbetaOut = outQueueDbeta.template DeQue<PD_GAMMA_TYPE>();
        CopyOut<PD_GAMMA_TYPE>(dbetaOutTensorGM, dbetaOut, N);
        outQueueDbeta.FreeTensor(dbetaOut);
    }

    // dgamma与dbeta一致
    if (td_->pdgammaIsRequire) {
        LocalTensor<float> dgammaData = inQueueDy.template AllocTensor<float>();
        CopyIn(dgammaData, dgammaWorkspaceGM, usedCoreNum * NAlign);
        inQueueDy.EnQue(dgammaData);
        dgammaData = inQueueDy.template DeQue<float>();

        uint32_t srcShape[2] = {static_cast<uint32_t>(usedCoreNum), static_cast<uint32_t>(NAlign)};
        LocalTensor<PD_GAMMA_TYPE> dgammaOut = outQueueDgamma.template AllocTensor<PD_GAMMA_TYPE>();
        if constexpr (IsSameType<PD_GAMMA_TYPE, float>::value) {
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(
                dgammaOut, dgammaData, tmpData.template ReinterpretCast<uint8_t>(), srcShape, false);
        } else {
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(
                reduceSumTensor, dgammaData, tmpData.template ReinterpretCast<uint8_t>(), srcShape, false);
            CopyUB2UBWithCast<PD_GAMMA_TYPE>(dgammaOut, reduceSumTensor, N);
        }
        inQueueDy.FreeTensor(dgammaData);
        outQueueDgamma.EnQue(dgammaOut);
        dgammaOut = outQueueDgamma.template DeQue<PD_GAMMA_TYPE>();
        CopyOut<PD_GAMMA_TYPE>(dgammaOutTensorGM, dgammaOut, N);
        outQueueDgamma.FreeTensor(dgammaOut);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3TransposeGammaBeta<T, PD_GAMMA_TYPE>::CopyInData(
    const int64_t mi, const int64_t mfactor, LocalTensor<float>& dyOut, LocalTensor<float>& xOut,
    LocalTensor<float>& meanOut, LocalTensor<float>& rstdOut)
{
    int64_t offset = mi * MAlign;
    int64_t gmOffset = offset * N;

    meanOut = inQueueMean.template AllocTensor<float>();
    CopyIn(meanOut, meanInTensorGM[offset], mfactor);
    inQueueMean.EnQue(meanOut);
    meanOut = inQueueMean.template DeQue<float>();

    rstdOut = inQueueRstd.template AllocTensor<float>();
    CopyIn(rstdOut, rstdInTensorGM[offset], mfactor);
    inQueueRstd.EnQue(rstdOut);
    rstdOut = inQueueRstd.template DeQue<float>();

    LocalTensor<float> dyMain = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyInTranspose<T>(dyMain.ReinterpretCast<T>(), dyInTensorGM[gmOffset], mfactor, N, MAlign);
        inQueueDy.EnQue(dyMain);
        dyOut = inQueueDy.template DeQue<float>();
    } else {
        // 非float: 搬入后原地Cast到fp32
        LocalTensor<T> dyCastTemp = dyMain.template ReinterpretCast<T>()[MAlign];
        CopyInTranspose<T>(dyCastTemp, dyInTensorGM[gmOffset], mfactor, N, 2 * MAlign);
        inQueueDy.EnQue(dyMain);
        dyOut = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyOut, dyCastTemp, NAlign, mfactor, MAlign);
    }

    LocalTensor<float> xMain = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyInTranspose<T>(xMain.ReinterpretCast<T>(), xInTensorGM[gmOffset], mfactor, N, MAlign);
        inQueueX.EnQue(xMain);
        xOut = inQueueX.template DeQue<float>();
    } else {
        LocalTensor<T> xCastTemp = xMain.template ReinterpretCast<T>()[MAlign];
        CopyInTranspose<T>(xCastTemp, xInTensorGM[gmOffset], mfactor, N, 2 * MAlign);
        inQueueX.EnQue(xMain);
        xOut = inQueueX.template DeQue<float>();
        CastToFp32From<T>(xOut, xCastTemp, NAlign, mfactor, MAlign);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3TransposeGammaBeta<T, PD_GAMMA_TYPE>::CopyOutData()
{
    if (td_->pdbetaIsRequire) {
        LocalTensor<float> dbetaResult = outQueueDbeta.template AllocTensor<float>();
        CopyUB2UB(dbetaResult, dbetaCache[resultCacheID * NAlign], NAlign);
        outQueueDbeta.EnQue(dbetaResult);
        dbetaResult = outQueueDbeta.template DeQue<float>();
        CopyOut<float>(dbetaWorkspaceGM, dbetaResult, NAlign);
        outQueueDbeta.FreeTensor(dbetaResult);
    }

    if (td_->pdgammaIsRequire) {
        LocalTensor<float> dgammaResult = outQueueDgamma.template AllocTensor<float>();
        CopyUB2UB(dgammaResult, dgammaCache[resultCacheID * NAlign], NAlign);
        outQueueDgamma.EnQue(dgammaResult);
        dgammaResult = outQueueDgamma.template DeQue<float>();
        CopyOut<float>(dgammaWorkspaceGM, dgammaResult, NAlign);
        outQueueDgamma.FreeTensor(dgammaResult);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3TransposeGammaBeta<T, PD_GAMMA_TYPE>::ComputeDyMulXNormMain(
    const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor, const LocalTensor<float>& rstdTensor,
    const LocalTensor<float>& meanTensor, const int64_t rowSize, const int64_t colSize, const int64_t stride)
{
    if (rowSize <= 0 || colSize <= 0) {
        return;
    }
    uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
    uint16_t innerLoopTimes = static_cast<uint16_t>(
        CeilDiv(static_cast<int64_t>(colSize * sizeof(float)), static_cast<int64_t>(GetVRegSize())));
    uint32_t outerLoopStride = static_cast<uint32_t>(stride);
    uint32_t innerLoopStride = VL_FP32;
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ float* x = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            __ubuf__ float* dy = (__ubuf__ float*)dyTensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(colSize);
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::RegTensor<float> xReg, dyReg, meanReg, rstdReg, tmpReg, resultReg;
            LoadAlign(meanReg, mean);
            LoadAlign(rstdReg, rstd);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                LoadAlign(xReg, x + i * outerLoopStride);
                LoadAlign(dyReg, dy + i * outerLoopStride);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(tmpReg, xReg, meanReg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, tmpReg, rstdReg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, resultReg, dyReg, pMask);
                StoreAlign(x + i * outerLoopStride, resultReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* x = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            __ubuf__ float* dy = (__ubuf__ float*)dyTensor.GetPhyAddr();
            AscendC::Reg::MaskReg pMask;
            AscendC::Reg::RegTensor<float> xReg, dyReg, meanReg, rstdReg, tmpReg, resultReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                uint32_t count = static_cast<uint32_t>(colSize);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    LoadAlign(meanReg, mean + j * innerLoopStride);
                    LoadAlign(rstdReg, rstd + j * innerLoopStride);
                    LoadAlign(xReg, x + i * outerLoopStride + j * innerLoopStride);
                    LoadAlign(dyReg, dy + i * outerLoopStride + j * innerLoopStride);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(tmpReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, tmpReg, rstdReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, resultReg, dyReg, pMask);
                    StoreAlign(x + i * outerLoopStride + j * innerLoopStride, resultReg, pMask);
                }
            }
        }
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3TransposeGammaBeta<T, PD_GAMMA_TYPE>::ComputeDyMulXNormFold(
    const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor, const LocalTensor<float>& rstdTensor,
    const LocalTensor<float>& meanTensor, LocalTensor<float>& xMainTensor, const int64_t rowSize, const int64_t colSize,
    const int64_t stride)
{
    if (rowSize <= 0 || colSize <= 0) {
        return;
    }
    uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
    uint16_t innerLoopTimes = static_cast<uint16_t>(
        CeilDiv(static_cast<int64_t>(colSize * sizeof(float)), static_cast<int64_t>(GetVRegSize())));
    uint32_t outerLoopStride = static_cast<uint32_t>(stride);
    uint32_t innerLoopStride = VL_FP32;
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ float* x = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            __ubuf__ float* dy = (__ubuf__ float*)dyTensor.GetPhyAddr();
            __ubuf__ float* xMain = (__ubuf__ float*)xMainTensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(colSize);
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::RegTensor<float> xReg, dyReg, meanReg, rstdReg, tmpReg, resultReg, xMainReg;
            LoadAlign(meanReg, mean);
            LoadAlign(rstdReg, rstd);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                LoadAlign(xReg, x + i * outerLoopStride);
                LoadAlign(dyReg, dy + i * outerLoopStride);
                LoadAlign(xMainReg, xMain + i * outerLoopStride);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(tmpReg, xReg, meanReg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, tmpReg, rstdReg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, resultReg, dyReg, pMask);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, xMainReg, resultReg, pMask);
                StoreAlign(xMain + i * outerLoopStride, resultReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* x = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            __ubuf__ float* dy = (__ubuf__ float*)dyTensor.GetPhyAddr();
            __ubuf__ float* xMain = (__ubuf__ float*)xMainTensor.GetPhyAddr();
            AscendC::Reg::MaskReg pMask;
            AscendC::Reg::RegTensor<float> xReg, dyReg, meanReg, rstdReg, tmpReg, resultReg, xMainReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                uint32_t count = static_cast<uint32_t>(colSize);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    LoadAlign(meanReg, mean + j * innerLoopStride);
                    LoadAlign(rstdReg, rstd + j * innerLoopStride);
                    LoadAlign(xReg, x + i * outerLoopStride + j * innerLoopStride);
                    LoadAlign(dyReg, dy + i * outerLoopStride + j * innerLoopStride);
                    LoadAlign(xMainReg, xMain + i * outerLoopStride + j * innerLoopStride);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(tmpReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, tmpReg, rstdReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, resultReg, dyReg, pMask);
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(resultReg, xMainReg, resultReg, pMask);
                    StoreAlign(xMain + i * outerLoopStride + j * innerLoopStride, resultReg, pMask);
                }
            }
        }
    }
}

} // namespace LayerNormGradV3
#endif // LAYER_NORM_GRAD_V3_TRANSPOSE_GAMMA_BETA_IMPL_
