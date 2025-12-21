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
 * \file avg_pool3d_ncdhw_big_kernel.h
 * \brief
 */

#ifndef AVG_POOL3D_NCDHW_BIG_KERNEL_H_
#define AVG_POOL3D_NCDHW_BIG_KERNEL_H_

#include "kernel_operator.h"
#include "avg_pool3d_common.h"

namespace AvgPool3d {
template <typename T>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        LocalTensor<T>& xLocal, LocalTensor<float>& castToFP32, LocalTensor<float>& sumBufLocal, uint32_t dataCount)
    {
        if constexpr (sizeof(T) == sizeof(float)) {
            Adds(castToFP32, xLocal, 0.0f, dataCount);
        } else {
            Cast(castToFP32, xLocal, RoundMode::CAST_NONE, dataCount);
        }
        PipeBarrier<PIPE_V>();
        Add(sumBufLocal, sumBufLocal, castToFP32, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

template <typename T, int32_t QUEUE_DEPTH>
class KernelAvgPool3dBigKernel {
public:
    __aicore__ inline KernelAvgPool3dBigKernel(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe_in);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Prepare(int64_t curIdx);
    __aicore__ inline void BaseCompute(int64_t curIdx);
    __aicore__ inline int64_t dhwCopyInput(int64_t offset, int64_t blockLen);
    __aicore__ inline int64_t hwCopyInput(int64_t offset, int64_t blockLen, int64_t blockCount);
    __aicore__ inline int64_t wCopyInput(int64_t offset, int64_t blockLen, int64_t blockCount, int64_t dLen);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline float CalPoolSize(int64_t curIdx);
    __aicore__ inline void ReduceMeanWindow(int64_t curIdx);
    __aicore__ inline void CopyAvgOut(int64_t curIdx);
    __aicore__ inline void dhwContinueProcess();
    __aicore__ inline void hwContinueProcess();
    __aicore__ inline void wContinueProcess();

    TPipe* pipe;
    // 输入队列
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQue;
    // fp16、bf16转换结果
    TBuf<> inputCastUB;
    // sumBuf累加空间
    TBuf<> sumBuf;
    LocalTensor<float> sumBufLocal;
    // 平均值ub
    TBuf<> avgUB;

    GlobalTensor<T> xGm;
    GlobalTensor<T> avgGm;

    const AvgPool3DTilingData* tilingData;

    uint32_t cBlockIdx;
    // shape info
    int64_t nc_ = 1;
    int64_t d_ = 1;
    int64_t h_ = 1;
    int64_t w_ = 1;
    int64_t dout_ = 1;
    int64_t ho_ = 1;
    int64_t wo_ = 1;
    int64_t kd_ = 1;
    int64_t kh_ = 1;
    int64_t kw_ = 1;
    int64_t sd_ = 1;
    int64_t sh_ = 1;
    int64_t sw_ = 1;
    int64_t pd_ = 1;
    int64_t ph_ = 1;
    int64_t pw_ = 1;
    int64_t divisorOverride_ = 0;
    int64_t countIncludePad_ = 0;

    // tiling info
    int64_t blockFactor_ = 1;
    int64_t blockTail_ = 1;
    int64_t coreNums = 1;
    int64_t totalIdx_ = 1;

    int64_t inHW_ = 1;
    int64_t inDHW_ = 1;
    int64_t outHW_ = 1;
    int64_t outDHW_ = 1;
    int64_t curNc = 0;
    int64_t curOriginD_ = 0;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t curkD_ = 1;
    int64_t curkH_ = 1;
    int64_t curkW_ = 1;
    int64_t curInOffset_ = 0;
    T padValueT_ = 0;
    int32_t maxCount_ = 10 * 1024;

    constexpr static int64_t BLOCK_NUM_T = BLOCK_SIZE / sizeof(T);
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* __restrict__ tiling, TPipe* pipe_in)
{
    pipe = pipe_in;

    // base info
    cBlockIdx = GetBlockIdx();

    // shape info
    d_ = tiling->inD;
    h_ = tiling->inH;
    w_ = tiling->inW;
    dout_ = tiling->outD;
    ho_ = tiling->outH;
    wo_ = tiling->outW;
    kd_ = tiling->kD;
    kh_ = tiling->kH;
    kw_ = tiling->kW;
    sd_ = tiling->dD;
    sh_ = tiling->dH;
    sw_ = tiling->dW;
    pd_ = tiling->pD;
    ph_ = tiling->pH;
    pw_ = tiling->pW;
    divisorOverride_ = tiling->divisorOverride;
    countIncludePad_ = tiling->countIncludePad;

    // tiling info
    uint32_t usedCoreNum_ = tiling->usedCoreNum;
    totalIdx_ = tiling->inN * tiling->inC * tiling->outD * tiling->outH * tiling->outW;
    blockFactor_ = totalIdx_ / usedCoreNum_;
    blockTail_ = totalIdx_ % usedCoreNum_;
    inHW_ = h_ * w_;
    inDHW_ = d_ * inHW_;
    outHW_ = ho_ * wo_;
    outDHW_ = dout_ * outHW_;
    padValueT_ = 0.0f;

    // GM
    xGm.SetGlobalBuffer((__gm__ T*)x);
    avgGm.SetGlobalBuffer((__gm__ T*)y);

    pipe->InitBuffer(inputQue, QUEUE_DEPTH, maxCount_ * sizeof(T));
    pipe->InitBuffer(inputCastUB, maxCount_ * sizeof(float));
    pipe->InitBuffer(sumBuf, maxCount_ * sizeof(float));
    pipe->InitBuffer(avgUB, 256);

    sumBufLocal = sumBuf.Get<float>();
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::Process()
{
    // init indices
    int64_t beginIdx = 0;
    int64_t endIdx = 0;
    if (cBlockIdx < blockTail_) {
        beginIdx = cBlockIdx * (blockFactor_ + 1);
        endIdx = beginIdx + blockFactor_ + 1;
    } else {
        beginIdx = cBlockIdx * blockFactor_ + blockTail_;
        endIdx = beginIdx + blockFactor_;
    }

    if (dout_ == 1 && ho_ == 1 && wo_ == 1) {
        curOriginD_ = 0;
        curOriginH_ = 0;
        curOriginW_ = 0;
        curOriginIndex_ = 0;
        curkD_ = Min(kd_ - pd_, d_);
        curkH_ = Min(kh_ - ph_, h_);
        curkW_ = Min(kw_ - pw_, w_);
    }

    // current blockdim range
    for (int64_t idx = beginIdx; idx < endIdx; idx++) {
        Prepare(idx);
        BaseCompute(idx);
    }
}

/*
 * 功能：动态计算输出点curIdx所对应的kernel size大小
 */
template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::Prepare(int64_t curIdx)
{
    if (dout_ == 1 && ho_ == 1 && wo_ == 1) {
        curNc = curIdx;
        curInOffset_ = curNc * inDHW_;
        return;
    }

    int64_t cur3D = curIdx % outDHW_; // 单个output中的输出点index
    int64_t cur2D = cur3D % outHW_;   // 单个output中的d维中的index

    int64_t curNc = curIdx / outDHW_; // curIdx所属nc前的nc数量
    int64_t curDo = cur3D / outHW_;   // curIdx所属d前的d数量
    int64_t curHo = cur2D / wo_;      // curIdx所属h前的h数量
    int64_t curWo = cur2D % wo_;      // curIdx的w维度位置

    // 计算输出点curIdx对应的kernel在原nc中的起始位置
    curOriginD_ = sd_ * curDo - pd_;
    if (curOriginD_ < 0) {
        curkD_ = Min(kd_ + curOriginD_, d_);
        curOriginD_ = 0;
    } else {
        curkD_ = Min(d_ - curOriginD_, kd_);
    }

    curOriginH_ = sh_ * curHo - ph_;
    if (curOriginH_ < 0) {
        curkH_ = Min(kh_ + curOriginH_, h_);
        curOriginH_ = 0;
    } else {
        curkH_ = Min(h_ - curOriginH_, kh_);
    }

    curOriginW_ = sw_ * curWo - pw_;
    if (curOriginW_ < 0) {
        curkW_ = Min(kw_ + curOriginW_, w_);
        curOriginW_ = 0;
    } else {
        curkW_ = Min(w_ - curOriginW_, kw_);
    }
    // 计算输出点curIdx对应的kernel在原nc数据中的起始index
    curOriginIndex_ = curOriginD_ * inHW_ + curOriginH_ * w_ + curOriginW_;
    // 计算输出点curIdx对应的kernel在原数据中的index，作为拷贝数据的起始偏移量
    curInOffset_ = curNc * inDHW_ + curOriginIndex_;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::BaseCompute(int64_t curIdx)
{
    // 1、sumBufLocal clear
    Duplicate(sumBufLocal, 0.0f, maxCount_);
    PipeBarrier<PIPE_V>();

    // 2、copyIn and sum gm to sumBufLocal
    if (curkW_ == w_ && curkH_ == h_) {
        dhwContinueProcess();
    } else if (curkW_ == w_ && curkH_ != h_) {
        hwContinueProcess();
    } else if (curkW_ != w_) {
        wContinueProcess();
    }

    // 3、ReduceSum and Mean
    ReduceMeanWindow(curIdx);

    // 4、copyOut avg value to gm
    CopyAvgOut(curIdx);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline int64_t KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::dhwCopyInput(int64_t offset, int64_t blockLen)
{
    LocalTensor<T> xLocal = inputQue.template AllocTensor<T>();
    int64_t blockLenAlign = CeilValue(blockLen, BLOCK_NUM_T);
    int64_t alignNum = blockLenAlign - blockLen;
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = alignNum != 0;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = padExtParams.isPad ? alignNum : 0;
    padExtParams.paddingValue = padValueT_;

    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[offset], extParams, padExtParams);
    inputQue.EnQue(xLocal);
    return blockLenAlign;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline int64_t KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::hwCopyInput(
    int64_t offset, int64_t blockLen, int64_t blockCount)
{
    LocalTensor<T> xLocal = inputQue.template AllocTensor<T>();
    int64_t blockLenAlign = CeilValue(blockLen, BLOCK_NUM_T);
    int64_t alignNum = blockLenAlign - blockLen;
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = alignNum != 0;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = padExtParams.isPad ? alignNum : 0;
    padExtParams.paddingValue = padValueT_;

    if (UINT32_MAX >= (inHW_ - blockLen) * sizeof(T)) {
        DataCopyExtParams extParams;
        extParams.blockCount = blockCount;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = (inHW_ - blockLen) * sizeof(T);
        extParams.dstStride = 0;
        DataCopyPad(xLocal, xGm[offset], extParams, padExtParams);
    } else {
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        for (int i = 0; i < blockCount; i++) {
            DataCopyPad(xLocal[i * blockLenAlign], xGm[offset + i * inHW_], extParams, padExtParams);
        }
    }
    inputQue.EnQue(xLocal);
    return blockLenAlign * blockCount;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline int64_t KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::wCopyInput(
    int64_t offset, int64_t blockLen, int64_t blockCount, int64_t dLen)
{
    LocalTensor<T> xLocal = inputQue.template AllocTensor<T>();
    int64_t blockLenAlign = CeilValue(blockLen, BLOCK_NUM_T);
    int64_t alignNum = blockLenAlign - blockLen;
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = alignNum != 0;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = padExtParams.isPad ? alignNum : 0;
    padExtParams.paddingValue = padValueT_;

    if (UINT32_MAX >= (w_ - blockLen) * sizeof(T)) {
        DataCopyExtParams extParams;
        extParams.blockCount = blockCount;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = (w_ - blockLen) * sizeof(T);
        extParams.dstStride = 0;
        for (int i = 0; i < dLen; i++) {
            DataCopyPad(xLocal[i * blockLenAlign * blockCount], xGm[offset + i * inHW_], extParams, padExtParams);
        }
    } else {
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        for (int i = 0; i < dLen * blockCount; i++) {
            DataCopyPad(xLocal[i * blockLenAlign], xGm[offset + i * w_], extParams, padExtParams);
        }
    }
    inputQue.EnQue(xLocal);
    return blockLenAlign * blockCount * dLen;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::Compute(int64_t dataCount)
{
    LocalTensor<T> xLocal = inputQue.template DeQue<T>();
    LocalTensor<float> castToFP32 = inputCastUB.Get<float>();
    InnerComputer<T> computer;
    computer.Compute(xLocal, castToFP32, sumBufLocal, dataCount);
    inputQue.template FreeTensor<T>(xLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline float KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::CalPoolSize(int64_t curIdx)
{
    Index index;
    uint64_t outputDIdx = curIdx % dout_;
    index.D.Compute(outputDIdx, d_, kd_, sd_, pd_, countIncludePad_);
    uint64_t outputHIdx = curIdx % ho_;
    index.H.Compute(outputHIdx, h_, kh_, sh_, ph_, countIncludePad_);
    uint64_t outputWIdx = curIdx % wo_;
    index.W.Compute(outputWIdx, w_, kw_, sw_, pw_, countIncludePad_);
    int64_t poolSize = divisorOverride_ ? divisorOverride_ : index.D.poolSize * index.H.poolSize * index.W.poolSize;
    float factor = 1.0f / static_cast<float>(poolSize);
    SToVSync();
    return factor;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::ReduceMeanWindow(int64_t curIdx)
{
    float factor = CalPoolSize(curIdx);
    LocalTensor<float> avgOutLocal = avgUB.Get<float>();

    // ReduceSum
    LocalTensor<float> reduceWorkLocalUB = inputCastUB.Get<float>();
    ReduceSum<float>(avgOutLocal, sumBufLocal, reduceWorkLocalUB, maxCount_);
    PipeBarrier<PIPE_V>();

    // divided after ReduceSum
    Muls<float>(avgOutLocal, avgOutLocal, factor, 1);
    PipeBarrier<PIPE_V>();
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::CopyAvgOut(int64_t curIdx)
{
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = 1 * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;

    // cast to T
    LocalTensor<float> avgOutLocal = avgUB.Get<float>();
    LocalTensor<T> castToT = inputCastUB.Get<T>();

    if constexpr (sizeof(T) == sizeof(float)) {
        castToT = avgOutLocal;
    } else {
        Cast(castToT, avgOutLocal, RoundMode::CAST_ROUND, 1);
    }
    PipeBarrier<PIPE_V>();

    VToMTE3Sync();
    DataCopyPad(avgGm[curIdx], castToT, extParams);
    MTE3ToMTE2Sync();
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::dhwContinueProcess()
{
    int64_t curkDHW = curkD_ * curkH_ * curkW_;
    int64_t AlignkDHW = CeilValue(curkDHW, BLOCK_NUM_T);
    int64_t inputOffset = curInOffset_;
    int64_t kernelOffset = 0;
    if (AlignkDHW <= maxCount_) {
        int32_t dataCount = dhwCopyInput(curInOffset_, curkDHW);
        Compute(dataCount);
    } else {
        int64_t dhwLoops = (curkDHW + maxCount_ - 1) / maxCount_;
        int32_t dhwFactor = maxCount_;
        int32_t dhwTail = curkDHW % maxCount_;
        if (dhwTail == 0) {
            dhwTail = dhwFactor;
        }
        for (int32_t dhwLoop = 0; dhwLoop < dhwLoops; dhwLoop++) {
            int32_t curFactor = dhwLoop == dhwLoops - 1 ? dhwTail : dhwFactor;
            int32_t dataCount = dhwCopyInput(inputOffset, curFactor);
            Compute(dataCount);
            inputOffset += curFactor;
            kernelOffset += curFactor;
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::hwContinueProcess()
{
    int64_t curkHW = curkH_ * curkW_;
    int64_t AlignkHW = CeilValue(curkHW, BLOCK_NUM_T);
    int64_t alignNum = 0;
    int64_t blockLenAlign = 0;
    int64_t inputOffset = curInOffset_;
    int64_t kernelOffset = 0;
    if (curkD_ * AlignkHW <= maxCount_) {
        int32_t dataCount = hwCopyInput(curInOffset_, curkHW, curkD_);
        Compute(dataCount);
    } else {
        if (AlignkHW > maxCount_) {
            int32_t dLoops = curkD_;
            int32_t dFactor = 1;
            int32_t dTail = 0;
            int32_t hwLoops = (curkHW + maxCount_ - 1) / maxCount_;
            int32_t hwFactor = maxCount_;
            int32_t hwTail = curkHW % maxCount_;
            if (hwTail == 0) {
                hwTail = hwFactor;
            }

            for (int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
                for (int32_t hwLoop = 0; hwLoop < hwLoops; hwLoop++) {
                    int32_t curFactor = hwLoop == hwLoops - 1 ? hwTail : hwFactor;
                    int32_t dataCount = hwCopyInput(inputOffset, curFactor, 1);
                    Compute(dataCount);
                    inputOffset += curFactor;
                    kernelOffset += curFactor;
                }
                inputOffset = curInOffset_ + (dLoop + 1) * inHW_;
            }
        } else {
            int32_t dFactor = maxCount_ / AlignkHW;
            int32_t dLoops = (curkD_ + dFactor - 1) / dFactor;
            int32_t dTail = curkD_ % dFactor;
            if (dTail == 0) {
                dTail = dFactor;
            }
            int32_t hwLoops = 1;
            int32_t hwFactor = curkHW;
            int32_t hwTail = 0;
            for (int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
                int32_t curdFactor = dLoop == dLoops - 1 ? dTail : dFactor;
                int32_t dataCount = hwCopyInput(inputOffset, hwFactor, curdFactor);
                Compute(dataCount);
                inputOffset = curInOffset_ + curdFactor * inHW_ * (dLoop + 1);
                kernelOffset += curdFactor * hwFactor;
            }
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dBigKernel<T, QUEUE_DEPTH>::wContinueProcess()
{
    int64_t curkHW = curkH_ * curkW_;
    int64_t AlignkW = CeilValue(curkW_, BLOCK_NUM_T);
    int64_t alignNum = 0;
    int64_t blockLenAlign = 0;
    int64_t inputOffset = curInOffset_;
    int64_t kernelOffset = 0;
    if (curkD_ * curkH_ * AlignkW <= maxCount_) {
        int32_t dataCount = wCopyInput(curInOffset_, curkW_, curkH_, curkD_);
        Compute(dataCount);
    } else {
        if (AlignkW > maxCount_) {
            int64_t dLoops = curkD_;
            int64_t dFactor = 1;
            int64_t dTail = 0;
            int64_t hLoops = curkH_;
            int64_t hFactor = 1;
            int64_t hTail = 0;
            int64_t wFactor = maxCount_;
            int64_t wLoops = (curkW_ + wFactor - 1) / wFactor;
            int64_t wTail = curkW_ % wFactor;
            if (wTail == 0) {
                wTail = wFactor;
            }
            for (int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
                for (int32_t hLoop = 0; hLoop < hLoops; hLoop++) {
                    inputOffset = curInOffset_ + dLoop * inHW_ + hLoop * w_;
                    for (int32_t wLoop = 0; wLoop < wLoops; wLoop++) {
                        int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
                        int32_t dataCount = dhwCopyInput(inputOffset, curFactor);
                        Compute(dataCount);
                        inputOffset += curFactor;
                        kernelOffset += curFactor;
                    }
                }
            }
        } else if (AlignkW * curkH_ > maxCount_) {
            int64_t dLoops = curkD_;
            int64_t dFactor = 1;
            int64_t dTail = 0;
            int64_t hFactor = maxCount_ / AlignkW;
            int64_t hLoops = (curkH_ + hFactor - 1) / hFactor;
            int64_t hTail = curkH_ % hFactor;
            if (hTail == 0) {
                hTail = hFactor;
            }
            int64_t wLoops = 1;
            int64_t wFactor = curkW_;
            int64_t wTail = 0;
            for (int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
                inputOffset = curInOffset_ + dLoop * inHW_;
                for (int32_t hLoop = 0; hLoop < hLoops; hLoop++) {
                    int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
                    int32_t curFactor = hLoop == hLoops - 1 ? hTail * wFactor : hFactor * wFactor;
                    int32_t dataCount = wCopyInput(inputOffset, wFactor, curhFactor, 1);
                    Compute(dataCount);
                    inputOffset += curhFactor * w_;
                    kernelOffset += curFactor;
                }
            }
        } else if (AlignkW * curkH_ * curkD_ > maxCount_) {
            int64_t dFactor = maxCount_ / (AlignkW * curkH_);
            int64_t dLoops = (curkD_ + dFactor - 1) / dFactor;
            int64_t dTail = curkD_ % dFactor;
            if (dTail == 0) {
                dTail = dFactor;
            }
            int64_t hLoops = 1;
            int64_t hFactor = curkH_;
            int64_t hTail = 0;
            int64_t wLoops = 1;
            int64_t wFactor = curkW_;
            int64_t wTail = 0;
            for (int32_t dLoop = 0; dLoop < dLoops; dLoop++) {
                int32_t curdFactor = dLoop == dLoops - 1 ? dTail : dFactor;
                int32_t curFactor = dLoop == dLoops - 1 ? dTail * hFactor * wFactor : dFactor * hFactor * wFactor;
                int32_t dataCount = wCopyInput(inputOffset, wFactor, hFactor, curdFactor);
                Compute(dataCount);
                inputOffset = curInOffset_ + curdFactor * inHW_ * (dLoop + 1);
                kernelOffset += curFactor;
            }
        }
    }
}
} // namespace AvgPool3d

#endif // AVG_POOL3D_NCDHW_BIG_KERNEL_H_
