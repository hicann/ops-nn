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
 * \file pool_3d_ndhwc_big_channel_pad.h
 * \brief
 */
#ifndef MAX_POOL_V3_NDHWC_BIG_CHANNEL_PAD_H_
#define MAX_POOL_V3_NDHWC_BIG_CHANNEL_PAD_H_

#include "copy_pad_impl.h"
#include "pool_3d_common.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"

namespace Pool3D
{

using namespace AscendC;

template <typename T, int32_t OP_TYPE, bool OUT_DIV=false>
class Pool3dNDHWCBigChannelPad
{
public:
    __aicore__ inline Pool3dNDHWCBigChannelPad(TPipe* pipe, const Pool3DSmallKernelNDHWCTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <typename M, typename U>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void InitDivisor();
    __aicore__ inline void CopyInMultiChannels(int64_t offset, TensorDescInfo &inputInfo);
    __aicore__ inline void CopyOutMultiChannels(int64_t offset, int64_t n, int64_t deps,
                                                int64_t rows, int64_t cols, int64_t channels);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleChannels(int64_t n, int64_t inDeps, int64_t inRows, int64_t inCols, int64_t outDeps,
                                                 int64_t outRows, int64_t outCols, int64_t expectDepStart,
                                                 int64_t expectRowStart, int64_t expectColStart, int64_t realDeps,
                                                 int64_t realRows, int64_t realCols, int64_t alignChannels);
    template <typename M, typename U>
    __aicore__ inline void CopyPadData(LocalTensor<T>& inLocal, LocalTensor<T>& outLocal, int64_t n, int64_t inDeps,
                                       int64_t inRows, int64_t cols, int64_t expectDepStart, int64_t expectRowStart,
                                       int64_t expectColStart, int64_t realDeps, int64_t realRows, int64_t realCols,
                                       int64_t alignChannels);
    __aicore__ inline void ComputeDivisor(int64_t start, int64_t num) ;
    __aicore__ inline void ComputeCurrentDivisor(uint32_t num);
    __aicore__ inline uint32_t GetCurrentDivisorIndex(uint32_t index);
    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // 输出ub
    TQue<QuePosition::VECOUT, BUFFER_NUM> maxUBOutput_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> scatterIndexBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;
    const Pool3DSmallKernelNDHWCTilingData* tilingData_;
    TBuf<QuePosition::VECCALC> divisorBuf_;
    bool needDivisorBuf_ = false;
    int64_t curOffsetInBatch_ = 0;
};

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::InitDivisor()
{   
    if (OUT_DIV && tilingData_->realCalcDivisor == 0 && needDivisorBuf_) {
        int64_t length = max(static_cast<int64_t>(tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim),
                             static_cast<int64_t>(platform::GetUbBlockSize() / sizeof(float)));
        ComputeDivisor(0, length);
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    needDivisorBuf_ = OP_TYPE == OP_TYPE_AVG_POOL_3D && tilingData_->divisor == 0;
    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, INDEX_SIZE);
    pipe_->InitBuffer(tmpBuf_, tilingData_->inUbSize * sizeof(T));
    if constexpr (OUT_DIV) {
        pipe_->InitBuffer(divisorBuf_, tilingData_->divisorUbSize);
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::Process()
{
    using indiceType = typename IndexTypeGet<T>::type;
    using computType = typename GetComputeType<T>::type;
    InitDivisor();
    BaseCompute<computType, indiceType>();
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename M, typename U>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::BaseCompute()
{
    int64_t sW = tilingData_->sW;
    int64_t sH = tilingData_->sH;
    int64_t sD = tilingData_->sD;
    uint32_t channels = tilingData_->channels;
    uint32_t outUbFactorW = tilingData_->outUbFactorW;
    uint32_t outUbFactorH = tilingData_->outUbFactorH;
    uint32_t outUbFactorD = tilingData_->outUbFactorD;
    uint32_t effectiveKd = (tilingData_->kD - 1) * tilingData_->dD + 1;
    uint32_t effectiveKh = (tilingData_->kH - 1) * tilingData_->dH + 1;
    uint32_t effectiveKw = (tilingData_->kW - 1) * tilingData_->dW + 1;
    int64_t startIdx = 0;
    int64_t endIdx = 0;
    if (GetBlockIdx() < tilingData_->blockTail) {
        startIdx = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx = startIdx + tilingData_->blockFactor + 1;
    } else {
        startIdx = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx = startIdx + tilingData_->blockFactor;
    }
    uint32_t alignChannels = ops::Aligned(channels, static_cast<uint32_t>(platform::GetUbBlockSize() / sizeof(T)));
    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t nIdx = idx / (tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop);
        int64_t dIdx = (idx - nIdx * tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop) /
                       (tilingData_->hLoop * tilingData_->wLoop);
        int64_t hIdx = (idx - nIdx * tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop -
                        dIdx * tilingData_->hLoop * tilingData_->wLoop) /
                       tilingData_->wLoop;
        int64_t wIdx = idx % tilingData_->wLoop;
        uint32_t n = nIdx == tilingData_->nLoop - 1 ? tilingData_->nOutDim - nIdx * tilingData_->ubFactorN
                                                   : tilingData_->ubFactorN;
        int64_t deps = dIdx == tilingData_->dLoop - 1 ?  tilingData_->dOutDim - dIdx * tilingData_->outUbFactorD
                                                    : tilingData_->outUbFactorD;
        int64_t rows = hIdx == tilingData_->hLoop - 1 ? tilingData_->hOutDim - hIdx * outUbFactorH : outUbFactorH;
        int64_t cols = wIdx == tilingData_->wLoop - 1 ? tilingData_->wOutDim - wIdx * outUbFactorW : outUbFactorW;
        int64_t expectDepStart = dIdx * sD * outUbFactorD;
        int64_t expectRowStart = hIdx * sH * outUbFactorH;
        int64_t expectColStart = wIdx * sW * outUbFactorW;
        int64_t dUpper =
            min(dIdx * sD * outUbFactorD + (deps - 1) * sD + effectiveKd - tilingData_->fPad, tilingData_->dInDim);
        int64_t dLower = max(dIdx * sD * outUbFactorD - tilingData_->fPad, (int64_t)0);
        int64_t hUpper =
            min(hIdx * sH * outUbFactorH + (rows - 1) * sH + effectiveKh - tilingData_->tPad, tilingData_->hInDim);
        int64_t hLower = max(hIdx * sH * outUbFactorH - tilingData_->tPad, (int64_t)0);
        int64_t wUpper =
            min(wIdx * sW * outUbFactorW + (cols - 1) * sW + effectiveKw - tilingData_->lPad, tilingData_->wInDim);
        int64_t wLower = max(wIdx * sW * outUbFactorW - tilingData_->lPad, (int64_t)0);
        uint32_t realDeps = tilingData_->splitMode == SPLIT_BATCHS ? tilingData_->dInDim : dUpper - dLower;
        uint32_t realRows = tilingData_->splitMode == SPLIT_BATCHS || tilingData_->splitMode == SPLIT_DEPTHS ?
                            tilingData_->hInDim : hUpper - hLower;
        uint32_t realCols = tilingData_->splitMode != SPLIT_COLS ? tilingData_->wInDim : wUpper - wLower;

        int64_t srcOffset =
            nIdx * tilingData_->ubFactorN * tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim * channels +
            dLower * tilingData_->hInDim * tilingData_->wInDim * channels + hLower * tilingData_->wInDim * channels +
            wLower * channels;
        int64_t dstOffset = nIdx * tilingData_->ubFactorN * tilingData_->dOutDim * tilingData_->hOutDim *
                                tilingData_->wOutDim * channels +
                            dIdx * outUbFactorD * tilingData_->hOutDim * tilingData_->wOutDim * channels +
                            hIdx * outUbFactorH * tilingData_->wOutDim * channels + wIdx * outUbFactorW * channels;
        uint32_t expectDeps = (deps - 1) * sD + effectiveKd;
        uint32_t expectRows = (rows - 1) * sH + effectiveKh;
        uint32_t expectCols = (cols - 1) * sW + effectiveKw;
        int64_t depStart = expectDepStart >= tilingData_->fPad ? 0 : tilingData_->fPad - expectDepStart;
        int64_t rowStart = expectRowStart >= tilingData_->tPad ? 0 : tilingData_->tPad - expectRowStart;
        int64_t colStart = expectColStart >= tilingData_->lPad ? 0 : tilingData_->lPad - expectColStart;

        curOffsetInBatch_ = dIdx * outUbFactorD * tilingData_->hOutDim * tilingData_->wOutDim
                            + hIdx * outUbFactorH * tilingData_->wOutDim + wIdx * outUbFactorW;
        uint32_t oneRowElements = realCols * alignChannels;
        uint32_t onePlaneElements = oneRowElements * realRows;
        uint32_t oneBatchElements = onePlaneElements * realDeps;
        TensorDescInfo inputInfo = {
            {channels, realCols, realRows, realDeps, n},
            {1, alignChannels, oneRowElements, onePlaneElements, oneBatchElements},
            {1, channels, tilingData_->wInDim * channels, tilingData_->wInDim * tilingData_->hInDim * channels,
             tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim * channels}};
        CopyInMultiChannels(srcOffset, inputInfo);
        ComputeSingleChannels<M, U>(n, expectDeps, expectRows, expectCols, deps, rows, cols, depStart, rowStart, colStart, realDeps,
                                    realRows, realCols, alignChannels);
        CopyOutMultiChannels(dstOffset, n, deps, rows, cols, channels);
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::CopyInMultiChannels(int64_t offset,
                                                                                 TensorDescInfo& inputInfo)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    uint32_t dstStride = 0;
    DataCopyExtParams extParams;
    extParams.blockCount =inputInfo.size[1];
    extParams.blockLen = inputInfo.size[0] * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = dstStride;
    // 切w 跳h 跳d
   if (tilingData_->splitMode == SPLIT_COLS) {
        extParams.blockCount = inputInfo.size[1];
        LoopModeParams loopParams;
        loopParams.loop2Size = inputInfo.size[3];
        loopParams.loop2SrcStride = inputInfo.srcStride[3] * sizeof(T);
        loopParams.loop2DstStride = inputInfo.dstStride[3] * sizeof(T);
        loopParams.loop1Size = inputInfo.size[2];
        loopParams.loop1SrcStride = inputInfo.srcStride[2] * sizeof(T);
        loopParams.loop1DstStride = inputInfo.dstStride[2] * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else if (tilingData_->splitMode == SPLIT_ROWS) {  //切h 跳d
        extParams.blockCount = inputInfo.size[1] * inputInfo.size[2];
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = inputInfo.size[3];
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = inputInfo.srcStride[3] * sizeof(T);
        loopParams.loop1DstStride = inputInfo.dstStride[3] * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }  else { // 切d n 不跳
        extParams.blockCount = inputInfo.size[1] * inputInfo.size[2] * inputInfo.size[3] * inputInfo.size[4];
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::CopyOutMultiChannels(int64_t offset, int64_t n,
                                                                                  int64_t deps, int64_t rows,
                                                                                  int64_t cols, int64_t channels)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.DeQue<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = n * deps * rows * cols;
    extParams.blockLen = channels * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad<T>(maxGm_[offset], maxOutLocal, extParams);
    maxUBOutput_.FreeTensor<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename M, typename U>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::CopyPadData(
    LocalTensor<T>& inLocal, LocalTensor<T>& outLocal, int64_t n, int64_t inDeps, int64_t inRows, int64_t cols,
    int64_t expectDepStart, int64_t expectRowStart, int64_t expectColStart, int64_t realDeps, int64_t realRows,
    int64_t realCols, int64_t alignChannels)
{
    uint32_t inDim0Size = static_cast<uint32_t>(alignChannels * realCols);
    uint32_t inDim1Size = static_cast<uint32_t>(realRows);
    uint32_t inDim2Size = static_cast<uint32_t>(realDeps);
    uint32_t inDim3Size = static_cast<uint32_t>(n);
    uint32_t dim2Stride = inDim0Size * inDim1Size;
    uint32_t dim3Stride = dim2Stride * inDim2Size;
    uint32_t dstDim1Stride = static_cast<uint32_t>(alignChannels * cols);
    uint32_t dstDim2Stride = static_cast<uint32_t>(inRows * dstDim1Stride);
    uint32_t dstDim3Stride = static_cast<uint32_t>(inDeps * dstDim2Stride);
    uint32_t expectDim0Start = static_cast<uint32_t>(alignChannels * expectColStart);
    uint32_t expectDim1Start = static_cast<uint32_t>(expectRowStart);
    uint32_t expectDim2Start = static_cast<uint32_t>(expectDepStart);
    uint32_t totalSize = static_cast<uint32_t>(n * dstDim3Stride);
    CopyPad::CopyPadShapeInfo shapeInfo = {
        {inDim0Size, inDim1Size, inDim2Size, inDim3Size, 1},
        {1, inDim0Size, dim2Stride, dim3Stride, 1},
        {1, dstDim1Stride, dstDim2Stride, dstDim3Stride, 1},
        {expectDim0Start, expectDim1Start, expectDim2Start, 0, 0},
        totalSize,
    };
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    if constexpr(OP_TYPE == Pool3D::OP_TYPE_MAX_POOL_3D) {
        CopyPad::CopyAndPad<M, U, CopyPad::DEFAULT_MODE_NEG_INF>(inLocal, outLocal, indexLocal, shapeInfo, CopyPad::COPY_SINGLE_ROW);
    } else {
        CopyPad::CopyAndPad<M, U, CopyPad::DEFAULT_MODE_ZERO>(inLocal, outLocal, indexLocal, shapeInfo, CopyPad::COPY_SINGLE_ROW);
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename M, typename U>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::ComputeSingleChannels(
    int64_t n, int64_t inDeps, int64_t inRows, int64_t inCols, int64_t outDeps, int64_t outRows, int64_t outCols,
    int64_t expectDepStart, int64_t expectRowStart, int64_t expectColStart, int64_t realDeps, int64_t realRows,
    int64_t realCols, int64_t alignChannels)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t kD = tilingData_->kD;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint16_t sD = tilingData_->sD;
    uint32_t outUbFactorD = outDeps;
    uint32_t outUbFactorH = outRows;
    uint32_t outUbFactorW = outCols;
    uint32_t loopN = n;
    uint32_t loopH = outRows;
    uint32_t loopW = outCols;
    uint32_t loopD = outDeps;
    uint32_t dalitionW = tilingData_->dW;
    uint32_t dalitionH = tilingData_->dH;
    uint32_t dalitionD = tilingData_->dD;
    U batchStride = static_cast<U>(inDeps * inRows * inCols * alignChannels);
    U colStride = static_cast<U>(alignChannels * dalitionW);
    uint32_t rowStride = static_cast<uint32_t>(inCols * alignChannels * dalitionH);
    uint32_t depStride = static_cast<uint32_t>(inRows * inCols * alignChannels * dalitionD);
    // dalition 如何添加
    U oneLoopStrideD = static_cast<U>(sD * inRows * inCols * alignChannels);
    U oneLoopStrideH = static_cast<U>(sH * inCols * alignChannels);
    U oneLoopStrideW = static_cast<U>(sW * alignChannels);
    U oneChannelOutElements = static_cast<U>(outUbFactorD * outUbFactorH * outUbFactorW * alignChannels);
    U outLoopStrideH = static_cast<U>(outCols * alignChannels);
    U outLoopStrideD = static_cast<U>(outRows * outCols * alignChannels);
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(U);
    uint16_t cLoop = alignChannels / repeatElm;
    uint16_t tailNum = alignChannels - cLoop * repeatElm;
    CopyPadData<M, U>(inLocal, xLocal, n, inDeps, inRows, inCols, expectDepStart, expectRowStart, expectColStart, realDeps,
                      realRows, realCols, alignChannels);
    uint32_t totalNum = n * outUbFactorD * outUbFactorH * outUbFactorW;
    ComputeCurrentDivisor(totalNum);
    uint32_t curIndex = 0;
    uint32_t index = 0;
    float32_t divisor = tilingData_->divisor;
    LocalTensor<float> divisorLocal;
    if constexpr (OUT_DIV) {
        divisorLocal = divisorBuf_.Get<float>();
    }
    for (uint16_t i = 0; i < loopN; i++) {
        for (uint16_t d = 0; d < loopD; d++) {
            for (uint16_t j = 0; j < loopH; j++) {
                for (uint16_t k = 0; k < loopW; k++) {
                    auto srcAddr =
                        xLocalAddr + i * batchStride + d * oneLoopStrideD + j * oneLoopStrideH + k * oneLoopStrideW;
                    auto dstAddr = dstLocalAddr + i * oneChannelOutElements + d * outLoopStrideD + j * outLoopStrideH +
                                   k * alignChannels;
                    if constexpr (OUT_DIV) {
                            curIndex = GetCurrentDivisorIndex(index++);
                    }
                    __VEC_SCOPE__
                    {
                        if constexpr (OUT_DIV) {
                            divisor = divisorLocal(curIndex);
                        }
                        for (uint16_t m = 0; m < cLoop; m++) {
                            auto curSrcAddr = srcAddr + m * repeatElm;
                            auto curDstAddr = dstAddr + m * repeatElm;
                            if constexpr (OP_TYPE == Pool3D::OP_TYPE_MAX_POOL_3D) {
                                MaxPoolSingleChannel<M, U>(curDstAddr, curSrcAddr, kD, kH, kW, depStride, rowStride,
                                                           colStride, repeatElm);
                            } else {
                                AvgPoolSingleChannelImpl<M, U>(curDstAddr, curSrcAddr, kD, kH, kW, depStride, rowStride,
                                                           colStride, repeatElm, divisor);
                            }
                        }
                        auto curSrcAddr = srcAddr + cLoop * repeatElm;
                        auto curDstAddr = dstAddr + cLoop * repeatElm;
                        if constexpr (OP_TYPE == Pool3D::OP_TYPE_MAX_POOL_3D) {
                            MaxPoolSingleChannel<M, U>(curDstAddr, curSrcAddr, kD, kH, kW, depStride, rowStride, colStride,
                                                       tailNum);
                        } else {
                            AvgPoolSingleChannelImpl<M, U>(curDstAddr, curSrcAddr, kD, kH, kW, depStride, rowStride, colStride,
                                                       tailNum, divisor);
                        }
                    }
                }
            }
        }
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::ComputeDivisor(int64_t start, int64_t num)   {

    Pool3D::CalcDivisorParam param = {
        tilingData_->kD, tilingData_->kH, tilingData_->kW,
        tilingData_->sD, tilingData_->sH, tilingData_->sW,
        tilingData_->fPad, tilingData_->backPad, tilingData_->tPad,
        tilingData_->bottomPad, tilingData_->lPad, tilingData_->rPad,
        tilingData_->dOutDim, tilingData_->hOutDim, tilingData_->wOutDim,
        tilingData_->dInDim, tilingData_->hInDim, tilingData_->wInDim
    };
    LocalTensor<float> divisorLocal = divisorBuf_.Get<float>();
    auto dstAddr = (__local_mem__ float*)divisorLocal.GetPhyAddr();
    // 0b000  -> (int32/int64, includepad/no_include, need_clac_multi_batch/no_need)
    ComputeDivisorCommon(tilingData_->divisorMode, dstAddr, param, start, num);
}


template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::ComputeCurrentDivisor(uint32_t num)  
{
    if constexpr (OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        if (tilingData_->splitMode != SPLIT_BATCHS && tilingData_->realCalcDivisor != 0) {
            ComputeDivisor(curOffsetInBatch_, num);
        }
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline uint32_t Pool3dNDHWCBigChannelPad<T, OP_TYPE, OUT_DIV>::GetCurrentDivisorIndex(uint32_t index)  
{
    uint32_t curIndex = index;
    if (tilingData_->splitMode == SPLIT_BATCHS) {
        uint32_t batchElement = tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
        curIndex = index % batchElement;
    } else if (tilingData_->realCalcDivisor == 0) {
        curIndex = index + curOffsetInBatch_;
    } else {
        return curIndex;
    }
    return curIndex;
}

}  // namespace Pool3D
#endif  // MAX_POOL_V3_NDHWC_BIG_CHANNEL_PAD_H_
