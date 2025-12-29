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
 * \file max_pool_v3_nhwc_small_kernel.h
 * \brief
 */
#ifndef MAX_POOL_V3_NHWC_SMALL_KERNEL_H_
#define MAX_POOL_V3_NHWC_SMALL_KERNEL_H_

#include "max_pool_v3_common.h"

namespace MaxPoolV3 {
using namespace AscendC;

template <typename T>
class MaxPoolV3NHWCSmallKernel {
public:
    __aicore__ inline MaxPoolV3NHWCSmallKernel(TPipe* pipe, const MaxPoolV3NHWCSmallKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <typename M, typename U, int32_t GATHER_MODE>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void CopyInSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInMultiRows(
        int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen, uint32_t inColsElms, uint32_t channels);
    __aicore__ inline void CopyInMultiChannels(
        int64_t offset, int64_t n, int64_t rows, int64_t cols, int64_t channels, int64_t alignChannels, int64_t winDim);
    __aicore__ inline void CopyMaxOut(
        int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen, int64_t channels);
    __aicore__ inline void CopyOutMultiChannels(
        int64_t offset, int64_t n, int64_t rows, int64_t cols, int64_t channels);
    template <typename M, typename U>
    __aicore__ inline void ComputeMultiRow(
        int64_t n, int64_t inRows, int64_t inColsElms, int64_t outRows, int64_t outCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleRow(
        int64_t n, int64_t inRows, int64_t inColsElms, int64_t outRows, int64_t outCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeMultiBatch(
        int64_t n, int64_t inRows, int64_t inColsElms, int64_t outRows, int64_t outCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleChannels(
        int64_t inRows, int64_t cols, int64_t outRows, int64_t outCols, int64_t alignChannels);
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void GenGatherIndex(
        uint32_t hFactorOut, uint32_t wFactorOut, uint32_t hIn, uint32_t wInElms, uint32_t hStride, uint32_t wStride,
        uint32_t channels, LocalTensor<U>& indexLocal);
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }

    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // 输出ub
    TQue<QuePosition::VECOUT, BUFFER_NUM> maxUBOutput_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;
    const MaxPoolV3NHWCSmallKernelTilingData* tilingData_;
    int64_t inHW_ = 1;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;
};

template <typename T>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, INDEX_SIZE);
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::Process()
{
    using indiceType = typename IndexTypeGet<T>::type;
    using computType = typename GetComputeType<T>::type;

    if (tilingData_->gatherMode == GATHER_SINGLE_ROW) {
        BaseCompute<computType, indiceType, GATHER_SINGLE_ROW>();
    } else if (tilingData_->gatherMode == GATHER_MULTI_ROW) {
        BaseCompute<computType, indiceType, GATHER_MULTI_ROW>();
    } else if (tilingData_->gatherMode == NOT_GATHER) {
        BaseCompute<computType, indiceType, NOT_GATHER>();
    } else {
        BaseCompute<computType, indiceType, GATHER_MULTI_BATCH>();
    }
}

template <typename T>
template <typename M, typename U, int32_t GATHER_MODE>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::BaseCompute()
{
    int64_t sW = tilingData_->sW;
    int64_t sH = tilingData_->sH;
    uint32_t channels = tilingData_->channels;

    uint32_t outUbFactorW = tilingData_->outUbFactorW;
    uint32_t outUbFactorH = tilingData_->outUbFactorH;
    int64_t startIdx = 0;
    int64_t endIdx = 0;
    if (GetBlockIdx() < tilingData_->blockTail) {
        startIdx = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx = startIdx + tilingData_->blockFactor + 1;
    } else {
        startIdx = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx = startIdx + tilingData_->blockFactor;
    }
    uint32_t alignCols = (outUbFactorW - 1) * sW + tilingData_->kW;
    uint32_t alignColsElms = 0;
    if (tilingData_->splitMode != SPLIT_COLS) {
        alignCols = tilingData_->wInDim;
        alignColsElms = alignCols * channels;
    } else {
        alignColsElms =
            Ops::Base::CeilAlign(alignCols * channels, static_cast<uint32_t>(Ops::Base::GetUbBlockSize() / sizeof(T)));
    }
    uint32_t alignChannels =
        Ops::Base::CeilAlign(channels, static_cast<uint32_t>(Ops::Base::GetUbBlockSize() / sizeof(T)));
    if constexpr (GATHER_MODE != NOT_GATHER) {
        LocalTensor<U> indexLocal = indexBuf_.Get<U>();
        GenGatherIndex<U, GATHER_MODE>(
            outUbFactorH, outUbFactorW, tilingData_->hInDim, alignColsElms, sH, sW, channels, indexLocal);
    }

    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t nIdx = idx / (tilingData_->hLoop * tilingData_->wLoop);
        int64_t hIdx = (idx - nIdx * tilingData_->hLoop * tilingData_->wLoop) / tilingData_->wLoop;
        int64_t wIdx = idx % tilingData_->wLoop;
        int64_t n = nIdx == tilingData_->nLoop - 1 ? tilingData_->nOutDim - nIdx * tilingData_->ubFactorN :
                                                     tilingData_->ubFactorN;
        int64_t rows = hIdx == tilingData_->hLoop - 1 ? tilingData_->hOutDim - hIdx * outUbFactorH : outUbFactorH;
        int64_t cols = wIdx == tilingData_->wLoop - 1 ? tilingData_->wOutDim - wIdx * outUbFactorW : outUbFactorW;
        int64_t rowStart = hIdx * sH * outUbFactorH;
        int64_t colStart = wIdx * sW * outUbFactorW;

        int64_t srcOffset = nIdx * tilingData_->ubFactorN * tilingData_->hInDim * tilingData_->wInDim * channels +
                            rowStart * tilingData_->wInDim * channels + colStart * channels;
        int64_t dstOffset = nIdx * tilingData_->ubFactorN * tilingData_->hOutDim * tilingData_->wOutDim * channels +
                            hIdx * outUbFactorH * tilingData_->wOutDim * channels + wIdx * outUbFactorW * channels;
        int64_t inRows =
            tilingData_->splitMode == SPLIT_BATCHS ? tilingData_->hInDim : (rows - 1) * sH + tilingData_->kH;
        int64_t inCols = tilingData_->splitMode != SPLIT_COLS ? tilingData_->wInDim : (cols - 1) * sW + tilingData_->kW;

        if constexpr (GATHER_MODE == NOT_GATHER) {
            CopyInMultiChannels(srcOffset, n, inRows, inCols, channels, alignChannels, tilingData_->wInDim);
        } else {
            CopyInMultiRows(srcOffset, n, inRows, inCols, alignColsElms, channels);
        }
        if constexpr (GATHER_MODE == NOT_GATHER) {
            ComputeSingleChannels<M, U>(inRows, inCols, rows, cols, alignChannels);
        } else if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
            ComputeSingleRow<M, U>(n, inRows, alignColsElms, rows, cols);
        } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
            ComputeMultiRow<M, U>(n, inRows, alignColsElms, rows, cols);
        } else {
            ComputeMultiBatch<M, U>(n, inRows, alignColsElms, rows, cols);
        }
        if constexpr (GATHER_MODE == NOT_GATHER) {
            CopyOutMultiChannels(dstOffset, n, rows, cols, channels);
        } else {
            CopyMaxOut(dstOffset, n, rows, cols, channels);
        }
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::CopyInSingleRow(int64_t offset, int64_t blockLen)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();

    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[offset], extParams, padExtParams);
    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::CopyInMultiRows(
    int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen, uint32_t inColsElms, uint32_t channels)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    if (tilingData_->splitMode != SPLIT_COLS) {
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = n * blockCount * blockLen * channels * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    } else {
        uint32_t dstStride = (inColsElms - blockLen * channels) * sizeof(T) / Ops::Base::GetUbBlockSize();
        DataCopyExtParams extParams;
        extParams.blockCount = blockCount;
        extParams.blockLen = blockLen * channels * sizeof(T);
        extParams.srcStride = (tilingData_->wInDim - blockLen) * channels * sizeof(T);
        extParams.dstStride = dstStride;
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::CopyInMultiChannels(
    int64_t offset, int64_t n, int64_t rows, int64_t cols, int64_t channels, int64_t alignChannels, int64_t winDim)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    uint32_t dstStride = (alignChannels - channels) * sizeof(T) / Ops::Base::GetUbBlockSize();
    DataCopyExtParams extParams;
    extParams.blockCount = n * rows * cols;
    extParams.blockLen = channels * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = dstStride;
    // w不切，地址连续
    if (tilingData_->splitMode != SPLIT_COLS) {
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    } else {
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = rows;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = winDim * channels * sizeof(T);
        loopParams.loop1DstStride = cols * alignChannels * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        extParams.blockCount = cols;
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::CopyMaxOut(
    int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen, int64_t channels)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.DeQue<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = (n * blockCount * blockLen * channels) * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad<T>(maxGm_[offset], maxOutLocal, extParams);
    maxUBOutput_.FreeTensor<T>(maxOutLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::CopyOutMultiChannels(
    int64_t offset, int64_t n, int64_t rows, int64_t cols, int64_t channels)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.DeQue<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = n * rows * cols;
    extParams.blockLen = channels * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad<T>(maxGm_[offset], maxOutLocal, extParams);
    maxUBOutput_.FreeTensor<T>(maxOutLocal);
}

template <typename T>
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::GenGatherIndex(
    uint32_t hFactorOut, uint32_t wFactorOut, uint32_t hIn, uint32_t wInElms, uint32_t hStride, uint32_t wStride,
    uint32_t channels, LocalTensor<U>& indexLocal)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        NHWCGenGatherIndexSingleRow<U>(wStride, channels, indexLocal);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        NHWCGenGatherIndexMultiRow<U>(wFactorOut, wInElms, hStride, wStride, channels, indexLocal);
    } else {
        NHWCGenGatherIndexMultiBatch<U>(hFactorOut, wFactorOut, hIn, wInElms, hStride, wStride, channels, indexLocal);
    }
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::ComputeMultiBatch(
    int64_t n, int64_t inRows, int64_t inColsElms, int64_t outRows, int64_t outCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint16_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint16_t channels = tilingData_->channels;
    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (outUbFactorH * outUbFactorW * channels));
    uint16_t loopN = static_cast<uint16_t>(n / nFactor);
    uint16_t tailN = static_cast<uint16_t>(n - loopN * nFactor);

    U oneChannelElements = static_cast<U>(inRows * inColsElms);
    U oneLoopStride = static_cast<U>(nFactor * inRows * inColsElms);
    U oneLoopElements = static_cast<U>(nFactor * outUbFactorW * outUbFactorH * channels);
    U tailLoopElements = static_cast<U>(tailN * outUbFactorW * outUbFactorH * channels);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        MaxPoolSplitBatch<T, U>(
            dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, inColsElms, oneLoopStride, oneLoopElements, tailLoopElements,
            channels);
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::ComputeMultiRow(
    int64_t n, int64_t inRows, int64_t inColsElms, int64_t outRows, int64_t outCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint16_t channels = tilingData_->channels;
    uint16_t loopN = n;
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t hFactor = static_cast<uint16_t>(repeatElm / (outUbFactorW * channels));
    uint16_t loopH = static_cast<uint16_t>(outUbFactorH / hFactor);
    uint16_t tailH = static_cast<uint16_t>(outUbFactorH - loopH * hFactor);

    U oneChannelElements = static_cast<U>(inRows * inColsElms);
    U oneLoopStrideH = static_cast<U>(hFactor * sH * inColsElms);
    U oneLoopElements = static_cast<U>(hFactor * outUbFactorW * channels);
    uint32_t tailLoopElements = static_cast<uint32_t>(tailH * outUbFactorW * channels);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        MaxPoolSplitH<T, U>(
            dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, loopH, oneChannelElements, inColsElms, oneLoopStrideH,
            oneLoopElements, tailLoopElements, channels);
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::ComputeSingleRow(
    int64_t n, int64_t inRows, int64_t inColsElms, int64_t outRows, int64_t outCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = tilingData_->ubFactorN;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint16_t channels = tilingData_->channels;
    uint16_t loopN = ubFactorN;
    uint16_t loopH = outUbFactorH;
    uint16_t wFactor = repeatElm / tilingData_->channels;
    uint16_t loopW = static_cast<uint16_t>(outUbFactorW / wFactor);
    uint16_t tailW = static_cast<uint16_t>(outUbFactorW - loopW * wFactor);

    U oneChannelElements = static_cast<U>(inRows * inColsElms);
    U oneLoopStrideH = static_cast<U>(sH * inColsElms);
    U oneLoopStrideW = static_cast<U>(sW * wFactor * channels);
    U oneLoopElements = static_cast<U>(wFactor * channels);
    U oneChannelOutElements = static_cast<U>(outUbFactorH * outUbFactorW * channels);
    uint32_t tailLoopElements = tailW * tilingData_->channels;
    if (ubFactorN == 1U) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> v0;
            MicroAPI::DataCopy(v0, indexAddr);
            MaxPoolSplitW<M, U>(
                dstLocalAddr, xLocalAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW, inColsElms,
                oneLoopElements, tailLoopElements, channels);
        }
    } else {
        for (uint16_t i = 0; i < loopN; i++) {
            __local_mem__ M* srcAddr = xLocalAddr + i * oneChannelElements;
            __local_mem__ M* dstAddr = dstLocalAddr + i * oneChannelOutElements;
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<U> v0;
                MicroAPI::DataCopy(v0, indexAddr);
                MaxPoolSplitW<M, U>(
                    dstAddr, srcAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW, inColsElms,
                    oneLoopElements, tailLoopElements, channels);
            }
        }
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallKernel<T>::ComputeSingleChannels(
    int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, int64_t alignChannels)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint16_t loopN = tilingData_->ubFactorN;
    uint16_t loopH = outRows;
    uint16_t loopW = outCols;
    U batchStride = inRows * inCols * alignChannels;
    U colStride = inCols * alignChannels;
    U oneLoopStrideH = static_cast<U>(sH * inCols * alignChannels);
    U oneLoopStrideW = static_cast<U>(sW * alignChannels);
    U oneChannelOutElements = static_cast<U>(outUbFactorH * outUbFactorW * alignChannels);
    U outLoopStrideH = static_cast<U>(outCols * alignChannels);
    uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(M);
    if constexpr (sizeof(T) == B64) {
        repeatElm = 2 * Ops::Base::GetVRegSize() / sizeof(M);
    }
    uint16_t cLoop = alignChannels / repeatElm;
    uint16_t tailNum = alignChannels - cLoop * repeatElm;

    for (uint16_t i = 0; i < loopN; i++) {
        for (uint16_t j = 0; j < loopH; j++) {
            __VEC_SCOPE__
            {
                for (uint16_t k = 0; k < loopW; k++) {
                    for (uint16_t m = 0; m < cLoop; m++) {
                        auto curSrcAddr =
                            xLocalAddr + i * batchStride + j * oneLoopStrideH + k * oneLoopStrideW + m * repeatElm;
                        auto curDstAddr = dstLocalAddr + i * oneChannelOutElements + j * outLoopStrideH +
                                          k * alignChannels + m * repeatElm;
                        MaxPoolSingleChannel(curDstAddr, curSrcAddr, kH, kW, colStride, alignChannels, repeatElm);
                    }
                    auto curSrcAddr =
                        xLocalAddr + i * batchStride + j * oneLoopStrideH + k * oneLoopStrideW + cLoop * repeatElm;
                    auto curDstAddr = dstLocalAddr + i * oneChannelOutElements + j * outLoopStrideH +
                                      k * alignChannels + cLoop * repeatElm;
                    MaxPoolSingleChannel(curDstAddr, curSrcAddr, kH, kW, colStride, alignChannels, tailNum);
                }
            }
        }
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

} // namespace MaxPoolV3
#endif // MAX_POOL_V3_NHWC_SMALL_KERNEL_H_
