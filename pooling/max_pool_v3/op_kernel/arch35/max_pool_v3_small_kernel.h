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
 * \file max_pool_v3_small_kernel.h
 * \brief
 */
#ifndef MAX_POOL_V3_SMALL_KERNEL_H_
#define MAX_POOL_V3_SMALL_KERNEL_H_

#include "max_pool_v3_common.h"

namespace MaxPoolV3 {
using namespace AscendC;

struct ComputeParam {
    int64_t n = 0;
    int64_t inRows = 0;
    int64_t inCols = 0;
    int64_t outRows = 0;
    int64_t outCols = 0;
    int64_t sH = 0;
    int64_t sW = 0;
    uint32_t batchElementsIn = 0;
};

constexpr int32_t SPARSE_THRESHOLD = 64;

template <typename T>
class MaxPoolV3SmallKernel {
public:
    __aicore__ inline MaxPoolV3SmallKernel(TPipe* pipe, const MaxPoolV3SmallKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <typename M, typename U, int32_t GATHER_MODE>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void CopyInMultiRowsSparse(
        int64_t offset, const ComputeParam& param, int32_t inCols, int64_t colsInUb, int64_t channelStride);
    __aicore__ inline void CopyInMultiRows(
        int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen, uint32_t colsInUb);
    __aicore__ inline void CopyMaxOut(int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen);
    template <typename M, typename U>
    __aicore__ inline void ComputeMultiRow(const ComputeParam& param);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleRow(const ComputeParam& param);
    template <typename M, typename U>
    __aicore__ inline void ComputeMultiBatch(const ComputeParam& param);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleKernel(const ComputeParam& param);
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void GenGatherIndex(
        uint32_t hFactorOut, uint32_t wFactorOut, uint32_t batchElements, uint32_t wIn, uint32_t hStride,
        uint32_t wStride, LocalTensor<U>& indexLocal);
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
    const MaxPoolV3SmallKernelTilingData* tilingData_;
    int64_t inHW_ = 1;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;
};

template <typename T>
__aicore__ inline void MaxPoolV3SmallKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, tilingData_->indiceUbSize);
}

template <typename T>
__aicore__ inline void MaxPoolV3SmallKernel<T>::Process()
{
    using indiceType = typename IndexTypeGet<T>::type;
    using computType = typename GetComputeType<T>::type;

    if (tilingData_->gatherMode == GATHER_SINGLE_ROW) {
        BaseCompute<computType, indiceType, GATHER_SINGLE_ROW>();
    } else if (tilingData_->gatherMode == GATHER_MULTI_ROW) {
        BaseCompute<computType, indiceType, GATHER_MULTI_ROW>();
    } else if (tilingData_->gatherMode == GATHER_MULTI_BATCH) {
        BaseCompute<computType, indiceType, GATHER_MULTI_BATCH>();
    } else {
        BaseCompute<computType, indiceType, GATHER_SINGLE_KERNEL>();
    }
}

template <typename T>
template <typename M, typename U, int32_t GATHER_MODE>
__aicore__ inline void MaxPoolV3SmallKernel<T>::BaseCompute()
{
    int64_t sW = tilingData_->sW;
    int64_t sH = tilingData_->sH;
    int64_t kW = tilingData_->kW;
    int64_t kH = tilingData_->kH;
    uint32_t outUbFactorW = tilingData_->outUbFactorW;
    uint32_t outUbFactorH = tilingData_->outUbFactorH;
    bool isSparse = (outUbFactorH != 1 && sH > kH && tilingData_->wInDim * sH * sizeof(T) > SPARSE_THRESHOLD) ||
                    (outUbFactorH == 1 && tilingData_->hInDim > kH &&
                     tilingData_->wInDim * tilingData_->hInDim * sizeof(T) > SPARSE_THRESHOLD);
    int64_t startIdx = 0;
    int64_t endIdx = 0;
    if (GetBlockIdx() < tilingData_->blockTail) {
        startIdx = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx = startIdx + tilingData_->blockFactor + 1;
    } else {
        startIdx = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx = startIdx + tilingData_->blockFactor;
    }
    uint32_t colsInUb = (outUbFactorW - 1) * sW + tilingData_->kW;
    if (tilingData_->splitMode != SPLIT_COLS) {
        colsInUb = tilingData_->wInDim;
    } else {
        colsInUb = Ops::Base::CeilAlign(colsInUb, static_cast<uint32_t>(Ops::Base::GetUbBlockSize() / sizeof(T)));
    }

    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    uint32_t batchElementsIn = tilingData_->hInDim * tilingData_->wInDim;
    if (isSparse) {
        uint32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
        batchElementsIn = Ops::Base::CeilAlign(static_cast<uint32_t>(outUbFactorH * kH * tilingData_->wInDim), elemNum);
        GenGatherIndex<U, GATHER_MODE>(outUbFactorH, outUbFactorW, batchElementsIn, colsInUb, kH, sW, indexLocal);
    } else {
        GenGatherIndex<U, GATHER_MODE>(outUbFactorH, outUbFactorW, batchElementsIn, colsInUb, sH, sW, indexLocal);
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

        int64_t srcOffset = nIdx * tilingData_->ubFactorN * tilingData_->hInDim * tilingData_->wInDim +
                            rowStart * tilingData_->wInDim + colStart;
        int64_t dstOffset = nIdx * tilingData_->ubFactorN * tilingData_->hOutDim * tilingData_->wOutDim +
                            hIdx * outUbFactorH * tilingData_->wOutDim + wIdx * outUbFactorW;
        int64_t inRows =
            tilingData_->splitMode == SPLIT_BATCHS ? tilingData_->hInDim : (rows - 1) * sH + tilingData_->kH;
        int64_t inCols = tilingData_->splitMode != SPLIT_COLS ? tilingData_->wInDim : (cols - 1) * sW + tilingData_->kW;
        ComputeParam param = {n, inRows, colsInUb, rows, cols, sH, sW, batchElementsIn};
        if (isSparse) {
            inRows = outUbFactorH * kH;
            param.inRows = outUbFactorH * kH;
            CopyInMultiRowsSparse(srcOffset, param, inCols, colsInUb, batchElementsIn);
            param.sH = kH;
        } else {
            CopyInMultiRows(srcOffset, n, inRows, inCols, colsInUb);
        }
        if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
            ComputeSingleRow<M, U>(param);
        } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
            ComputeMultiRow<M, U>(param);
        } else if constexpr (GATHER_MODE == GATHER_MULTI_BATCH) {
            ComputeMultiBatch<M, U>(param);
        } else {
            ComputeSingleKernel<M, U>(param);
        }
        CopyMaxOut(dstOffset, n, rows, cols);
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3SmallKernel<T>::CopyInMultiRowsSparse(
    int64_t offset, const ComputeParam& param, int32_t inCols, int64_t colsInUb, int64_t channelStride)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    DataCopyExtParams extParams;
    DataCopyPadExtParams<T> padExtParams;
    LoopModeParams loopParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    if (tilingData_->splitMode != SPLIT_COLS) {
        uint32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
        extParams.blockCount = param.outRows;
        extParams.blockLen = tilingData_->kH * tilingData_->wInDim * sizeof(T);
        extParams.srcStride = (tilingData_->sH - tilingData_->kH) * tilingData_->wInDim * sizeof(T);
        extParams.dstStride = 0;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = param.n;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->wInDim * tilingData_->hInDim * sizeof(T);
        loopParams.loop1DstStride = channelStride * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else {
        uint32_t dstStride = (colsInUb - inCols) * sizeof(T) / Ops::Base::GetUbBlockSize();
        extParams.blockCount = tilingData_->kH;
        extParams.blockLen = inCols * sizeof(T);
        extParams.srcStride = (tilingData_->wInDim - inCols) * sizeof(T);
        extParams.dstStride = dstStride;
        loopParams.loop2Size = param.n;
        loopParams.loop1Size = param.outRows;
        loopParams.loop2SrcStride = tilingData_->wInDim * tilingData_->hInDim * sizeof(T);
        loopParams.loop2DstStride = param.outRows * tilingData_->kH * colsInUb * sizeof(T);
        loopParams.loop1SrcStride = tilingData_->wInDim * tilingData_->sH * sizeof(T);
        loopParams.loop1DstStride = tilingData_->kH * colsInUb * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3SmallKernel<T>::CopyInMultiRows(
    int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen, uint32_t colsInUb)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    int64_t channelStride = blockCount * colsInUb;
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    if (tilingData_->splitMode != SPLIT_COLS) {
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = n * blockCount * blockLen * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    } else {
        uint32_t dstStride = (colsInUb - blockLen) * sizeof(T) / Ops::Base::GetUbBlockSize();
        DataCopyExtParams extParams;
        extParams.blockCount = blockCount;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = (tilingData_->wInDim - blockLen) * sizeof(T);
        extParams.dstStride = dstStride;
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3SmallKernel<T>::CopyMaxOut(
    int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.DeQue<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = (n * blockCount * blockLen) * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad<T>(maxGm_[offset], maxOutLocal, extParams);
    maxUBOutput_.FreeTensor<T>(maxOutLocal);
}

template <typename T>
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void MaxPoolV3SmallKernel<T>::GenGatherIndex(
    uint32_t hFactorOut, uint32_t wFactorOut, uint32_t batchElements, uint32_t wIn, uint32_t hStride, uint32_t wStride,
    LocalTensor<U>& indexLocal)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        GenGatherIndexSingleRow<U>(wStride, indexLocal);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        GenGatherIndexMultiRow<U>(wFactorOut, wIn, hStride, wStride, indexLocal);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_BATCH) {
        GenGatherIndexMultiBatch<U>(hFactorOut, wFactorOut, batchElements, wIn, hStride, wStride, indexLocal);
    } else {
        GenGatherIndexSingleKernel(wIn, tilingData_->kW, tilingData_->kH, indexLocal);
    }
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3SmallKernel<T>::ComputeMultiBatch(const ComputeParam& param)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint16_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = param.outCols;
    uint32_t outUbFactorH = param.outRows;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = param.sH;
    uint16_t sW = param.sW;
    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (outUbFactorH * outUbFactorW));
    uint16_t loopN = static_cast<uint16_t>(param.n / nFactor);
    uint16_t tailN = static_cast<uint16_t>(param.n - loopN * nFactor);
    if (tailN == 0) {
        loopN -= 1;
        tailN = nFactor;
    }
    U oneChannelElements = static_cast<U>(param.batchElementsIn);
    U oneLoopStride = static_cast<U>(nFactor * oneChannelElements);
    U oneLoopElements = static_cast<U>(nFactor * outUbFactorW * outUbFactorH);
    U tailLoopElements = static_cast<U>(tailN * outUbFactorW * outUbFactorH);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        MaxPoolSplitBatch<T, U>(
            dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, param.inCols, oneLoopStride, oneLoopElements,
            tailLoopElements);
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3SmallKernel<T>::ComputeMultiRow(const ComputeParam& param)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    uint32_t outUbFactorW = param.outCols;
    uint32_t outUbFactorH = param.outRows;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = param.sH;
    uint16_t sW = param.sW;
    uint16_t loopN = param.n;
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t hFactor = static_cast<uint16_t>(repeatElm / outUbFactorW);
    uint16_t loopH = static_cast<uint16_t>(outUbFactorH / hFactor);
    uint16_t tailH = static_cast<uint16_t>(outUbFactorH - loopH * hFactor);
    if (tailH == 0) {
        loopH -= 1;
        tailH = hFactor;
    }
    U oneChannelElements = static_cast<U>(param.batchElementsIn);
    U oneLoopStrideH = static_cast<U>(hFactor * sH * param.inCols);
    U oneLoopElements = static_cast<U>(hFactor * outUbFactorW);
    uint32_t num = static_cast<uint32_t>(hFactor * outUbFactorW);
    uint32_t tailNum = static_cast<uint32_t>(tailH * outUbFactorW);
    uint32_t tailLoopElements = static_cast<uint32_t>(tailH * outUbFactorW);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        MaxPoolSplitH<T, U>(
            dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, loopH, oneChannelElements, param.inCols, oneLoopStrideH,
            oneLoopElements, tailLoopElements);
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3SmallKernel<T>::ComputeSingleRow(const ComputeParam& param)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = param.outCols;
    uint32_t outUbFactorH = param.outRows;
    uint32_t ubFactorN = tilingData_->ubFactorN;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = param.sH;
    uint16_t sW = param.sW;
    uint16_t loopN = param.n;
    uint16_t loopH = outUbFactorH;
    uint16_t wFactor = repeatElm;
    uint16_t loopW = static_cast<uint16_t>(outUbFactorW / wFactor);
    uint16_t tailW = static_cast<uint16_t>(outUbFactorW - loopW * wFactor);
    if (tailW == 0) {
        loopW -= 1;
        tailW = wFactor;
    }
    U oneChannelElements = static_cast<U>(param.batchElementsIn);
    U oneLoopStrideH = static_cast<U>(sH * param.inCols);
    U oneLoopStrideW = static_cast<U>(sW * wFactor);
    U oneLoopElements = static_cast<U>(wFactor);
    U oneChannelOutElements = static_cast<U>(outUbFactorH * outUbFactorW);
    uint32_t num = wFactor;
    uint32_t tailNum = tailW;
    uint32_t tailLoopElements = tailW;
    if (ubFactorN == 1U) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> v0;
            MicroAPI::DataCopy(v0, indexAddr);
            MaxPoolSplitW<M, U>(
                dstLocalAddr, xLocalAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW, param.inCols, num,
                tailW);
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
                    dstAddr, srcAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW, param.inCols, num,
                    tailW);
            }
        }
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3SmallKernel<T>::ComputeSingleKernel(const ComputeParam& param)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = param.outCols;
    uint32_t outUbFactorH = param.outRows;
    uint32_t ubFactorN = param.n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = param.sH;
    uint16_t sW = param.sW;
    uint16_t loopN = static_cast<uint16_t>(ubFactorN);
    uint16_t loopH = static_cast<uint16_t>(outUbFactorH);
    uint16_t loopW = static_cast<uint16_t>(outUbFactorW);
    U oneChannelElements = static_cast<U>(param.batchElementsIn);
    U oneLoopStrideH = static_cast<U>(param.sH * param.inCols);
    U oneLoopStrideW = static_cast<U>(param.sW);

    uint16_t regNum = (kH * kW + repeatElm - 1) / repeatElm;
    U tailLoopElements = static_cast<U>(kH * kW - (regNum - 1) * repeatElm);

    switch (regNum) {
        case 1:
            MaxPoolSingleKernelCommon<T, U, ONE>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 2:
            MaxPoolSingleKernelCommon<T, U, TWO>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 3:
            MaxPoolSingleKernelCommon<T, U, THREE>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 4:
            MaxPoolSingleKernelCommon<T, U, FOUR>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 5:
            MaxPoolSingleKernelCommon<T, U, FIVE>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 6:
            MaxPoolSingleKernelCommon<T, U, SIX>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 7:
            MaxPoolSingleKernelCommon<T, U, SEVEN>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 8:
            MaxPoolSingleKernelCommon<T, U, EIGHT>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 9:
            MaxPoolSingleKernelCommon<T, U, NINE>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 10:
            MaxPoolSingleKernelCommon<T, U, TEN>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 11:
            MaxPoolSingleKernelCommon<T, U, ELEVEN>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 12:
            MaxPoolSingleKernelCommon<T, U, TWELVE>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 13:
            MaxPoolSingleKernelCommon<T, U, THIRTEEN>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 14:
            MaxPoolSingleKernelCommon<T, U, FOURTEEN>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 15:
            MaxPoolSingleKernelCommon<T, U, FIFTEEN>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
        case 16:
            MaxPoolSingleKernelCommon<T, U, SIXTEEN>(
                dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH,
                oneLoopStrideW, tailLoopElements);
            break;
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

} // namespace MaxPoolV3
#endif // MAX_POOL_V3_SMALL_KERNEL_H_
