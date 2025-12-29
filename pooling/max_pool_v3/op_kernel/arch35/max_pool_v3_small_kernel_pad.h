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
 * \file max_pool_v3_small_kernel_pad.h
 * \brief
 */
#ifndef MAX_POOL_V3_SMALL_KERNEL_PAD_H_
#define MAX_POOL_V3_SMALL_KERNEL_PAD_H_

#include "max_pool_v3_common.h"

namespace MaxPoolV3 {
using namespace AscendC;

template <typename T>
class MaxPoolV3SmallPadKernel {
public:
    __aicore__ inline MaxPoolV3SmallPadKernel(TPipe* pipe, const MaxPoolV3SmallKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <typename M, typename U, int32_t GATHER_MODE>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void CopyInMultiRowsCompact(int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen);
    __aicore__ inline void CopyInMultiRows(int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen);
    __aicore__ inline void CopyMaxOut(int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen);
    template <typename M, typename U>
    __aicore__ inline void ComputeMultiRow(
        int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, int64_t expectRowStart,
        int64_t expectColStart, int64_t realRows, int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleRow(
        int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, int64_t expectRowStart,
        int64_t expectColStart, int64_t realRows, int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeMultiBatch(
        int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, int64_t expectRowStart,
        int64_t expectColStart, int64_t realRows, int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleKernel(
        int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, int64_t expectRowStart,
        int64_t expectColStart, int64_t realRows, int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void CopyAndPad(
        LocalTensor<M>& inLocal, int64_t n, int64_t inRows, int64_t inCols, int64_t expectRowStart,
        int64_t expectColStart, int64_t realRows, int64_t realCols);
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void GenGatherIndex(
        uint32_t hFactorOut, uint32_t wFactorOut, uint32_t batchElements, uint32_t wIn, uint32_t hStride,
        uint32_t wStride, LocalTensor<U>& indexLocal);
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }
    __aicore__ inline int64_t max(int64_t a, int64_t b)
    {
        return (a < b) ? b : a;
    }
    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // 输出ub
    TQue<QuePosition::VECOUT, BUFFER_NUM> maxUBOutput_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    TBuf<QuePosition::VECCALC> scatterIndexBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;
    int32_t onceCopyRows_ = 1;
    const MaxPoolV3SmallKernelTilingData* tilingData_;
};

template <typename T>
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(tmpBuf_, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, tilingData_->indiceUbSize);
    pipe_->InitBuffer(scatterIndexBuf_, INDEX_SIZE);
}

template <typename T>
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::Process()
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
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::BaseCompute()
{
    int64_t sW = tilingData_->sW;
    int64_t sH = tilingData_->sH;

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
    if (tilingData_->splitMode != SPLIT_COLS) {
        alignCols = max(alignCols, static_cast<uint32_t>(tilingData_->wInDim + tilingData_->lPad));
    }
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<U> scatterIndexLocal = scatterIndexBuf_.Get<U>();
    // maxRows only used for full load hin*win
    uint32_t maxRows = (outUbFactorH - 1) * sH + tilingData_->kH;
    uint32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
    uint32_t batchElementsIn = static_cast<uint32_t>(maxRows * alignCols);
    GenGatherIndex<U, GATHER_MODE>(outUbFactorH, outUbFactorW, batchElementsIn, alignCols, sH, sW, indexLocal);

    if (tilingData_->copyMode == SCATTER_MULTI_ROW) {
        GenScatterIndex<U, false>(tilingData_->wInDim, alignCols, scatterIndexLocal);
    } else if (tilingData_->copyMode == SCATTER_SINGLE_ROW) {
        GenScatterIndex<U, true>(tilingData_->wInDim, alignCols, scatterIndexLocal);
    }
    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t nIdx = idx / (tilingData_->hLoop * tilingData_->wLoop);
        int64_t hIdx = (idx - nIdx * tilingData_->hLoop * tilingData_->wLoop) / tilingData_->wLoop;
        int64_t wIdx = idx % tilingData_->wLoop;
        int64_t n = nIdx == tilingData_->nLoop - 1 ? tilingData_->nOutDim - nIdx * tilingData_->ubFactorN :
                                                     tilingData_->ubFactorN;
        int64_t rows = hIdx == tilingData_->hLoop - 1 ? tilingData_->hOutDim - hIdx * outUbFactorH : outUbFactorH;
        int64_t cols = wIdx == tilingData_->wLoop - 1 ? tilingData_->wOutDim - wIdx * outUbFactorW : outUbFactorW;
        int64_t expectRowStart = hIdx * sH * outUbFactorH;
        int64_t expectColStart = wIdx * sW * outUbFactorW;
        int64_t hUpper =
            min(hIdx * sH * outUbFactorH + (rows - 1) * sH + tilingData_->kH - tilingData_->tPad, tilingData_->hInDim);
        int64_t hLower = max(hIdx * sH * outUbFactorH - tilingData_->tPad, (int64_t)0);
        int64_t wUpper =
            min(wIdx * sW * outUbFactorW + (cols - 1) * sW + tilingData_->kW - tilingData_->lPad, tilingData_->wInDim);
        int64_t wLower = max(wIdx * sW * outUbFactorW - tilingData_->lPad, (int64_t)0);
        int64_t realRows = hUpper - hLower;
        int64_t realCols = tilingData_->splitMode != SPLIT_COLS ? tilingData_->wInDim : wUpper - wLower;
        int64_t srcOffset = nIdx * tilingData_->ubFactorN * tilingData_->hInDim * tilingData_->wInDim +
                            hLower * tilingData_->wInDim + wLower;
        int64_t dstOffset = nIdx * tilingData_->ubFactorN * tilingData_->hOutDim * tilingData_->wOutDim +
                            hIdx * outUbFactorH * tilingData_->wOutDim + wIdx * outUbFactorW;
        int64_t expectRows = (rows - 1) * sH + tilingData_->kH;
        int64_t expectCols = (cols - 1) * sW + tilingData_->kW;
        int64_t rowStart = expectRowStart >= tilingData_->tPad ? 0 : tilingData_->tPad - expectRowStart;
        int64_t colStart = expectColStart >= tilingData_->lPad ? 0 : tilingData_->lPad - expectColStart;
        if (tilingData_->copyMode == SCATTER_MULTI_ROW) {
            CopyInMultiRowsCompact(srcOffset, n, realRows, realCols);
        } else {
            CopyInMultiRows(srcOffset, n, realRows, realCols);
        }
        if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
            ComputeSingleRow<M, U>(n, expectRows, alignCols, rows, cols, rowStart, colStart, realRows, realCols);
        } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
            ComputeMultiRow<M, U>(n, expectRows, alignCols, rows, cols, rowStart, colStart, realRows, realCols);
        } else if constexpr (GATHER_MODE == GATHER_MULTI_BATCH) {
            ComputeMultiBatch<M, U>(n, expectRows, alignCols, rows, cols, rowStart, colStart, realRows, realCols);
        } else {
            ComputeSingleKernel<M, U>(n, expectRows, alignCols, rows, cols, rowStart, colStart, realRows, realCols);
        }
        CopyMaxOut(dstOffset, n, rows, cols);
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::CopyInMultiRowsCompact(
    int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
    int64_t channelStride = Ops::Base::CeilAlign(static_cast<int32_t>(blockCount * blockLen), elemNum);
    DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = (tilingData_->wInDim - blockLen) * sizeof(T);
    extParams.dstStride = 0;

    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = n;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = tilingData_->wInDim * tilingData_->hInDim * sizeof(T);
    loopParams.loop1DstStride = channelStride * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::CopyInMultiRows(
    int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
    int64_t channelStride = blockCount * Ops::Base::CeilAlign(static_cast<int32_t>(blockLen), elemNum);
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    uint32_t dstStride = 0;
    DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = (tilingData_->wInDim - blockLen) * sizeof(T);
    extParams.dstStride = dstStride;
    if (n > 1) {
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = n;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->wInDim * tilingData_->hInDim * sizeof(T);
        loopParams.loop1DstStride = channelStride * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else {
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::CopyMaxOut(
    int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.DeQue<T>();
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
    int64_t channelStride = Ops::Base::CeilAlign(static_cast<int32_t>(blockCount * blockLen), elemNum);
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
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::GenGatherIndex(
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
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::CopyAndPad(
    LocalTensor<M>& inLocal, int64_t n, int64_t inRows, int64_t inCols, int64_t expectRowStart, int64_t expectColStart,
    int64_t realRows, int64_t realCols)
{
    LocalTensor<U> indexLocal = scatterIndexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    M negInf = GetNegInf<M>();
    __local_mem__ M* inLocalAddr = (__local_mem__ M*)inLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();

    uint32_t dupRepeatElm = Ops::Base::GetVRegSize() / sizeof(M);
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        dupRepeatElm = Ops::Base::GetVRegSize() / sizeof(int32_t);
    }
    uint32_t ubFactorN = n;
    U oneChannelElements = static_cast<U>(inRows * inCols);

    uint32_t totalDupNum = ubFactorN * oneChannelElements;
    uint32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
    uint16_t dupLoop = static_cast<uint16_t>((totalDupNum + dupRepeatElm - 1) / dupRepeatElm);
    if (tilingData_->copyMode == COPY_SINGLE_ROW) {
        constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
        uint16_t preColsLoop = realCols / repeatElm;
        uint32_t tailPreCols = realCols - preColsLoop * repeatElm;
        if (tailPreCols == 0) {
            preColsLoop -= 1;
            tailPreCols = repeatElm;
        }
        uint32_t srcColStride = Ops::Base::CeilAlign(static_cast<uint32_t>(realCols), elemNum);
        uint32_t dstColStride = inCols;
        uint32_t srcBatchStride = srcColStride * realRows;
        uint32_t rowOffsetInUb = expectRowStart;
        uint32_t colOffsetInUb = expectColStart;
        uint32_t hInUb = realRows;
        __VEC_SCOPE__
        {
            CustomDuplicate<M>(xLocalAddr, totalDupNum, dupLoop);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            CustomCopy(
                xLocalAddr, inLocalAddr, srcBatchStride, srcColStride, oneChannelElements, dstColStride, rowOffsetInUb,
                colOffsetInUb, ubFactorN, hInUb, preColsLoop, tailPreCols, repeatElm);
        }
    } else if (tilingData_->copyMode == SCATTER_MULTI_ROW) {
        uint32_t srcBatchStride = Ops::Base::CeilAlign(static_cast<uint32_t>(realRows * realCols), elemNum);
        uint32_t onceCopyRow = min(realRows, static_cast<uint32_t>(tilingData_->onceCopyRow));
        uint32_t srcRowStride = onceCopyRow * realCols;
        uint32_t dstBatchStride = inRows * inCols;
        uint32_t dstRowStride = onceCopyRow * inCols;
        uint32_t dstOffset = expectRowStart * inCols + expectColStart;
        uint32_t loopN = n;
        uint32_t loopRows = realRows / onceCopyRow;
        uint32_t repeatElm = onceCopyRow * realCols;
        uint32_t tailRepeatElm = (realRows - loopRows * onceCopyRow) * realCols;
        if (tailRepeatElm == 0) {
            loopRows -= 1;
            tailRepeatElm = onceCopyRow * realCols;
        }
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> v0;
            MicroAPI::DataCopy(v0, indexAddr);
            CustomDuplicate<M>(xLocalAddr, totalDupNum, dupLoop);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            CustomCopyByScatterMultiRows<M, U>(
                xLocalAddr, inLocalAddr, v0, srcBatchStride, srcRowStride, dstBatchStride, dstRowStride, dstOffset,
                loopN, loopRows, repeatElm, tailRepeatElm);
        }
    } else {
        constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
        uint16_t preColsLoop = (realCols + repeatElm - 1) / repeatElm;
        uint32_t totalCols = realCols;
        uint32_t srcColStride = Ops::Base::CeilAlign(static_cast<uint32_t>(realCols), elemNum);
        uint32_t dstColStride = inCols;
        uint32_t srcBatchStride = srcColStride * realRows;
        uint32_t rowOffsetInUb = expectRowStart;
        uint32_t colOffsetInUb = expectColStart;
        uint32_t hInUb = realRows;
        __VEC_SCOPE__
        {
            CustomDuplicate<M>(xLocalAddr, totalDupNum, dupLoop);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            CustomCopyByScatterSingleRow<M, U>(
                xLocalAddr, inLocalAddr, srcBatchStride, srcColStride, oneChannelElements, dstColStride, rowOffsetInUb,
                colOffsetInUb, ubFactorN, hInUb, preColsLoop, totalCols, repeatElm);
        }
    }
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::ComputeMultiBatch(
    int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, int64_t expectRowStart,
    int64_t expectColStart, int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    M negInf = GetNegInf<M>();
    __local_mem__ M* inLocalAddr = (__local_mem__ M*)inLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;

    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (outUbFactorH * outUbFactorW));
    uint16_t loopN = static_cast<uint16_t>(n / nFactor);
    uint16_t tailN = static_cast<uint16_t>(n - loopN * nFactor);
    if (tailN == 0) {
        loopN -= 1;
        tailN = nFactor;
    }
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(M);
    U oneChannelElements = static_cast<U>(inRows * inCols);
    U oneLoopStride = static_cast<U>(nFactor * inRows * inCols);
    U oneLoopElements = static_cast<U>(nFactor * outUbFactorW * outUbFactorH);

    U tailLoopElements = static_cast<U>(tailN * outUbFactorW * outUbFactorH);

    CopyAndPad<M, U>(inLocal, n, inRows, inCols, expectRowStart, expectColStart, realRows, realCols);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        MaxPoolSplitBatch<T, U>(
            dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, inCols, oneLoopStride, oneLoopElements, tailLoopElements);
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::ComputeMultiRow(
    int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, int64_t expectRowStart,
    int64_t expectColStart, int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    M negInf = GetNegInf<M>();
    __local_mem__ M* inLocalAddr = (__local_mem__ M*)inLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint16_t loopN = ubFactorN;
    uint16_t hFactor = static_cast<uint16_t>(repeatElm / outUbFactorW);
    uint16_t loopH = static_cast<uint16_t>(outUbFactorH / hFactor);
    uint16_t tailH = static_cast<uint16_t>(outUbFactorH - loopH * hFactor);
    if (tailH == 0) {
        loopH -= 1;
        tailH = hFactor;
    }
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(M);
    U oneChannelElements = static_cast<U>(inRows * inCols);
    U oneLoopStrideH = static_cast<U>(hFactor * sH * inCols);
    U oneLoopElements = static_cast<U>(hFactor * outUbFactorW);
    uint32_t tailLoopElements = static_cast<uint32_t>(tailH * outUbFactorW);

    CopyAndPad<M, U>(inLocal, n, inRows, inCols, expectRowStart, expectColStart, realRows, realCols);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        MaxPoolSplitH<T, U>(
            dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, loopH, oneChannelElements, inCols, oneLoopStrideH,
            oneLoopElements, tailLoopElements);
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::ComputeSingleRow(
    int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, int64_t expectRowStart,
    int64_t expectColStart, int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    M negInf = GetNegInf<M>();
    __local_mem__ M* inLocalAddr = (__local_mem__ M*)inLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint16_t loopN = static_cast<uint16_t>(ubFactorN);
    uint16_t loopH = static_cast<uint16_t>(outUbFactorH);
    uint16_t wFactor = static_cast<uint16_t>(repeatElm);
    uint16_t loopW = static_cast<uint16_t>(outUbFactorW / wFactor);
    uint16_t tailW = static_cast<uint16_t>(outUbFactorW - loopW * wFactor);
    if (tailW == 0) {
        loopW -= 1;
        tailW = wFactor;
    }
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(M);
    U oneChannelElements = static_cast<U>(inRows * inCols);
    U oneLoopStrideH = static_cast<U>(sH * inCols);
    U oneLoopStrideW = static_cast<U>(sW * wFactor);
    U oneLoopElements = static_cast<U>(wFactor);
    U oneChannelOutElements = static_cast<U>(outUbFactorH * outUbFactorW);
    uint32_t num = wFactor;

    CopyAndPad<M, U>(inLocal, n, inRows, inCols, expectRowStart, expectColStart, realRows, realCols);
    if (ubFactorN == 1U) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> v0;
            MicroAPI::DataCopy(v0, indexAddr);
            MaxPoolSplitW<M, U>(
                dstLocalAddr, xLocalAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW, inCols, num, tailW);
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
                    dstAddr, srcAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW, inCols, num, tailW);
            }
        }
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3SmallPadKernel<T>::ComputeSingleKernel(
    int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, int64_t expectRowStart,
    int64_t expectColStart, int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    M negInf = GetNegInf<M>();
    __local_mem__ M* inLocalAddr = (__local_mem__ M*)inLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint16_t loopN = static_cast<uint16_t>(ubFactorN);
    uint16_t loopH = static_cast<uint16_t>(outUbFactorH);
    uint16_t loopW = static_cast<uint16_t>(outUbFactorW);
    U oneChannelElements = static_cast<U>(inRows * inCols);
    U oneLoopStrideH = static_cast<U>(sH * inCols);
    U oneLoopStrideW = static_cast<U>(sW);

    uint16_t regNum = (kH * kW + repeatElm - 1) / repeatElm;
    U tailLoopElements = static_cast<U>(kH * kW - (regNum - 1) * repeatElm);

    CopyAndPad<M, U>(inLocal, n, inRows, inCols, expectRowStart, expectColStart, realRows, realCols);
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
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

} // namespace MaxPoolV3
#endif // MAX_POOL_V3_SMALL_KERNEL_PAD_H_
