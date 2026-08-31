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
 * \file max_pool_v3_nhwc_small_kernel_pad.h
 * \brief
 */
#ifndef MAX_POOL_V3_NHWC_SMALL_KERNEL_PAD_H_
#define MAX_POOL_V3_NHWC_SMALL_KERNEL_PAD_H_

#include "max_pool_v3_common.h"
#include "pool_utils/arch35/data_move/pool_2d_multi_channel_data_move.h"
#include "pool_utils/arch35/data_move/pool_2d_nhwc_small_kernel_data_move.h"
#include "pool_utils/arch35/index/pool_2d_nhwc_small_kernel_index.h"
#include "pool_utils/arch35/data_move/pool_2d_row_data_move.h"

namespace MaxPoolV3 {
using namespace AscendC;

template <typename T>
class MaxPoolV3NHWCSmallPadKernel {
public:
    __aicore__ inline MaxPoolV3NHWCSmallPadKernel(TPipe* pipe,
                                                  const MaxPoolV3NHWCSmallKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <typename M, typename U, int32_t GATHER_MODE>
    __aicore__ inline void BaseCompute();
    template <typename M, typename U>
    __aicore__ inline void ComputeMultiRow(int64_t n, int64_t inRows, int64_t inColsElms, int64_t outRows,
                                           int64_t outCols, int64_t expectRowStart, int64_t expectColStart,
                                           int64_t realRows, int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleRow(int64_t n, int64_t inRows, int64_t inColsElms, int64_t outRows,
                                            int64_t outCols, int64_t expectRowStart, int64_t expectColStart,
                                            int64_t realRows, int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeMultiBatch(int64_t n, int64_t inRows, int64_t inColsElms, int64_t outRows,
                                             int64_t outCols, int64_t expectRowStart, int64_t expectColStart,
                                             int64_t realRows, int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleChannels(int64_t n, int64_t inRows, int64_t inCols, int64_t outRows,
                                                 int64_t outCols, int64_t expectRowStart, int64_t expectColStart,
                                                 int64_t realRows, int64_t realCols, int64_t alignChannels);
    template <typename M, typename U>
    __aicore__ inline void CopyAndPad(LocalTensor<M>& inLocal, int64_t n, int64_t inRows, int64_t inColsElms,
                                      int64_t expectRowStart, int64_t expectColStart, int64_t realRows,
                                      int64_t realCols, int64_t channels);
    template <typename M, typename U>
    __aicore__ inline void MultiChannelCopyAndPad(LocalTensor<M>& inLocal, int64_t n, int64_t inRows, int64_t inCols,
                                                  int64_t expectRowStart, int64_t expectColStart, int64_t realRows,
                                                  int64_t realCols, int64_t channels);
    __aicore__ inline int64_t min(int64_t a, int64_t b) { return (a > b) ? b : a; }
    __aicore__ inline int64_t max(int64_t a, int64_t b) { return (a < b) ? b : a; }
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
    const MaxPoolV3NHWCSmallKernelTilingData* tilingData_;
};

template <typename T>
__aicore__ inline void MaxPoolV3NHWCSmallPadKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(tmpBuf_, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, INDEX_SIZE);
    pipe_->InitBuffer(scatterIndexBuf_, INDEX_SIZE);
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCSmallPadKernel<T>::Process()
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
__aicore__ inline void MaxPoolV3NHWCSmallPadKernel<T>::BaseCompute()
{
    int64_t sW = tilingData_->sW;
    int64_t sH = tilingData_->sH;

    uint32_t outUbFactorW = tilingData_->outUbFactorW;
    uint32_t outUbFactorH = tilingData_->outUbFactorH;
    uint32_t channels = tilingData_->channels;
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
    uint32_t alignColsElms = Ops::Base::CeilAlign(alignCols * channels,
                                                  static_cast<uint32_t>(Ops::Base::GetUbBlockSize() / sizeof(T)));
    uint32_t alignChannels = Ops::Base::CeilAlign(channels,
                                                  static_cast<uint32_t>(Ops::Base::GetUbBlockSize() / sizeof(T)));
    LocalTensor<U> scatterIndexLocal = scatterIndexBuf_.Get<U>();
    // maxRows only used for full load hin*win
    uint32_t maxRows = max((outUbFactorH - 1) * sH + tilingData_->kH, tilingData_->hInDim + tilingData_->tPad);
    if constexpr (GATHER_MODE != NOT_GATHER) {
        LocalTensor<U> indexLocal = indexBuf_.Get<U>();
        PoolUtils::Index::GenGatherIndex<U, GATHER_MODE>(outUbFactorH, outUbFactorW, maxRows, alignColsElms, sH, sW,
                                                         channels, indexLocal);
    }

    if (tilingData_->copyMode == SCATTER_MULTI_ROW) {
        PoolUtils::Index::NHWCGenScatterIndex<U, false>(tilingData_->wInDim, alignColsElms, channels,
                                                        scatterIndexLocal);
    } else if (tilingData_->copyMode == SCATTER_SINGLE_ROW) {
        PoolUtils::Index::NHWCGenScatterIndex<U, true>(tilingData_->wInDim, alignColsElms, channels, scatterIndexLocal);
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
        int64_t hUpper = min(hIdx * sH * outUbFactorH + (rows - 1) * sH + tilingData_->kH - tilingData_->tPad,
                             tilingData_->hInDim);
        int64_t hLower = max(hIdx * sH * outUbFactorH - tilingData_->tPad, (int64_t)0);
        int64_t wUpper = min(wIdx * sW * outUbFactorW + (cols - 1) * sW + tilingData_->kW - tilingData_->lPad,
                             tilingData_->wInDim);
        int64_t wLower = max(wIdx * sW * outUbFactorW - tilingData_->lPad, (int64_t)0);
        int64_t realRows = tilingData_->splitMode == SPLIT_BATCHS ? tilingData_->hInDim : hUpper - hLower;
        int64_t realCols = tilingData_->splitMode != SPLIT_COLS ? tilingData_->wInDim : wUpper - wLower;
        int64_t srcOffset = nIdx * tilingData_->ubFactorN * tilingData_->hInDim * tilingData_->wInDim * channels +
                            hLower * tilingData_->wInDim * channels + wLower * channels;
        int64_t dstOffset = nIdx * tilingData_->ubFactorN * tilingData_->hOutDim * tilingData_->wOutDim * channels +
                            hIdx * outUbFactorH * tilingData_->wOutDim * channels + wIdx * outUbFactorW * channels;
        int64_t expectRows = (rows - 1) * sH + tilingData_->kH;
        int64_t expectCols = (cols - 1) * sW + tilingData_->kW;
        int64_t rowStart = expectRowStart >= tilingData_->tPad ? 0 : tilingData_->tPad - expectRowStart;
        int64_t colStart = expectColStart >= tilingData_->lPad ? 0 : tilingData_->lPad - expectColStart;
        if (tilingData_->splitMode == SPLIT_BATCHS) {
            expectRows = max(expectRows, tilingData_->hInDim + tilingData_->tPad);
        }

        if constexpr (GATHER_MODE == NOT_GATHER) {
            PoolUtils::DataMove::CopyInMultiChannels(inputQue_, xGm_,
                                                     {srcOffset, n, realRows, realCols, channels, alignChannels,
                                                      tilingData_->wInDim, tilingData_->splitMode});
        } else {
            if (tilingData_->copyMode == SCATTER_MULTI_ROW) {
                PoolUtils::DataMove::CopyInSingleRow(inputQue_, xGm_, srcOffset, n * realRows * realCols * channels);
            } else {
                PoolUtils::DataMove::Pad::CopyInMultiRows(inputQue_, xGm_, srcOffset, n, realRows, realCols,
                                                          alignColsElms, channels, tilingData_->wInDim,
                                                          tilingData_->hInDim);
            }
        }

        if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
            ComputeSingleRow<M, U>(n, expectRows, alignColsElms, rows, cols, rowStart, colStart, realRows, realCols);
        } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
            ComputeMultiRow<M, U>(n, expectRows, alignColsElms, rows, cols, rowStart, colStart, realRows, realCols);
        } else if constexpr (GATHER_MODE == NOT_GATHER) {
            ComputeSingleChannels<M, U>(n, expectRows, alignCols, rows, cols, rowStart, colStart, realRows, realCols,
                                        alignChannels);
        } else {
            ComputeMultiBatch<M, U>(n, expectRows, alignColsElms, rows, cols, rowStart, colStart, realRows, realCols);
        }

        if constexpr (GATHER_MODE == NOT_GATHER) {
            PoolUtils::DataMove::CopyOutMultiChannels(maxUBOutput_, maxGm_, {dstOffset, n, rows, cols, channels});
        } else {
            PoolUtils::DataMove::CopyMaxOut(maxUBOutput_, maxGm_, dstOffset, n, rows, cols, channels);
        }
    }
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallPadKernel<T>::CopyAndPad(LocalTensor<M>& inLocal, int64_t n, int64_t inRows,
                                                                  int64_t inColsElms, int64_t expectRowStart,
                                                                  int64_t expectColStart, int64_t realRows,
                                                                  int64_t realCols, int64_t channels)
{
    LocalTensor<U> indexLocal = scatterIndexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    M negInf = GetNegInf<M>();
    __ubuf__ M* inLocalAddr = (__ubuf__ M*)inLocal.GetPhyAddr();
    __ubuf__ M* xLocalAddr = (__ubuf__ M*)xLocal.GetPhyAddr();

    uint32_t dupRepeatElm = Ops::Base::GetVRegSize() / sizeof(M);
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        dupRepeatElm = Ops::Base::GetVRegSize() / sizeof(int32_t);
    }
    uint32_t ubFactorN = n;
    U oneBatchElements = static_cast<U>(inRows * inColsElms);

    uint32_t totalDupNum = ubFactorN * oneBatchElements;
    uint16_t dupLoop = static_cast<uint16_t>((totalDupNum + dupRepeatElm - 1) / dupRepeatElm);
    if (tilingData_->copyMode == COPY_SINGLE_ROW) {
        constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
        uint16_t realColsElm = realCols * channels;
        uint16_t preColsLoop = static_cast<uint16_t>(realColsElm / repeatElm);
        uint32_t tailPreCols = realColsElm - preColsLoop * repeatElm;
        uint32_t colStride = inColsElms;
        uint32_t srcBatchStride = realRows * inColsElms;
        uint32_t rowOffsetInUb = expectRowStart;
        uint32_t colOffsetInUb = expectColStart * channels;
        uint32_t hInUb = realRows;
        __VEC_SCOPE__
        {
            CustomDuplicate<M>(xLocalAddr, totalDupNum, dupLoop);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
            CustomCopy(xLocalAddr, inLocalAddr, srcBatchStride, colStride, oneBatchElements, colStride, rowOffsetInUb,
                       colOffsetInUb, ubFactorN, hInUb, preColsLoop, tailPreCols, repeatElm);
        }
    } else if (tilingData_->copyMode == SCATTER_MULTI_ROW) {
        uint32_t srcBatchStride = realRows * realCols * channels;
        uint32_t srcRowStride = tilingData_->onceCopyRow * realCols * channels;
        uint32_t dstBatchStride = inRows * inColsElms;
        uint32_t dstRowStride = tilingData_->onceCopyRow * inColsElms;
        uint32_t dstOffset = expectRowStart * inColsElms + expectColStart * channels;
        uint32_t loopN = n;
        uint32_t loopRows = realRows / tilingData_->onceCopyRow;
        uint32_t repeatElm = tilingData_->onceCopyRow * realCols * channels;
        uint32_t tailRepeatElm = (realRows - loopRows * tilingData_->onceCopyRow) * realCols * channels;
        __VEC_SCOPE__
        {
            Reg::RegTensor<U> v0;
            Reg::LoadAlign(v0, indexAddr);
            CustomDuplicate<M>(xLocalAddr, totalDupNum, dupLoop);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
            CustomCopyByScatterMultiRows<M, U>(xLocalAddr, inLocalAddr, v0, srcBatchStride, srcRowStride,
                                               dstBatchStride, dstRowStride, dstOffset, loopN, loopRows, repeatElm,
                                               tailRepeatElm);
        }
    } else {
        constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
        uint16_t realColsElm = realCols * channels;
        uint16_t preColsLoop = static_cast<uint16_t>((realColsElm + repeatElm - 1) / repeatElm);
        uint32_t totalCols = realCols;
        uint32_t colStride = inColsElms;
        uint32_t srcBatchStride = realRows * inColsElms;
        uint32_t rowOffsetInUb = expectRowStart;
        uint32_t colOffsetInUb = expectColStart * channels;
        uint32_t hInUb = realRows;
        __VEC_SCOPE__
        {
            CustomDuplicate<M>(xLocalAddr, totalDupNum, dupLoop);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
            CustomCopyByScatterSingleRow<M, U>(xLocalAddr, inLocalAddr, srcBatchStride, colStride, oneBatchElements,
                                               colStride, rowOffsetInUb, colOffsetInUb, ubFactorN, hInUb, preColsLoop,
                                               realColsElm, repeatElm);
        }
    }
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallPadKernel<T>::ComputeMultiBatch(int64_t n, int64_t inRows, int64_t inColsElms,
                                                                         int64_t outRows, int64_t outCols,
                                                                         int64_t expectRowStart, int64_t expectColStart,
                                                                         int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    M negInf = GetNegInf<M>();
    __ubuf__ M* inLocalAddr = (__ubuf__ M*)inLocal.GetPhyAddr();
    __ubuf__ M* xLocalAddr = (__ubuf__ M*)xLocal.GetPhyAddr();
    __ubuf__ M* dstLocalAddr = (__ubuf__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t channels = tilingData_->channels;

    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (outUbFactorH * outUbFactorW * channels));
    uint16_t loopN = static_cast<uint16_t>(n / nFactor);
    uint16_t tailN = static_cast<uint16_t>(n - loopN * nFactor);

    U oneLoopStride = static_cast<U>(nFactor * inRows * inColsElms);
    U oneLoopElements = static_cast<U>(nFactor * outUbFactorW * outUbFactorH * channels);

    U tailLoopElements = static_cast<U>(tailN * outUbFactorW * outUbFactorH * channels);

    CopyAndPad<M, U>(inLocal, n, inRows, inColsElms, expectRowStart, expectColStart, realRows, realCols, channels);
    __VEC_SCOPE__
    {
        Reg::RegTensor<U> v0;
        Reg::LoadAlign(v0, indexAddr);
        MaxPoolSplitBatch<T, U>(dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, inColsElms, oneLoopStride, oneLoopElements,
                                tailLoopElements, channels);
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallPadKernel<T>::ComputeMultiRow(int64_t n, int64_t inRows, int64_t inColsElms,
                                                                       int64_t outRows, int64_t outCols,
                                                                       int64_t expectRowStart, int64_t expectColStart,
                                                                       int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    M negInf = GetNegInf<M>();
    __ubuf__ M* inLocalAddr = (__ubuf__ M*)inLocal.GetPhyAddr();
    __ubuf__ M* xLocalAddr = (__ubuf__ M*)xLocal.GetPhyAddr();
    __ubuf__ M* dstLocalAddr = (__ubuf__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t channels = tilingData_->channels;
    uint16_t loopN = ubFactorN;
    uint16_t hFactor = static_cast<uint16_t>(repeatElm / (outUbFactorW * channels));
    uint16_t loopH = static_cast<uint16_t>(outUbFactorH / hFactor);
    uint16_t tailH = static_cast<uint16_t>(outUbFactorH - loopH * hFactor);

    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(M);
    U oneBatchElements = static_cast<U>(inRows * inColsElms);
    U oneLoopStrideH = static_cast<U>(hFactor * sH * inColsElms);
    U oneLoopElements = static_cast<U>(hFactor * outUbFactorW * channels);
    uint32_t tailLoopElements = static_cast<uint32_t>(tailH * outUbFactorW * channels);

    CopyAndPad<M, U>(inLocal, n, inRows, inColsElms, expectRowStart, expectColStart, realRows, realCols, channels);
    __VEC_SCOPE__
    {
        Reg::RegTensor<U> v0;
        Reg::LoadAlign(v0, indexAddr);
        MaxPoolSplitH<T, U>(dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, loopH, oneBatchElements, inColsElms,
                            oneLoopStrideH, oneLoopElements, tailLoopElements, channels);
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallPadKernel<T>::ComputeSingleRow(int64_t n, int64_t inRows, int64_t inColsElms,
                                                                        int64_t outRows, int64_t outCols,
                                                                        int64_t expectRowStart, int64_t expectColStart,
                                                                        int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    M negInf = GetNegInf<M>();
    __ubuf__ M* inLocalAddr = (__ubuf__ M*)inLocal.GetPhyAddr();
    __ubuf__ M* xLocalAddr = (__ubuf__ M*)xLocal.GetPhyAddr();
    __ubuf__ M* dstLocalAddr = (__ubuf__ M*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint16_t channels = tilingData_->channels;
    uint16_t loopN = static_cast<uint16_t>(ubFactorN);
    uint16_t loopH = static_cast<uint16_t>(outUbFactorH);
    uint16_t wFactor = static_cast<uint16_t>(repeatElm / channels);
    uint16_t loopW = static_cast<uint16_t>(outUbFactorW / wFactor);
    uint16_t tailW = static_cast<uint16_t>(outUbFactorW - loopW * wFactor);

    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(M);
    U oneBatchElements = static_cast<U>(inRows * inColsElms);
    U oneLoopStrideH = static_cast<U>(sH * inColsElms);
    U oneLoopStrideW = static_cast<U>(sW * wFactor * channels);
    U oneLoopElements = static_cast<U>(wFactor * channels);
    U oneChannelOutElements = static_cast<U>(outUbFactorH * outUbFactorW * channels);
    U tailLoopElements = static_cast<U>(tailW * channels);

    CopyAndPad<M, U>(inLocal, n, inRows, inColsElms, expectRowStart, expectColStart, realRows, realCols, channels);
    if (ubFactorN == 1U) {
        __VEC_SCOPE__
        {
            Reg::RegTensor<U> v0;
            Reg::LoadAlign(v0, indexAddr);
            MaxPoolSplitW<M, U>(dstLocalAddr, xLocalAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW,
                                inColsElms, oneLoopElements, tailLoopElements, channels);
        }
    } else {
        for (uint16_t i = 0; i < loopN; i++) {
            __ubuf__ M* srcAddr = xLocalAddr + i * oneBatchElements;
            __ubuf__ M* dstAddr = dstLocalAddr + i * oneChannelOutElements;
            __VEC_SCOPE__
            {
                Reg::RegTensor<U> v0;
                Reg::LoadAlign(v0, indexAddr);
                MaxPoolSplitW<M, U>(dstAddr, srcAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW,
                                    inColsElms, oneLoopElements, tailLoopElements, channels);
            }
        }
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallPadKernel<T>::ComputeSingleChannels(int64_t n, int64_t inRows, int64_t inCols,
                                                                             int64_t outRows, int64_t outCols,
                                                                             int64_t expectRowStart,
                                                                             int64_t expectColStart, int64_t realRows,
                                                                             int64_t realCols, int64_t alignChannels)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    __ubuf__ M* inLocalAddr = (__ubuf__ M*)inLocal.GetPhyAddr();
    __ubuf__ M* xLocalAddr = (__ubuf__ M*)xLocal.GetPhyAddr();
    __ubuf__ M* dstLocalAddr = (__ubuf__ M*)maxOutLocal.GetPhyAddr();

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint16_t loopN = n;
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

    MultiChannelCopyAndPad<M, U>(inLocal, n, inRows, inCols, expectRowStart, expectColStart, realRows, realCols,
                                 alignChannels);

    for (uint16_t i = 0; i < loopN; i++) {
        for (uint16_t j = 0; j < loopH; j++) {
            __VEC_SCOPE__
            {
                for (uint16_t k = 0; k < loopW; k++) {
                    for (uint16_t m = 0; m < cLoop; m++) {
                        auto curSrcAddr = xLocalAddr + i * batchStride + j * oneLoopStrideH + k * oneLoopStrideW +
                                          m * repeatElm;
                        auto curDstAddr = dstLocalAddr + i * oneChannelOutElements + j * outLoopStrideH +
                                          k * alignChannels + m * repeatElm;
                        MaxPoolSingleChannel(curDstAddr, curSrcAddr, kH, kW, colStride, alignChannels, repeatElm);
                    }
                    auto curSrcAddr = xLocalAddr + i * batchStride + j * oneLoopStrideH + k * oneLoopStrideW +
                                      cLoop * repeatElm;
                    auto curDstAddr = dstLocalAddr + i * oneChannelOutElements + j * outLoopStrideH +
                                      k * alignChannels + cLoop * repeatElm;
                    MaxPoolSingleChannel(curDstAddr, curSrcAddr, kH, kW, colStride, alignChannels, tailNum);
                }
            }
        }
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T>
template <typename M, typename U>
__aicore__ inline void MaxPoolV3NHWCSmallPadKernel<T>::MultiChannelCopyAndPad(LocalTensor<M>& inLocal, int64_t n,
                                                                              int64_t inRows, int64_t inCols,
                                                                              int64_t expectRowStart,
                                                                              int64_t expectColStart, int64_t realRows,
                                                                              int64_t realCols, int64_t alignChannels)
{
    LocalTensor<U> indexLocal = scatterIndexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    M negInf = GetNegInf<M>();
    __ubuf__ M* inLocalAddr = (__ubuf__ M*)inLocal.GetPhyAddr();
    __ubuf__ M* xLocalAddr = (__ubuf__ M*)xLocal.GetPhyAddr();

    uint32_t dupRepeatElm = Ops::Base::GetVRegSize() / sizeof(M);
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        dupRepeatElm = Ops::Base::GetVRegSize() / sizeof(int32_t);
    }
    uint32_t ubFactorN = n;
    U oneBatchElements = static_cast<U>(inRows * inCols * alignChannels);

    uint32_t totalDupNum = ubFactorN * oneBatchElements;
    uint16_t dupLoop = static_cast<uint16_t>((totalDupNum + dupRepeatElm - 1) / dupRepeatElm);
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t realColsElm = realCols * alignChannels;
    uint16_t preColsLoop = static_cast<uint16_t>(realColsElm / repeatElm);
    uint32_t tailPreCols = realColsElm - preColsLoop * repeatElm;
    uint32_t colStride = realCols * alignChannels;
    uint32_t dstColStride = inCols * alignChannels;
    uint32_t srcBatchStride = realRows * realCols * alignChannels;
    uint32_t rowOffsetInUb = expectRowStart;
    uint32_t colOffsetInUb = expectColStart * alignChannels;
    uint32_t hInUb = realRows;
    __VEC_SCOPE__
    {
        CustomDuplicate<M>(xLocalAddr, totalDupNum, dupLoop);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
        CustomCopy(xLocalAddr, inLocalAddr, srcBatchStride, colStride, oneBatchElements, dstColStride, rowOffsetInUb,
                   colOffsetInUb, ubFactorN, hInUb, preColsLoop, tailPreCols, repeatElm);
    }
}

} // namespace MaxPoolV3
#endif // MAX_POOL_V3_NHWC_SMALL_KERNEL_PAD_H_
