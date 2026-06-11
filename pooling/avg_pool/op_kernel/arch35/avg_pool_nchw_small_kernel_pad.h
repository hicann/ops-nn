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
 * \file avg_pool_nchw_small_kernel_pad.h
 * \brief
 */
#ifndef AVG_POOL_NCHW_SMALL_KERNEL_PAD_H_
#define AVG_POOL_NCHW_SMALL_KERNEL_PAD_H_

#include "avg_pool_common.h"
#include "op_kernel/platform_util.h" 
#include "../inc/kernel_utils.h"
#include "op_kernel/math_util.h"
#include "avg_pool_struct.h"

namespace AvgPool
{
using namespace AscendC;

template <typename T, bool OUT_DIV=false>
class AvgPoolNCHWSmallPadKernel
{
public:
    __aicore__ inline AvgPoolNCHWSmallPadKernel(TPipe* pipe, const AvgPoolNCHWSmallKernelTilingData* __restrict tiling)
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
    __aicore__ inline void ComputeMultiRow(int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols,
                                           int64_t expectRowStart, int64_t expectColStart, int64_t realRows,
                                           int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleRow(int64_t n, int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols,
                                            int64_t expectRowStart, int64_t expectColStart, int64_t realRows,
                                            int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeMultiBatch(int64_t n, int64_t inRows, int64_t inCols, int64_t outRows,
                                             int64_t outCols, int64_t expectRowStart, int64_t expectColStart,
                                             int64_t realRows, int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleKernel(int64_t n, int64_t inRows, int64_t inCols, int64_t outRows,
                                              int64_t outCols, int64_t expectRowStart, int64_t expectColStart,
                                              int64_t realRows, int64_t realCols);
    template <typename M, typename U>
    __aicore__ inline void CopyAndPad(LocalTensor<M>& inLocal, int64_t n, int64_t inRows, int64_t inCols,
                                      int64_t expectRowStart, int64_t expectColStart, int64_t realRows,
                                      int64_t realCols);
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void GenGatherIndex(uint32_t hFactorOut, uint32_t wFactorOut, uint32_t batchElements, uint32_t wIn,
                                          uint32_t hStride, uint32_t wStride, LocalTensor<U>& indexLocal);
    
    __aicore__ inline void InitDivisor();
    __aicore__ inline void ComputeDivisor(int64_t start, int64_t num);
    __aicore__ inline void DivComputeNCHWPad(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, uint32_t num);

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
    TBuf<QuePosition::VECCALC> divisorBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;

    const AvgPoolNCHWSmallKernelTilingData* tilingData_;
    bool needDivisorBuf_ = false;
    int64_t curOffsetInBatch_ = 0;
};

template <typename T, bool OUT_DIV>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);
    needDivisorBuf_ = tilingData_->divisor == 0;

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(tmpBuf_, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, tilingData_->indiceUbSize);
    pipe_->InitBuffer(scatterIndexBuf_, INDEX_SIZE);
    if constexpr (OUT_DIV) {
        pipe_->InitBuffer(divisorBuf_, tilingData_->divisorUbSize);
    }
}


template <typename T, bool OUT_DIV>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::ComputeDivisor(int64_t start, int64_t num)   {

    AvgPool::CalcDivisorParam param = {
        tilingData_->kH, tilingData_->kW,
        tilingData_->sH, tilingData_->sW,
        tilingData_->tPad, tilingData_->bottomPad,
        tilingData_->lPad, tilingData_->rPad,
        tilingData_->hOutDim, tilingData_->wOutDim,
        tilingData_->hInDim, tilingData_->wInDim
    };
    LocalTensor<float> divisorLocal = divisorBuf_.Get<float>();
    auto dstAddr = (__local_mem__ float*)divisorLocal.GetPhyAddr();
    // 0b000  -> (int32/int64, includepad/no_include, need_clac_multi_batch/no_need)
    ComputeDivisorCommon(tilingData_->divisorMode, dstAddr, param, start, num);
}

template <typename T, bool OUT_DIV>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::InitDivisor()
{
    if (OUT_DIV && tilingData_->realCalcDivisor == 0 && needDivisorBuf_) {
        
        int32_t oneVL = Ops::Base::GetVRegSize() / sizeof(float32_t);
        int32_t oneBatchNum = static_cast<int32_t>(tilingData_->hOutDim * tilingData_->wOutDim);
        ComputeDivisor(0, max(oneVL, oneBatchNum));
    }
}


template <typename T, bool OUT_DIV>
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::GenGatherIndex(uint32_t hFactorOut, uint32_t wFactorOut,
                                                                  uint32_t batchElements, uint32_t wIn, uint32_t hStride,
                                                                  uint32_t wStride, LocalTensor<U>& indexLocal)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        GenGatherIndexSingleRow<U>(wStride, indexLocal);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        GenGatherIndexMultiRow<U>(wFactorOut, wIn, hStride, wStride, indexLocal);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_BATCH)  {
        GenGatherIndexMultiBatch<U>(hFactorOut, wFactorOut, batchElements, wIn, hStride, wStride, indexLocal);
    } else {
        GenGatherIndexSingleKernel(wIn, tilingData_->kW, tilingData_->kH, indexLocal);
    }
}

template <typename T, bool OUT_DIV>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::CopyInMultiRowsCompact(int64_t offset, int64_t n, int64_t blockCount,
                                                                   int64_t blockLen)
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

template <typename T, bool OUT_DIV>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::CopyInMultiRows(int64_t offset, int64_t n, int64_t blockCount,
                                                                int64_t blockLen)
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

template <typename T, bool OUT_DIV>
template <typename M, typename U>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::CopyAndPad(LocalTensor<M>& inLocal, int64_t n, int64_t inRows,
                                                              int64_t inCols, int64_t expectRowStart,
                                                              int64_t expectColStart, int64_t realRows,
                                                              int64_t realCols)
{
    LocalTensor<U> indexLocal = scatterIndexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    __local_mem__ M* inLocalAddr = (__local_mem__ M*)inLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();

    uint32_t dupRepeatElm = Ops::Base::GetVRegSize() / sizeof(M);
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        dupRepeatElm = Ops::Base::GetVRegSize() /  sizeof(int32_t);
    }
    uint32_t ubFactorN = n;
    U oneChannelElements = static_cast<U>(inRows * inCols);

    uint32_t totalDupNum = ubFactorN * oneChannelElements;
    uint32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);

    uint16_t dupLoop = static_cast<uint16_t>((totalDupNum + dupRepeatElm - 1) / dupRepeatElm);
    if (tilingData_->copyMode == COPY_SINGLE_ROW) {
        constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
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
            CustomCopy(xLocalAddr, inLocalAddr, srcBatchStride, srcColStride, oneChannelElements, dstColStride, rowOffsetInUb,
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
            CustomCopyByScatterMultiRows<M, U>(xLocalAddr, inLocalAddr, v0, srcBatchStride, srcRowStride,
                                               dstBatchStride, dstRowStride, dstOffset, loopN, loopRows, repeatElm,
                                               tailRepeatElm);
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
            CustomCopyByScatterSingleRow<M, U>(xLocalAddr, inLocalAddr, srcBatchStride, srcColStride, oneChannelElements,
                                               dstColStride, rowOffsetInUb, colOffsetInUb, ubFactorN, hInUb, preColsLoop,
                                               totalCols, repeatElm);
        }
    }
}

template <typename T, bool OUT_DIV>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::DivComputeNCHWPad(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, uint32_t num)  
{
    if constexpr (OUT_DIV) {
        LocalTensor<float> divisorLocal = divisorBuf_.Get<float>();
        auto divAddr = (__local_mem__ float*)divisorLocal.GetPhyAddr();

        if (tilingData_->splitMode == SPLIT_BATCHS) {
            uint32_t batchElement = static_cast<uint32_t>(tilingData_->hOutDim * tilingData_->wOutDim);
            uint32_t oneVL = Ops::Base::GetVRegSize() / sizeof(float32_t);
            uint32_t batch = num / batchElement;
            AvgPoolDivMultiBatch<T, false>(dstAddr, srcAddr, divAddr, batch, batchElement);
        } else if (tilingData_->realCalcDivisor == 0) {
            auto curDivAddr = divAddr + curOffsetInBatch_;
            AvgPoolDivNorm<T, false>(dstAddr, srcAddr, curDivAddr, num);
        } else {
            ComputeDivisor(curOffsetInBatch_, num);
            AvgPoolDivNorm<T, false>(dstAddr, srcAddr, divAddr, num);
        }
    }
}

template <typename T, bool OUT_DIV>
template <typename M, typename U>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::ComputeSingleRow(int64_t n, int64_t inRows, int64_t inCols,
                                                                    int64_t outRows, int64_t outCols,
                                                                    int64_t expectRowStart, int64_t expectColStart,
                                                                    int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    
    __local_mem__ M* inLocalAddr = (__local_mem__ M*)inLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    using Z = typename std::conditional<sizeof(T)==B16 && OUT_DIV, float32_t, T>::type;
    __local_mem__ Z* dstLocalAddr = (__local_mem__ Z*)(maxOutLocal.template ReinterpretCast<Z>().GetPhyAddr());

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

    U oneChannelElements = static_cast<U>(inRows * inCols);
    U oneLoopStrideH = static_cast<U>(sH * inCols);
    U oneLoopStrideW = static_cast<U>(sW * wFactor);
    U oneLoopElements = static_cast<U>(wFactor);
    U oneChannelOutElements = static_cast<U>(outUbFactorH * outUbFactorW);
    U tailLoopElements = static_cast<U>(tailW);

    float32_t divisor = static_cast<float32_t>(tilingData_->divisor);
    constexpr U oneRegNumFp32 = Ops::Base::GetVRegSize() / sizeof(float32_t);
    U halfLoopOut0 = oneLoopElements > oneRegNumFp32 ? oneRegNumFp32 : oneLoopElements;
    U halfLoopOut1 = oneLoopElements > oneRegNumFp32 ? oneLoopElements - oneRegNumFp32 : 0;
    U tailHalfLoopOut0 = tailLoopElements > oneRegNumFp32 ? oneRegNumFp32 : tailLoopElements;
    U tailHalfLoopOut1 = tailLoopElements > oneRegNumFp32 ? tailLoopElements - oneRegNumFp32 : 0;

    CopyAndPad<M, U>(inLocal, n, inRows, inCols, expectRowStart, expectColStart, realRows, realCols);
    if (ubFactorN == 1U) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> v0;
            MicroAPI::DataCopy(v0, indexAddr);
            AvgPoolSplitW<T, U, Z, OUT_DIV>(
                dstLocalAddr, xLocalAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW, inCols,
                oneLoopElements, tailLoopElements, halfLoopOut0, halfLoopOut1, tailHalfLoopOut0, tailHalfLoopOut1,
                divisor);
        }
    } else {
        for (uint16_t i = 0; i < loopN; i++) {
            __local_mem__ T* srcAddr = xLocalAddr + i * oneChannelElements;
            __local_mem__ Z* dstAddr = dstLocalAddr + i * oneChannelOutElements;
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<U> v0;
                MicroAPI::DataCopy(v0, indexAddr);
                AvgPoolSplitW<T, U, Z, OUT_DIV>(
                    dstAddr, srcAddr, v0, kH, kW, loopH, loopW, oneLoopStrideH, oneLoopStrideW, inCols,
                    oneLoopElements, tailLoopElements, halfLoopOut0, halfLoopOut1, tailHalfLoopOut0, tailHalfLoopOut1,
                    divisor);
            }
        }
    }
    if constexpr (OUT_DIV) {
        __local_mem__ T* newDstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
        uint32_t totalOut = n * outUbFactorH * outUbFactorW;
        DivComputeNCHWPad(newDstAddr, dstLocalAddr, totalOut);
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T, bool OUT_DIV>
template <typename M, typename U>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::ComputeMultiRow(int64_t n, int64_t inRows, int64_t inCols,
                                                                   int64_t outRows, int64_t outCols,
                                                                   int64_t expectRowStart, int64_t expectColStart,
                                                                   int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    __local_mem__ M* inLocalAddr = (__local_mem__ M*)inLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    using Z = typename std::conditional<sizeof(T)==B16 && OUT_DIV, float32_t, T>::type;
    __local_mem__ Z* dstLocalAddr = (__local_mem__ Z*)(maxOutLocal.template ReinterpretCast<Z>().GetPhyAddr());

    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t sH = tilingData_->sH;

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
    float32_t divisor = static_cast<float32_t>(tilingData_->divisor);
    constexpr U oneRegNumFp32 = Ops::Base::GetVRegSize() / sizeof(float32_t);
    U halfLoopOut0 = oneLoopElements > oneRegNumFp32 ? oneRegNumFp32 : oneLoopElements;
    U halfLoopOut1 = oneLoopElements > oneRegNumFp32 ? oneLoopElements - oneRegNumFp32 : 0;
    U tailHalfLoopOut0 = tailLoopElements > oneRegNumFp32 ? oneRegNumFp32 : tailLoopElements;
    U tailHalfLoopOut1 = tailLoopElements > oneRegNumFp32 ? tailLoopElements - oneRegNumFp32 : 0;

    CopyAndPad<M, U>(inLocal, n, inRows, inCols, expectRowStart, expectColStart, realRows, realCols);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        AvgPoolSplitH<T, U, Z, OUT_DIV>(
            dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, loopH, oneChannelElements, inCols, oneLoopStrideH,
            oneLoopElements, tailLoopElements, halfLoopOut0, halfLoopOut1, tailHalfLoopOut0, tailHalfLoopOut1, divisor);
    }
    if constexpr (OUT_DIV) {
        __local_mem__ T* newDstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
        uint32_t totalOut = n * outUbFactorH * outUbFactorW;
        DivComputeNCHWPad(newDstAddr, dstLocalAddr, totalOut);
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T, bool OUT_DIV>
template <typename M, typename U>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::ComputeMultiBatch(int64_t n, int64_t inRows, int64_t inCols,
                                                                     int64_t outRows, int64_t outCols,
                                                                     int64_t expectRowStart, int64_t expectColStart,
                                                                     int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* inLocalAddr = (__local_mem__ M*)inLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    using Z = typename std::conditional<sizeof(T)==B16 && OUT_DIV, float32_t, T>::type;
    __local_mem__ Z* dstLocalAddr = (__local_mem__ Z*)(maxOutLocal.template ReinterpretCast<Z>().GetPhyAddr());

    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint32_t outUbFactorW = outCols;
    uint32_t outUbFactorH = outRows;
    uint32_t ubFactorN = n;

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;

    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (outUbFactorH * outUbFactorW));
    uint16_t loopN = static_cast<uint16_t>(n / nFactor);
    uint16_t tailN = static_cast<uint16_t>(n - loopN * nFactor);
    if (tailN == 0) {
        loopN -= 1;
        tailN = nFactor;
    }

    U oneLoopStride = static_cast<U>(nFactor * inRows * inCols);
    U oneLoopElements = static_cast<U>(nFactor * outUbFactorW * outUbFactorH);
    U tailLoopElements = static_cast<U>(tailN * outUbFactorW * outUbFactorH);

    float32_t divisor = static_cast<float32_t>(tilingData_->divisor);
    constexpr U oneRegNumFp32 = Ops::Base::GetVRegSize() / sizeof(float32_t);
    U halfLoopOut0 = oneLoopElements > oneRegNumFp32 ? oneRegNumFp32 : oneLoopElements;
    U halfLoopOut1 = oneLoopElements > oneRegNumFp32 ? oneLoopElements - oneRegNumFp32 : 0;
    U tailHalfLoopOut0 = tailLoopElements > oneRegNumFp32 ? oneRegNumFp32 : tailLoopElements;
    U tailHalfLoopOut1 = tailLoopElements > oneRegNumFp32 ? tailLoopElements - oneRegNumFp32 : 0;

    CopyAndPad<M, U>(inLocal, n, inRows, inCols, expectRowStart, expectColStart, realRows, realCols);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> v0;
        MicroAPI::DataCopy(v0, indexAddr);
        AvgPoolSplitBatch<T, U, Z, OUT_DIV>(
            dstLocalAddr, xLocalAddr, v0, kH, kW, loopN, inCols, oneLoopStride, oneLoopElements, tailLoopElements,
            halfLoopOut0, halfLoopOut1, tailHalfLoopOut0, tailHalfLoopOut1, divisor);
    }
    if constexpr (OUT_DIV) {
        __local_mem__ T* newDstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
        uint32_t totalOut = n * outUbFactorH * outUbFactorW;
        DivComputeNCHWPad(newDstAddr, dstLocalAddr, totalOut);
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T, bool OUT_DIV>
template <typename M, typename U>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::ComputeSingleKernel(int64_t n, int64_t inRows, int64_t inCols,
                                                                   int64_t outRows, int64_t outCols,
                                                                   int64_t expectRowStart, int64_t expectColStart,
                                                                   int64_t realRows, int64_t realCols)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> inLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    LocalTensor<M> xLocal = tmpBuf_.Get<M>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

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
    float32_t divisor = static_cast<float32_t>(tilingData_->divisor);

    uint16_t kernelSize = kH * kW;
    uint16_t regNum = (kernelSize + repeatElm - 1) / repeatElm;
    U tailLoopElements = static_cast<U>(kernelSize - (regNum - 1) * repeatElm);

    CopyAndPad<M, U>(inLocal, n, inRows, inCols, expectRowStart, expectColStart, realRows, realCols);
    switch (regNum) {
        case 1:
            AvgPoolSingleKernelCommon<T, U, ONE>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 2:
            AvgPoolSingleKernelCommon<T, U, TWO>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 3:
            AvgPoolSingleKernelCommon<T, U, THREE>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 4:
            AvgPoolSingleKernelCommon<T, U, FOUR>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 5:
            AvgPoolSingleKernelCommon<T, U, FIVE>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 6:
            AvgPoolSingleKernelCommon<T, U, SIX>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 7:
            AvgPoolSingleKernelCommon<T, U, SEVEN>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 8:
            AvgPoolSingleKernelCommon<T, U, EIGHT>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 9:
            AvgPoolSingleKernelCommon<T, U, NINE>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 10:
            AvgPoolSingleKernelCommon<T, U, TEN>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 11:
            AvgPoolSingleKernelCommon<T, U, ELEVEN>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 12:
            AvgPoolSingleKernelCommon<T, U, TWELVE>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        case 13:
            AvgPoolSingleKernelCommon<T, U, THIRTEEN>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, tailLoopElements, divisor);
            break;
        default:
            AvgPoolSingleKernelDefault<T, U>(dstLocalAddr, xLocalAddr, indexAddr, loopN, loopH, loopW, oneChannelElements, oneLoopStrideH, oneLoopStrideW, divisor, regNum, kernelSize);
            break;
    }
    inputQue_.FreeTensor<M>(inLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T, bool OUT_DIV>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::CopyMaxOut(int64_t offset, int64_t n, int64_t blockCount,
                                                              int64_t blockLen)
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

template <typename T, bool OUT_DIV>
template <typename M, typename U, int32_t GATHER_MODE>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::BaseCompute()
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
        int64_t n = nIdx == tilingData_->nLoop - 1 ? tilingData_->nOutDim - nIdx * tilingData_->ubFactorN
                                                   : tilingData_->ubFactorN;
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
        
        curOffsetInBatch_ = hIdx * outUbFactorH * tilingData_->wOutDim + wIdx * outUbFactorW;
        
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
        } else if constexpr (GATHER_MODE == GATHER_MULTI_BATCH)  {
            ComputeMultiBatch<M, U>(n, expectRows, alignCols, rows, cols, rowStart, colStart, realRows, realCols);
        } else {
            ComputeSingleKernel<M, U>(n, expectRows, alignCols, rows, cols, rowStart, colStart, realRows, realCols);
        }
        CopyMaxOut(dstOffset, n, rows, cols);
    }
}

template <typename T, bool OUT_DIV>
__aicore__ inline void AvgPoolNCHWSmallPadKernel<T, OUT_DIV>::Process()
{
    using indiceType = typename IndexTypeGet<T>::type;
    using computType = typename GetComputeType<T>::type;
    InitDivisor();
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

} // namespace AvgPool
#endif  // AVG_POOL_NCHW_SMALL_KERNEL_PAD_H_