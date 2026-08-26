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
 * \file pool_3d_big_kernel_ndhwc.h
 * \brief pool_3d big kernel for ndhwc
 */
#ifndef POOL_3D_NDHWC_BIG_KERNEL_H_
#define POOL_3D_NDHWC_BIG_KERNEL_H_

#include "pool_3d_common.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace Pool3D {
using namespace AscendC;

static constexpr int32_t NO_SPLIT_KERNEL = 0;
static constexpr int32_t SPLIT_KERNEL_D = 1;
static constexpr int32_t SPLIT_KERNEL_H = 2;
static constexpr int32_t SPLIT_KERNEL_W = 3;
static constexpr int32_t SPLIT_KERNEL_C = 4;
static constexpr int64_t GATHER_THRES = 32;
static constexpr int64_t MOV_ALIGN_THRES = 128;

template <typename T, int32_t OP_TYPE>
class Pool3DNDHWCBigKernel {
public:
    __aicore__ inline Pool3DNDHWCBigKernel(TPipe* pipe, const Pool3DBigKernelNDHWCTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcKernelSize(int64_t curIdx, int64_t& curkD, int64_t& curkH, int64_t& curkW,
                                          int64_t& curInOffset);
    template <int32_t SPLIT_MODE>
    __aicore__ inline void BaseCompute(int64_t beginIdx, int64_t endIdx);
    __aicore__ inline void CopyInSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInMultiRows(int64_t offset, int64_t dLen, int64_t hLen, int64_t wLen, int64_t blockLen);
    __aicore__ inline void CopyInMultiRowsContiguous(int64_t offset, int64_t dLen, int64_t hLen, int64_t wLen,
                                                     int64_t blockLen);
    __aicore__ inline void CopyMaxOut(int64_t curIdx);
    __aicore__ inline void CopyOutSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void NoSplitKernelProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW,
                                                int64_t curInOffset);
    __aicore__ inline void SplitKernelDProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW,
                                               int64_t curInOffset);
    __aicore__ inline void SplitKernelHProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW,
                                               int64_t curInOffset);
    __aicore__ inline void SplitKernelWProcess(int32_t localCurIdx, int64_t curkD, int64_t curkH, int64_t curkW,
                                               int64_t curInOffset);
    __aicore__ inline void SplitChannelProcess(int32_t curIdx, int64_t curkD, int64_t curkH, int64_t curkW,
                                               int64_t curInOffset);
    template <bool MERGE, bool IS_LAST_LOOP>
    __aicore__ inline void ComputeSingle(int32_t localCurIdx, int64_t loop, int64_t dataCount);
    template <bool MERGE, bool IS_LAST_LOOP>
    __aicore__ inline void ComputeSingleWithGather(int32_t localCurIdx, int64_t loop, int64_t dataCount);
    template <bool MERGE, bool IS_LAST_LOOP>
    __aicore__ inline void ComputeSingleNorm(int32_t localCurIdx, int64_t loop, int64_t dataCount);
    template <bool MERGE, bool IS_LAST_LOOP>
    __aicore__ inline void ComputeSingleNormForAvgNotFp32(int32_t localCurIdx, int64_t loop, int64_t dataCount);
    template <bool MERGE, bool IS_LAST_LOOP>
    __aicore__ inline void ComputeSingleWithGatherForAvgNotFp32(int32_t localCurIdx, int64_t loop, int64_t dataCount);
    template <bool CLEAR>
    __aicore__ inline void InitOutLocal(int32_t localCurIdx);
    __aicore__ inline void ComputeSum(LocalTensor<T>& xLocal, int64_t dataCount);
    __aicore__ inline void ComputeAvg(int64_t length);
    __aicore__ inline int64_t min(int64_t a, int64_t b) { return (a > b) ? b : a; }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<> sumBuf_;
    TBuf<> outputBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;

    const Pool3DBigKernelNDHWCTilingData* tilingData_;

    int64_t inDHW_ = 1;
    int64_t outDHW_ = 1;
    int64_t curOriginD_ = 0;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;
    int64_t channelAlign_ = 0;
    int64_t inStrideD_ = 0;
    int64_t inStrideH_ = 0;
    int64_t inStrideW_ = 0;
    int32_t maxOutLen_ = 0;
    float mulsFactor_ = 0.0f;
    static constexpr int64_t vRegLen_ = platform::GetVRegSize() / sizeof(T);
    static constexpr int64_t eleBlockSize_ = platform::GetUbBlockSize() / sizeof(T);
};

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::Init(GM_ADDR x, GM_ADDR y)
{
    constexpr int64_t byteWidth = sizeof(T);
    inDHW_ = tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim;
    outDHW_ = tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
    channelAlign_ = ops::CeilAlign(tilingData_->channel, eleBlockSize_);
    if (GetBlockIdx() < tilingData_->blockTail) {
        beginIdx_ = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx_ = beginIdx_ + tilingData_->blockFactor + 1;
    } else {
        beginIdx_ = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx_ = beginIdx_ + tilingData_->blockFactor;
    }
    maxOutLen_ = min(static_cast<int64_t>((endIdx_ - beginIdx_) * tilingData_->channel), tilingData_->outUbSize);
    inStrideW_ = (tilingData_->dW - 1) * tilingData_->channel * byteWidth;
    inStrideH_ = tilingData_->dH * tilingData_->wInDim * tilingData_->channel * byteWidth;
    inStrideD_ = tilingData_->dD * tilingData_->hInDim * tilingData_->wInDim * tilingData_->channel * byteWidth;
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * byteWidth);
    pipe_->InitBuffer(outputBuf_, tilingData_->outUbSize * byteWidth);
    if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D && !std::is_same<T, float>::value) {
        pipe_->InitBuffer(sumBuf_, tilingData_->outUbSize * byteWidth);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::Process()
{
    if (tilingData_->tilingMode == NO_SPLIT_KERNEL) {
        BaseCompute<NO_SPLIT_KERNEL>(beginIdx_, endIdx_);
    } else if (tilingData_->tilingMode == SPLIT_KERNEL_D) {
        BaseCompute<SPLIT_KERNEL_D>(beginIdx_, endIdx_);
    } else if (tilingData_->tilingMode == SPLIT_KERNEL_H) {
        BaseCompute<SPLIT_KERNEL_H>(beginIdx_, endIdx_);
    } else if (tilingData_->tilingMode == SPLIT_KERNEL_W) {
        BaseCompute<SPLIT_KERNEL_W>(beginIdx_, endIdx_);
    } else if (tilingData_->tilingMode == SPLIT_KERNEL_C) {
        BaseCompute<SPLIT_KERNEL_C>(beginIdx_, endIdx_);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::CalcKernelSize(int64_t curIdx, int64_t& curkD, int64_t& curkH,
                                                                        int64_t& curkW, int64_t& curInOffset)
{
    int64_t cur3D = curIdx % outDHW_;
    int64_t curN = curIdx / outDHW_;
    int64_t curDo = cur3D / (tilingData_->hOutDim * tilingData_->wOutDim);
    int64_t cur2D = cur3D % (tilingData_->hOutDim * tilingData_->wOutDim);
    int64_t curHo = cur2D / tilingData_->wOutDim;
    int64_t curWo = cur2D % tilingData_->wOutDim;

    int64_t curkPadD = 0;
    CalcKernelSizeCore(ParamsForDim{tilingData_->dInDim, curDo, tilingData_->kD, tilingData_->sD, tilingData_->dD,
                                    tilingData_->fPad, tilingData_->backPad},
                       curkD, curkPadD, curOriginD_);
    int64_t curkPadH = 0;
    CalcKernelSizeCore(ParamsForDim{tilingData_->hInDim, curHo, tilingData_->kH, tilingData_->sH, tilingData_->dH,
                                    tilingData_->tPad, tilingData_->bottomPad},
                       curkH, curkPadH, curOriginH_);
    int64_t curkPadW = 0;
    CalcKernelSizeCore(ParamsForDim{tilingData_->wInDim, curWo, tilingData_->kW, tilingData_->sW, tilingData_->dW,
                                    tilingData_->lPad, tilingData_->rPad},
                       curkW, curkPadW, curOriginW_);

    curOriginIndex_ = (curOriginD_ * tilingData_->hInDim * tilingData_->wInDim + curOriginH_ * tilingData_->wInDim +
                       curOriginW_) *
                      tilingData_->channel;
    curInOffset = curN * inDHW_ * tilingData_->channel + curOriginIndex_;

    if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        if (tilingData_->divisorOverride > 0) {
            mulsFactor_ = 1.0f / static_cast<float>(tilingData_->divisorOverride);
        } else if (tilingData_->countIncludePad == 0) {
            mulsFactor_ = curkD * curkH * curkW == 0 ? 0 : 1.0f / static_cast<float>(curkD * curkH * curkW);
        } else {
            mulsFactor_ = 1.0f / static_cast<float>(curkPadD * curkPadH * curkPadW);
        }
    }
}

template <typename T, int32_t OP_TYPE>
template <int32_t SPLIT_MODE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::BaseCompute(int64_t beginIdx, int64_t endIdx)
{
    int64_t curkD = 1;
    int64_t curkH = 1;
    int64_t curkW = 1;
    int64_t curInOffset = 0;
    // current blockdim range
    for (int64_t idx = beginIdx; idx < endIdx; idx++) {
        CalcKernelSize(idx, curkD, curkH, curkW, curInOffset); // compute kernel_size of cur_out, and offset of cur_in
        int32_t localCurIdx = (idx - beginIdx) % tilingData_->onceOutNum;
        if constexpr (SPLIT_MODE == NO_SPLIT_KERNEL) {
            InitOutLocal<true>(localCurIdx); // not need init
            NoSplitKernelProcess(localCurIdx, curkD, curkH, curkW, curInOffset);
            CopyMaxOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_D) {
            InitOutLocal<true>(localCurIdx); // need init when localCurIdx is 0
            SplitKernelDProcess(localCurIdx, curkD, curkH, curkW, curInOffset);
            CopyMaxOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_H) {
            InitOutLocal<true>(localCurIdx); // need init when localCurIdx is 0
            SplitKernelHProcess(localCurIdx, curkD, curkH, curkW, curInOffset);
            CopyMaxOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_W) {
            InitOutLocal<true>(localCurIdx); // need init when localCurIdx is 0
            SplitKernelWProcess(localCurIdx, curkD, curkH, curkW, curInOffset);
            CopyMaxOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_C) {
            SplitChannelProcess(idx, curkD, curkH, curkW, curInOffset);
        }
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::CopyInSingleRow(int64_t offset, int64_t blockLen)
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

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::CopyInMultiRows(int64_t offset, int64_t dLen, int64_t hLen,
                                                                         int64_t wLen, int64_t blockLen)
{
    if (tilingData_->channel * sizeof(T) < GATHER_THRES) {
        CopyInMultiRowsContiguous(offset, dLen, hLen, wLen, tilingData_->channel);
    } else {
        LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
        LoopModeParams loopParams;
        loopParams.loop1Size = hLen;
        loopParams.loop1SrcStride = inStrideH_;
        loopParams.loop1DstStride = wLen * channelAlign_ * sizeof(T);
        loopParams.loop2Size = dLen;
        loopParams.loop2SrcStride = inStrideD_;
        loopParams.loop2DstStride = hLen * wLen * channelAlign_ * sizeof(T);

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPadExtParams<T> padExtParams;
        padExtParams.isPad = false;
        padExtParams.leftPadding = 0;
        padExtParams.rightPadding = 0;
        padExtParams.paddingValue = 0;

        DataCopyExtParams extParams;
        extParams.blockCount = wLen;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = inStrideW_;
        extParams.dstStride = 0;
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        inputQue_.EnQue(xLocal);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::CopyInMultiRowsContiguous(int64_t offset, int64_t dLen,
                                                                                   int64_t hLen, int64_t wLen,
                                                                                   int64_t blockLen)
{
    bool isEnableMovAlign = ((wLen * blockLen) % eleBlockSize_) == 0 && (wLen * blockLen >= MOV_ALIGN_THRES);
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    if (isEnableMovAlign) {
        LoopModeParams loopParams;
        loopParams.loop1Size = hLen;
        loopParams.loop1SrcStride = inStrideH_;
        loopParams.loop1DstStride = wLen * blockLen * sizeof(T);
        loopParams.loop2Size = dLen;
        loopParams.loop2SrcStride = inStrideD_;
        loopParams.loop2DstStride = hLen * wLen * blockLen * sizeof(T);

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPadExtParams<T> padExtParams;
        padExtParams.isPad = false;
        padExtParams.leftPadding = 0;
        padExtParams.rightPadding = 0;
        padExtParams.paddingValue = 0;

        DataCopyExtParams extParams;
        extParams.blockCount = wLen;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = inStrideW_ * sizeof(T);
        extParams.dstStride = 0;
        DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else {
        NdDmaLoopInfo<FOUR> loopInfo;
        loopInfo.loopSize[0] = blockLen;
        loopInfo.loopSize[1] = wLen;
        loopInfo.loopSize[TWO] = hLen;
        loopInfo.loopSize[THREE] = dLen;
        loopInfo.loopSrcStride[0] = 1;
        loopInfo.loopSrcStride[1] = tilingData_->dW * tilingData_->channel;
        loopInfo.loopSrcStride[TWO] = tilingData_->dH * tilingData_->wInDim * tilingData_->channel;
        loopInfo.loopSrcStride[THREE] = tilingData_->dD * tilingData_->hInDim * tilingData_->wInDim *
                                        tilingData_->channel;
        loopInfo.loopDstStride[0] = 1;
        loopInfo.loopDstStride[1] = blockLen;
        loopInfo.loopDstStride[TWO] = wLen * blockLen;
        loopInfo.loopDstStride[THREE] = hLen * wLen * blockLen;
        NdDmaParams<T, FOUR> dmaParams = {loopInfo, 0};
        DataCopy(xLocal, xGm_[offset], dmaParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::CopyOutSingleRow(int64_t offset, int64_t blockLen)
{
    LocalTensor<T> maxOutLocal = outputBuf_.Get<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    DataCopyPad(maxGm_[offset], maxOutLocal, extParams);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::CopyMaxOut(int64_t curIdx)
{
    int32_t localCurIdx = (curIdx - beginIdx_) % tilingData_->onceOutNum;

    if (localCurIdx == tilingData_->onceOutNum - 1 || curIdx == endIdx_ - 1) {
        CopyOutSingleRow((curIdx - localCurIdx) * tilingData_->channel, (localCurIdx + 1) * tilingData_->channel);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::ComputeAvg(int64_t length)
{
    if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        // 求平均并转回原类型
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<T> sumLocal = outputBuf_.Get<T>();
            Muls(sumLocal, sumLocal, mulsFactor_, length);
        } else {
            LocalTensor<float> sumLocal = sumBuf_.Get<float>();
            LocalTensor<T> avgLocal = outputBuf_.Get<T>();
            Muls(sumLocal, sumLocal, mulsFactor_, length);
            Cast(avgLocal, sumLocal, RoundMode::CAST_ROUND, length);
        }
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::NoSplitKernelProcess(int32_t localCurIdx, int64_t curkD,
                                                                              int64_t curkH, int64_t curkW,
                                                                              int64_t curInOffset)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    CopyInMultiRows(curInOffset, curkD, curkH, curkW, tilingData_->channel);
    ComputeSingle<false, true>(localCurIdx, curkW * curkH * curkD, tilingData_->channel);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::SplitKernelDProcess(int32_t localCurIdx, int64_t curkD,
                                                                             int64_t curkH, int64_t curkW,
                                                                             int64_t curInOffset)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    // 整行搬入
    int64_t dFactor = tilingData_->inUbSize / channelAlign_ / curkW / curkH;
    int64_t dLoops = ops::Ceil(curkD, dFactor);
    int64_t dTail = curkD - (dLoops - 1) * dFactor;

    for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
        int32_t curDFactor = dLoop == dLoops - 1 ? dTail : dFactor;
        int64_t inputOffset = curInOffset + dLoop * dFactor * tilingData_->hInDim * tilingData_->wInDim *
                                                tilingData_->channel * tilingData_->dD;
        bool isLastLoop = dLoop == dLoops - 1;
        CopyInMultiRows(inputOffset, curDFactor, curkH, curkW, tilingData_->channel);
        if (!isLastLoop) {
            ComputeSingle<true, false>(localCurIdx, curkH * curkW * curDFactor, tilingData_->channel);
        } else {
            ComputeSingle<true, true>(localCurIdx, curkH * curkW * curDFactor, tilingData_->channel);
        }
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::SplitKernelHProcess(int32_t localCurIdx, int64_t curkD,
                                                                             int64_t curkH, int64_t curkW,
                                                                             int64_t curInOffset)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    // 整行搬入
    int64_t dLoops = curkD;
    int64_t hFactor = tilingData_->inUbSize / channelAlign_ / curkW;
    int64_t hLoops = ops::Ceil(curkH, hFactor);
    int64_t hTail = curkH - (hLoops - 1) * hFactor;

    for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
        int64_t inputOffset = curInOffset + dLoop * tilingData_->hInDim * tilingData_->wInDim * tilingData_->channel *
                                                tilingData_->dD;
        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
            bool isLastLoop = hLoop == hLoops - 1 && dLoop == dLoops - 1;
            CopyInMultiRows(inputOffset, 1, curhFactor, curkW, tilingData_->channel);
            if (!isLastLoop) {
                ComputeSingle<true, false>(localCurIdx, curkW * curhFactor, tilingData_->channel);
            } else {
                ComputeSingle<true, true>(localCurIdx, curkW * curhFactor, tilingData_->channel);
            }
            inputOffset += curhFactor * tilingData_->wInDim * tilingData_->channel * tilingData_->dH;
        }
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::SplitKernelWProcess(int32_t localCurIdx, int64_t curkD,
                                                                             int64_t curkH, int64_t curkW,
                                                                             int64_t curInOffset)
{
    if (curkD * curkH * curkW == 0) {
        return;
    }
    // 单行很大，单行循环搬
    int64_t dLoops = curkD;
    int64_t hLoops = curkH;
    int64_t wFactor = tilingData_->inUbSize / channelAlign_;
    int64_t wLoops = ops::Ceil(curkW, wFactor);
    int64_t wTail = curkW - (wLoops - 1) * wFactor;

    for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
        int64_t dOffset = curInOffset +
                          dLoop * tilingData_->hInDim * tilingData_->wInDim * tilingData_->channel * tilingData_->dD;
        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            int64_t hOffset = dOffset + hLoop * tilingData_->wInDim * tilingData_->channel * tilingData_->dH;
            for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
                bool isLastLoop = wLoop == wLoops - 1 && hLoop == hLoops - 1 && dLoop == dLoops - 1;
                int64_t inputOffset = hOffset + wLoop * wFactor * tilingData_->channel * tilingData_->dW;
                CopyInMultiRows(inputOffset, 1, 1, curFactor, tilingData_->channel);
                if (!isLastLoop) {
                    ComputeSingle<true, false>(localCurIdx, curFactor, tilingData_->channel);
                } else {
                    ComputeSingle<true, true>(localCurIdx, curFactor, tilingData_->channel);
                }
            }
        }
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::ComputeSum(LocalTensor<T>& xLocal, int64_t dataCount)
{
    LocalTensor<float> sumLocal = sumBuf_.Get<float>();
    __ubuf__ T* xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ float* sumLocalAddr = (__ubuf__ float*)sumLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(float);
    uint16_t repeatTimes = static_cast<uint16_t>(ops::Ceil(dataCount, static_cast<int64_t>(repeatElm)));
    uint32_t len = dataCount;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> in;
        Reg::RegTensor<float> inFp32;
        Reg::RegTensor<float> sum;
        Reg::MaskReg mask;
        uint32_t num = len;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = Reg::UpdateMask<float>(num);
            auto sumReg = Reg::CreateAddrReg<float>(i, static_cast<uint16_t>(repeatElm));
            auto srcReg = Reg::CreateAddrReg<T>(i, static_cast<uint16_t>(repeatElm));
            Reg::LoadAlign(in, xLocalAddr, srcReg);
            Reg::LoadAlign(sum, sumLocalAddr, sumReg);
            Reg::UnPack((Reg::RegTensor<uint32_t>&)in, (Reg::RegTensor<uint16_t>&)in);
            Reg::Cast<float, T, castTraitT2Fp32>(inFp32, in, mask);
            Reg::Add(sum, inFp32, sum, mask);
            Reg::StoreAlign(sumLocalAddr, sum, sumReg, mask);
        }
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::SplitChannelProcess(int32_t curIdx, int64_t curkD,
                                                                             int64_t curkH, int64_t curkW,
                                                                             int64_t curInOffset)
{
    if (curkD * curkH * curkW == 0) {
        InitOutLocal<true>(0);
        int64_t cFactor = tilingData_->inUbSize / vRegLen_ * vRegLen_;
        int64_t cLoops = ops::Ceil(tilingData_->channel, cFactor);
        int64_t cTail = tilingData_->channel - (cLoops - 1) * cFactor;
        for (int64_t cLoop = 0; cLoop < cLoops; cLoop++) {
            int32_t curFactor = cLoop == cLoops - 1 ? cTail : cFactor;
            CopyOutSingleRow(curIdx * tilingData_->channel + cLoop * cFactor, curFactor);
        }
        return;
    }
    // 单行很大，单行循环搬
    int64_t dLoops = curkD;
    int64_t hLoops = curkH;
    int64_t wLoops = curkW;
    int64_t cFactor = tilingData_->inUbSize / vRegLen_ * vRegLen_;
    int64_t cLoops = ops::Ceil(tilingData_->channel, cFactor);
    int64_t cTail = tilingData_->channel - (cLoops - 1) * cFactor;
    for (int64_t cLoop = 0; cLoop < cLoops; cLoop++) {
        int32_t curFactor = cLoop == cLoops - 1 ? cTail : cFactor;
        InitOutLocal<true>(0);
        for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
            int64_t dOffset = curInOffset +
                              tilingData_->dD * dLoop * tilingData_->hInDim * tilingData_->wInDim *
                                  tilingData_->channel +
                              cLoop * cFactor;
            for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
                int64_t hOffset = dOffset + tilingData_->dH * hLoop * tilingData_->wInDim * tilingData_->channel;
                for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                    int64_t inputOffset = hOffset + tilingData_->dW * wLoop * tilingData_->channel;
                    CopyInSingleRow(inputOffset, curFactor);
                    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
                    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
                        LocalTensor<T> maxOutLocal = outputBuf_.Get<T>();
                        Max(maxOutLocal, xLocal, maxOutLocal, curFactor);
                    } else if constexpr (std::is_same<T, float>::value) {
                        LocalTensor<T> sumLocal = outputBuf_.Get<T>();
                        Add(sumLocal, xLocal, sumLocal, curFactor);
                    } else {
                        ComputeSum(xLocal, curFactor);
                    }
                    inputQue_.FreeTensor<T>(xLocal);
                }
            }
        }
        ComputeAvg(curFactor);
        CopyOutSingleRow(curIdx * tilingData_->channel + cLoop * cFactor, curFactor);
    }
}

template <typename T, int32_t OP_TYPE>
template <bool CLEAR>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::InitOutLocal(int32_t localCurIdx)
{
    if (localCurIdx != 0) {
        return;
    }

    int32_t maxLocalLen = maxOutLen_;
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);

    if constexpr (!CLEAR) { // kerel 全载场景无需merge，因此无需初始化output
        return;
    }
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D || std::is_same<T, float>::value) {
        LocalTensor<T> maxOutLocal = outputBuf_.Get<T>();
        __ubuf__ T* dstAddr = (__ubuf__ T*)maxOutLocal.GetPhyAddr();
        constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T);
        uint16_t repeatTimes = CeilDivision(maxLocalLen, repeatElm);
        uint32_t num = maxLocalLen;
        __ubuf__ T* addr = (__ubuf__ T*)dstAddr;
        __VEC_SCOPE__ { CustomDuplicate<T, OP_TYPE>(addr, num, repeatTimes); }
    } else {
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        __ubuf__ float* dstAddr = (__ubuf__ float*)sumLocal.GetPhyAddr();
        constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(float);
        uint16_t repeatTimes = CeilDivision(maxLocalLen, repeatElm);
        uint32_t num = maxLocalLen;
        __ubuf__ float* addr = (__ubuf__ float*)dstAddr;
        __VEC_SCOPE__ { CustomDuplicate<float, OP_TYPE>(addr, num, repeatTimes); }
    }
}

template <typename T, int32_t OP_TYPE>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::ComputeSingleNorm(int32_t localCurIdx, int64_t loop,
                                                                           int64_t dataCount)
{
    LocalTensor<T> maxOutLocal = outputBuf_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    T negInf = GetNegInf<T>();
    __ubuf__ T* xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ T* dstLocalAddr = (__ubuf__ T*)maxOutLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(T);
    uint16_t repeatTimes = CeilDivision(dataCount, repeatElm);
    uint32_t num = dataCount;
    uint32_t channelStride = ops::CeilAlign(dataCount, eleBlockSize_);
    uint16_t loopNum = loop;
    uint32_t padNum = tilingData_->onceOutNum > 1 ? repeatTimes * repeatElm - dataCount : 0;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vd0;
        Reg::RegTensor<T> res;
        AscendC::Reg::UnalignRegForStore u0;
        Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
        uint32_t sregChannel = num;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::MaskReg p0 = Reg::UpdateMask<T>(sregChannel);
            auto srcAddr = xLocalAddr + i * repeatElm;
            auto dstAddr = dstLocalAddr + i * repeatElm;
            DuplicateReg<T, OP_TYPE>(res, maskAll);
            for (uint16_t j = 0; j < loopNum; j++) {
                Reg::AddrReg offset = Reg::CreateAddrReg<T>(j, channelStride);
                Reg::LoadAlign(vd0, srcAddr, offset);
                if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
                    Reg::Max(res, vd0, res, p0);
                } else {
                    Reg::Add(res, vd0, res, p0);
                }
            }
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeMaxParaRes<T, OP_TYPE>(res, dstAddr, repeatElm);
            }
            if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D && IS_LAST_LOOP) {
                Reg::Muls(res, res, mulsFactor_, p0);
            }
            Reg::StoreUnAlign(dstAddr, res, u0, repeatElm);
            Reg::StoreUnAlignPost(dstAddr, u0, 0);
        }
        DuplicateValue<T, OP_TYPE>(dstLocalAddr, padNum, dataCount);
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T, int32_t OP_TYPE>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::ComputeSingleNormForAvgNotFp32(int32_t localCurIdx,
                                                                                        int64_t loop, int64_t dataCount)
{
    LocalTensor<float> sumLocal = sumBuf_.Get<float>();
    LocalTensor<T> outLocal = outputBuf_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    auto xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    auto sumLocalAddr = (__ubuf__ float*)sumLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    auto dstLocalAddr = (__ubuf__ T*)outLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    constexpr uint32_t repeatElm = GetVFLen<T, OP_TYPE>();
    uint16_t repeatTimes = CeilDivision(dataCount, repeatElm);
    uint32_t num = dataCount;
    uint32_t channelStride = ops::CeilAlign(dataCount, eleBlockSize_);
    uint16_t loopNum = loop;
    uint32_t padNum = tilingData_->onceOutNum > 1 ? repeatTimes * repeatElm - dataCount : 0;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> in;
        Reg::RegTensor<float> inFp32;
        Reg::RegTensor<float> resFp32;
        Reg::UnalignRegForStore u0;
        uint32_t sregChannel = num;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            Reg::MaskReg p0 = Reg::UpdateMask<float>(sregChannel);
            auto srcAddr = xLocalAddr + i * repeatElm;
            auto sumAddr = sumLocalAddr + i * repeatElm;
            auto dstAddr = dstLocalAddr + i * repeatElm;
            Reg::Duplicate(resFp32, 0);
            for (uint16_t j = 0; j < loopNum; j++) {
                Reg::AddrReg offset = Reg::CreateAddrReg<T>(j, channelStride);
                Reg::LoadAlign(in, srcAddr, offset);
                Reg::UnPack((Reg::RegTensor<uint32_t>&)in, (Reg::RegTensor<uint16_t>&)in);
                Reg::Cast<float, T, castTraitT2Fp32>(inFp32, in, p0);
                Reg::Add(resFp32, inFp32, resFp32, p0);
            }
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeMaxParaRes<float, OP_TYPE>(resFp32, sumAddr, repeatElm);
            }
            if constexpr (!IS_LAST_LOOP) {
                Reg::StoreUnAlign(sumAddr, resFp32, u0, repeatElm);
                Reg::StoreUnAlignPost(sumAddr, u0, 0);
            } else {
                Reg::Muls(resFp32, resFp32, mulsFactor_, p0);
                Reg::Cast<T, float, castTraitFp322T>(in, resFp32, p0);
                Reg::Pack((Reg::RegTensor<uint16_t>&)in, (Reg::RegTensor<uint32_t>&)in);
                Reg::StoreUnAlign(dstAddr, in, u0, repeatElm);
                Reg::StoreUnAlignPost(dstAddr, u0, 0);
            }
        }
        DuplicateValue<float, OP_TYPE>(sumLocalAddr, padNum, dataCount);
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T, int32_t OP_TYPE>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::ComputeSingleWithGatherForAvgNotFp32(int32_t localCurIdx,
                                                                                              int64_t loop,
                                                                                              int64_t dataCount)
{
    LocalTensor<float> sumLocal = sumBuf_.Get<float>();
    LocalTensor<T> outLocal = outputBuf_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    auto xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    auto dstLocalAddr = (__ubuf__ T*)outLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    auto sumLocalAddr = (__ubuf__ float*)sumLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(float);
    uint16_t repeatTimes = loop / repeatElm;
    uint16_t tailLoop = loop - repeatTimes * repeatElm;
    uint32_t channelNum = dataCount;
    uint32_t channelStride = dataCount * repeatElm;
    uint16_t loopNum = dataCount;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> res;
        Reg::RegTensor<T> in;
        Reg::RegTensor<float> inFp32;
        Reg::RegTensor<uint16_t> v0;
        AscendC::Reg::UnalignRegForStore u0;
        Reg::Arange((Reg::RegTensor<int16_t>&)v0, 0);
        uint32_t tailSreg = tailLoop;
        uint32_t mainSreg = repeatElm;
        uint32_t tailLen = tailLoop;
        uint32_t mainLen = repeatElm;
        Reg::MaskReg pTail = Reg::UpdateMask<T>(tailSreg);
        Reg::MaskReg pMain = Reg::UpdateMask<T>(mainSreg);
        Reg::MaskReg maskFp32 = Reg::UpdateMask<float>(mainLen);
        Reg::MaskReg tailMaskFp32 = Reg::UpdateMask<float>(tailLen);
        Reg::Muls(v0, v0, channelNum, pMain);
        for (uint16_t i = 0; i < loopNum; i++) {
            auto srcAddr = xLocalAddr + i;
            auto dstAddr = dstLocalAddr + i;
            auto sumAddr = sumLocalAddr + i;
            Reg::Duplicate(res, 0);
            for (uint16_t j = 0; j < repeatTimes; j++) {
                Reg::Gather(in, srcAddr + j * channelStride, v0, pMain);
                Reg::UnPack((Reg::RegTensor<uint32_t>&)in, (Reg::RegTensor<uint16_t>&)in);
                Reg::Cast<float, T, castTraitT2Fp32>(inFp32, in, maskFp32);
                Reg::Add(res, inFp32, res, maskFp32);
            }
            Reg::RegTensor<float> tmp;
            Reg::Gather(in, srcAddr + repeatTimes * channelStride, v0, pTail);
            Reg::UnPack((Reg::RegTensor<uint32_t>&)in, (Reg::RegTensor<uint16_t>&)in);
            Reg::Cast<float, T, castTraitT2Fp32>(inFp32, in, tailMaskFp32);
            Reg::Add(tmp, inFp32, res, tailMaskFp32);
            Reg::Move<float, Reg::MaskMergeMode::MERGING>(res, tmp, tailMaskFp32);
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::Reduce<Reg::ReduceType::SUM>(res, res, maskAll);
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeMaxRes<float, OP_TYPE>(res, sumAddr, 0);
            }
            if constexpr (!IS_LAST_LOOP) {
                Reg::StoreUnAlign(sumAddr, res, u0, 1);
                Reg::StoreUnAlignPost(sumAddr, u0, 0);
            } else {
                Reg::Muls(res, res, mulsFactor_, maskAll);
                Reg::Cast<T, float, castTraitFp322T>(in, res, maskAll);
                Reg::Pack((Reg::RegTensor<uint16_t>&)in, (Reg::RegTensor<uint32_t>&)in);
                Reg::StoreUnAlign(dstAddr, in, u0, 1);
                Reg::StoreUnAlignPost(dstAddr, u0, 0);
            }
        }
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T, int32_t OP_TYPE>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::ComputeSingleWithGather(int32_t localCurIdx, int64_t loop,
                                                                                 int64_t dataCount)
{
    LocalTensor<T> maxOutLocal = outputBuf_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    using U = typename IndexTypeGet<T>::type;
    T negInf = GetNegInf<T>();
    __ubuf__ T* xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ T* dstLocalAddr = (__ubuf__ T*)maxOutLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(U);
    uint16_t repeatTimes = loop / repeatElm;
    uint16_t tailLoop = loop - repeatTimes * repeatElm;
    uint32_t channelNum = dataCount;
    uint32_t channelStride = dataCount * repeatElm;
    uint16_t loopNum = dataCount;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        Reg::RegTensor<T> res;
        Reg::RegTensor<U> v0;
        AscendC::Reg::UnalignRegForStore u0;
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        uint32_t tailSreg = tailLoop;
        uint32_t mainSreg = repeatElm;
        Reg::MaskReg pTail = Reg::UpdateMask<U>(tailSreg);
        Reg::MaskReg pMain = Reg::UpdateMask<U>(mainSreg);
        Reg::MaskReg maskAllForT = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
        Reg::Muls(v0, v0, channelNum, pMain);
        for (uint16_t i = 0; i < loopNum; i++) {
            auto srcAddr = xLocalAddr + i;
            auto dstAddr = dstLocalAddr + i;
            DuplicateReg<T, OP_TYPE>(res, maskAllForT);
            for (uint16_t j = 0; j < repeatTimes; j++) {
                MaxWithGather<false, OP_TYPE>(res, srcAddr + j * channelStride, v0, pMain);
            }
            Reg::MaskReg maskAll = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
            MaxWithGather<true, OP_TYPE>(res, srcAddr + repeatTimes * channelStride, v0, pTail);
            ReduceAll<T, OP_TYPE>(res, res, maskAll);
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeMaxRes<T, OP_TYPE>(res, dstAddr, 0);
            }
            if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D && IS_LAST_LOOP) {
                Reg::Muls(res, res, mulsFactor_, maskAll);
            }
            Reg::StoreUnAlign(dstAddr, res, u0, 1);
            Reg::StoreUnAlignPost(dstAddr, u0, 0);
        }
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T, int32_t OP_TYPE>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void Pool3DNDHWCBigKernel<T, OP_TYPE>::ComputeSingle(int32_t localCurIdx, int64_t loop,
                                                                       int64_t dataCount)
{
    if (tilingData_->channel * sizeof(T) < GATHER_THRES) {
        if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D || std::is_same<T, float>::value) {
            ComputeSingleWithGather<MERGE, IS_LAST_LOOP>(localCurIdx, loop, dataCount);
        } else {
            ComputeSingleWithGatherForAvgNotFp32<MERGE, IS_LAST_LOOP>(localCurIdx, loop, dataCount);
        }
    } else {
        if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D || std::is_same<T, float>::value) {
            ComputeSingleNorm<MERGE, IS_LAST_LOOP>(localCurIdx, loop, dataCount);
        } else {
            ComputeSingleNormForAvgNotFp32<MERGE, IS_LAST_LOOP>(localCurIdx, loop, dataCount);
        }
    }
}

} // namespace Pool3D
#endif // POOL_3D_NDHWC_BIG_KERNEL_H_
