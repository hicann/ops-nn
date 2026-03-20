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
 * \file avg_pool_big_kernel_nhwc.h
 * \brief avg_pool_big kernel for nhwc
 */
#ifndef AVG_POOL_NHWC_BIG_KERNEL_H_
#define AVG_POOL_NHWC_BIG_KERNEL_H_

#include "avg_pool_common.h"
#include "op_kernel/math_util.h"
#include "avg_pool_struct.h"

namespace AvgPool
{
using namespace AscendC;

static constexpr int32_t NO_SPLIT_KERNEL = 0;
static constexpr int32_t SPLIT_KERNEL_H = 1;
static constexpr int32_t SPLIT_KERNEL_W = 2;
static constexpr int32_t SPLIT_KERNEL_C = 3;
static constexpr int64_t GATHER_THRES = 32;
static constexpr int64_t MOV_ALIGN_THRES = 128;

template <typename T>
class AvgPoolNhwcBigKernel
{
public:
    __aicore__ inline AvgPoolNhwcBigKernel(TPipe* pipe, const AvgPoolBigKernelNhwcTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcKernelSize(int64_t curIdx, int64_t& curkH, int64_t& curkW,
        int64_t& curInOffset);
    template <int32_t SPLIT_MODE>
    __aicore__ inline void BaseCompute(int64_t beginIdx, int64_t endIdx);
    __aicore__ inline void CopyInSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInMultiRows(int64_t offset, int64_t hLen, int64_t wLen, int64_t blockLen);
    __aicore__ inline void CopyInMultiRowsContiguous(int64_t offset, int64_t hLen, int64_t wLen);
    __aicore__ inline void CopyAvgOut(int64_t curIdx);
    __aicore__ inline void CopyOutSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void NoSplitKernelProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset);
    __aicore__ inline void SplitKernelHProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset);
    __aicore__ inline void SplitKernelWProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset);
    __aicore__ inline void SplitChannelProcess(int32_t curIdx, int64_t curkH, int64_t curkW, int64_t curInOffset);
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
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<> sumBuf_;
    TBuf<> outputBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> avgGm_;

    const AvgPoolBigKernelNhwcTilingData* tilingData_;

    int64_t inHW_ = 1;
    int64_t outHW_ = 1;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;
    int64_t channelAlign_ = 0;
    int64_t inStrideH_ = 0;
    int64_t inStrideW_ = 0;
    int32_t maxOutLen_ = 0;
    float mulsFactor_ = 0.0f;
    static constexpr int64_t vRegLen_ = Ops::Base::GetVRegSize() / sizeof(T);
    static constexpr int64_t eleBlockSize_ = Ops::Base::GetUbBlockSize() / sizeof(T);
};

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    constexpr int64_t byteWidth = sizeof(T);
    inHW_ = tilingData_->hInDim * tilingData_->wInDim;
    outHW_ = tilingData_->hOutDim * tilingData_->wOutDim;
    channelAlign_ = Ops::Base::CeilAlign(tilingData_->channel, eleBlockSize_);
    if (GetBlockIdx() < tilingData_->blockTail) {
        beginIdx_ = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx_ = beginIdx_ + tilingData_->blockFactor + 1;
    } else {
        beginIdx_ = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx_ = beginIdx_ + tilingData_->blockFactor;
    }
    maxOutLen_ = min(static_cast<int64_t>((endIdx_ - beginIdx_) * tilingData_->channel), tilingData_->outUbSize);
    inStrideW_ = 0;
    inStrideH_ = tilingData_->wInDim * tilingData_->channel * byteWidth;
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    avgGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * byteWidth);
    pipe_->InitBuffer(outputBuf_, tilingData_->outUbSize * byteWidth);
    if constexpr (!std::is_same<T, float>::value) {
        pipe_->InitBuffer(sumBuf_, tilingData_->outUbSize * byteWidth);
    }
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::Process()
{
    if (tilingData_->tilingMode == NO_SPLIT_KERNEL) {
        BaseCompute<NO_SPLIT_KERNEL>(beginIdx_, endIdx_);
    } else if (tilingData_->tilingMode == SPLIT_KERNEL_H) {
        BaseCompute<SPLIT_KERNEL_H>(beginIdx_, endIdx_);
    } else if (tilingData_->tilingMode == SPLIT_KERNEL_W) {
        BaseCompute<SPLIT_KERNEL_W>(beginIdx_, endIdx_);
    } else if (tilingData_->tilingMode == SPLIT_KERNEL_C) {
        BaseCompute<SPLIT_KERNEL_C>(beginIdx_, endIdx_);
    }
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::CalcKernelSize(int64_t curIdx, int64_t& curkH,
    int64_t& curkW, int64_t& curInOffset)
{
    int64_t cur2D = curIdx % outHW_;
    int64_t curN = curIdx / outHW_;
    int64_t curHo = cur2D / tilingData_->wOutDim;
    int64_t curWo = cur2D % tilingData_->wOutDim;

    int64_t curkPadH = 0;
    CalcKernelSizeCore(PoolParamsForDim{tilingData_->hInDim, curHo, tilingData_->kH, tilingData_->sH,
        tilingData_->tPad, tilingData_->bPad}, curkH, curkPadH, curOriginH_);
    int64_t curkPadW = 0;
    CalcKernelSizeCore(PoolParamsForDim{tilingData_->wInDim, curWo, tilingData_->kW, tilingData_->sW,
        tilingData_->lPad, tilingData_->rPad}, curkW, curkPadW, curOriginW_);

    curOriginIndex_ = (curOriginH_ * tilingData_->wInDim + curOriginW_) * tilingData_->channel;
    curInOffset = curN * inHW_ * tilingData_->channel + curOriginIndex_;

    if (tilingData_->divisorOverride > 0) {
        mulsFactor_ = 1.0f / static_cast<float>(tilingData_->divisorOverride);
    } else if (tilingData_->countIncludePad == 0) {
        mulsFactor_ = curkH * curkW == 0 ? 0 : 1.0f / static_cast<float>(curkH * curkW);
    } else {
        mulsFactor_ = 1.0f / static_cast<float>(curkPadH * curkPadW);
    }
}

template <typename T>
template <int32_t SPLIT_MODE>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::BaseCompute(int64_t beginIdx, int64_t endIdx)
{
    int64_t curkH = 1;
    int64_t curkW = 1;
    int64_t curInOffset = 0;
    // current blockdim range
    for (int64_t idx = beginIdx; idx < endIdx; idx++) {
        CalcKernelSize(idx, curkH, curkW, curInOffset); // compute kernel_size of cur_out, and offset of cur_in
        int32_t localCurIdx = (idx - beginIdx) % tilingData_->onceOutNum;
        if constexpr (SPLIT_MODE == NO_SPLIT_KERNEL) {
            InitOutLocal<true>(localCurIdx); // not need init
            NoSplitKernelProcess(localCurIdx, curkH, curkW, curInOffset);
            CopyAvgOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_H) {
            InitOutLocal<true>(localCurIdx); // need init when localCurIdx is 0
            SplitKernelHProcess(localCurIdx, curkH, curkW, curInOffset);
            CopyAvgOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_W) {
            InitOutLocal<true>(localCurIdx); // need init when localCurIdx is 0
            SplitKernelWProcess(localCurIdx, curkH, curkW, curInOffset);
            CopyAvgOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_C) {
            SplitChannelProcess(idx, curkH, curkW, curInOffset);
        }
    }
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::CopyInSingleRow(int64_t offset, int64_t blockLen)
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
__aicore__ inline void AvgPoolNhwcBigKernel<T>::CopyInMultiRows(int64_t offset, int64_t hLen,
    int64_t wLen, int64_t blockLen)
{
    if (tilingData_->channel * sizeof(T) <= GATHER_THRES) {
        CopyInMultiRowsContiguous(offset, hLen, wLen * tilingData_->channel);
    } else {
        LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = hLen;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->wInDim * tilingData_->channel * sizeof(T);
        loopParams.loop1DstStride = wLen * channelAlign_ * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPadExtParams<T> padExtParams;
        padExtParams.isPad = false;
        padExtParams.leftPadding = 0;
        padExtParams.rightPadding = 0;
        padExtParams.paddingValue = 0;

        DataCopyExtParams extParams;
        extParams.blockCount = wLen;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        inputQue_.EnQue(xLocal);
    }
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::CopyInMultiRowsContiguous(int64_t offset,
    int64_t hLen, int64_t wLen)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();

    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    DataCopyExtParams extParams;
    extParams.blockCount = hLen;
    extParams.blockLen = wLen * sizeof(T);
    extParams.srcStride = (tilingData_->wInDim * tilingData_->channel - wLen) * sizeof(T);
    extParams.dstStride = 0;
    DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);

    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::CopyOutSingleRow(int64_t offset, int64_t blockLen)
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
    DataCopyPad(avgGm_[offset], maxOutLocal, extParams);
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::CopyAvgOut(int64_t curIdx)
{
    int32_t localCurIdx = (curIdx - beginIdx_) % tilingData_->onceOutNum;

    if (localCurIdx == tilingData_->onceOutNum - 1 || curIdx == endIdx_ - 1) {
        CopyOutSingleRow((curIdx - localCurIdx) * tilingData_->channel, (localCurIdx + 1) * tilingData_->channel);
    }
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::ComputeAvg(int64_t length)
{
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

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::NoSplitKernelProcess(int32_t localCurIdx,
    int64_t curkH, int64_t curkW, int64_t curInOffset)
{
    if (curkH * curkW == 0) {
        return;
    }
    CopyInMultiRows(curInOffset, curkH, curkW, tilingData_->channel);
    ComputeSingle<false, true>(localCurIdx, curkW * curkH, tilingData_->channel);
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::SplitKernelHProcess(int32_t localCurIdx,
    int64_t curkH, int64_t curkW, int64_t curInOffset)
{
    if (curkH * curkW == 0) {
        return;
    }
    // 整行搬入
    int64_t hFactor = tilingData_->inUbSize / channelAlign_ / curkW;
    int64_t hLoops = ops::Ceil(curkH, hFactor);
    int64_t hTail = curkH - (hLoops - 1) * hFactor;

    int64_t inputOffset = curInOffset;
    for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
        int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
        bool isLastLoop = hLoop == hLoops - 1;
        CopyInMultiRows(inputOffset, curhFactor, curkW, tilingData_->channel);
        if (!isLastLoop) {
            ComputeSingle<true, false>(localCurIdx, curkW * curhFactor, tilingData_->channel);
        } else {
            ComputeSingle<true, true>(localCurIdx, curkW * curhFactor, tilingData_->channel);
        }
        inputOffset += curhFactor * tilingData_->wInDim * tilingData_->channel;
    }
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::SplitKernelWProcess(int32_t localCurIdx,
    int64_t curkH, int64_t curkW, int64_t curInOffset)
{
    if (curkH * curkW == 0) {
        return;
    }
    // 单行很大，单行循环搬
    int64_t hLoops = curkH;
    int64_t wFactor = tilingData_->inUbSize / channelAlign_;
    int64_t wLoops = ops::Ceil(curkW, wFactor);
    int64_t wTail = curkW - (wLoops - 1) * wFactor;
    
    for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
        int64_t hOffset = curInOffset + hLoop * tilingData_->wInDim * tilingData_->channel;
        for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
            int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
            bool isLastLoop = wLoop == wLoops - 1 && hLoop == hLoops - 1;
            int64_t inputOffset = hOffset + wLoop * wFactor * tilingData_->channel;
            CopyInMultiRows(inputOffset, 1, curFactor, tilingData_->channel);
            if (!isLastLoop) {
                ComputeSingle<true, false>(localCurIdx, curFactor, tilingData_->channel);
            } else {
                ComputeSingle<true, true>(localCurIdx, curFactor, tilingData_->channel);
            }
        }
    }
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::ComputeSum(LocalTensor<T>& xLocal, int64_t dataCount)
{
    LocalTensor<float> sumLocal = sumBuf_.Get<float>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ float* sumLocalAddr = (__local_mem__ float*)sumLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t repeatTimes = static_cast<uint16_t>(ops::Ceil(dataCount, static_cast<int64_t>(repeatElm)));
    uint32_t len = dataCount;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> in;
        MicroAPI::RegTensor<float> inFp32;
        MicroAPI::RegTensor<float> sum;
        MicroAPI::MaskReg mask;
        uint32_t num = len;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = MicroAPI::UpdateMask<float>(num);
            auto sumReg = MicroAPI::CreateAddrReg<float>(i, static_cast<uint16_t>(repeatElm));
            auto srcReg = MicroAPI::CreateAddrReg<T>(i, static_cast<uint16_t>(repeatElm));
            MicroAPI::DataCopy(in, xLocalAddr, srcReg);
            MicroAPI::DataCopy(sum, sumLocalAddr, sumReg);
            MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t>&)in, (MicroAPI::RegTensor<uint16_t>&)in);
            MicroAPI::Cast<float, T, castTraitT2Fp32>(inFp32, in, mask);
            MicroAPI::Add(sum, inFp32, sum, mask);
            MicroAPI::DataCopy(sumLocalAddr, sum, sumReg, mask);
        }
    }
}

template <typename T>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::SplitChannelProcess(int32_t curIdx,
    int64_t curkH, int64_t curkW, int64_t curInOffset)
{
    if (curkH * curkW == 0) {
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
    int64_t hLoops = curkH;
    int64_t wLoops = curkW;
    int64_t cFactor = tilingData_->inUbSize / vRegLen_ * vRegLen_;
    int64_t cLoops = ops::Ceil(tilingData_->channel, cFactor);
    int64_t cTail = tilingData_->channel - (cLoops - 1) * cFactor;
    for (int64_t cLoop = 0; cLoop < cLoops; cLoop++) {
        int32_t curFactor = cLoop == cLoops - 1 ? cTail : cFactor;
        InitOutLocal<true>(0);
        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            int64_t inputOffset = curInOffset + hLoop * tilingData_->wInDim * tilingData_->channel + cLoop * cFactor;
            for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                CopyInSingleRow(inputOffset, curFactor);
                LocalTensor<T> xLocal = inputQue_.DeQue<T>();
                if constexpr (std::is_same<T, float>::value) {
                    LocalTensor<T> sumLocal = outputBuf_.Get<T>();
                    Add(sumLocal, xLocal, sumLocal, curFactor);
                } else {
                    ComputeSum(xLocal, curFactor);
                }
                inputQue_.FreeTensor<T>(xLocal);
                inputOffset += tilingData_->channel;
            }
        }
        ComputeAvg(curFactor);
        CopyOutSingleRow(curIdx * tilingData_->channel + cLoop * cFactor, curFactor);
    }
}

template <typename T>
template <bool CLEAR>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::InitOutLocal(int32_t localCurIdx)
{
    if (localCurIdx != 0) {
        return;
    }

    int32_t maxLocalLen = maxOutLen_;
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);

    if constexpr (!CLEAR) {  // kerel 全载场景无需merge，因此无需初始化output
        return;
    }
    if constexpr (std::is_same<T, float>::value) {
        LocalTensor<T> maxOutLocal = outputBuf_.Get<T>();
        __local_mem__ T* dstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
        constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
        uint16_t repeatTimes = CeilDivision(maxLocalLen, repeatElm);
        uint32_t num = maxLocalLen;
        __local_mem__ T* addr = (__local_mem__ T*)dstAddr;
        __VEC_SCOPE__
        {
            CustomDuplicate<T>(addr, num, repeatTimes);
        }
    } else {
        LocalTensor<float> sumLocal = sumBuf_.Get<float>();
        __local_mem__ float* dstAddr = (__local_mem__ float*)sumLocal.GetPhyAddr();
        constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(float);
        uint16_t repeatTimes = CeilDivision(maxLocalLen, repeatElm);
        uint32_t num = maxLocalLen;
        __local_mem__ float* addr = (__local_mem__ float*)dstAddr;
        __VEC_SCOPE__
        {
            CustomDuplicate<float>(addr, num, repeatTimes);
        }
    }
}

template <typename T>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::ComputeSingleNorm(int32_t localCurIdx, int64_t loop,
    int64_t dataCount)
{
    LocalTensor<T> maxOutLocal = outputBuf_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* dstLocalAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
    uint16_t repeatTimes = CeilDivision(dataCount, repeatElm);
    uint32_t num = dataCount;
    uint32_t channelStride = Ops::Base::CeilAlign(dataCount, eleBlockSize_);
    uint16_t loopNum = loop;
    uint32_t padNum = tilingData_->onceOutNum > 1 ? repeatTimes * repeatElm - dataCount : 0;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vd0;
        MicroAPI::RegTensor<T> res;
        AscendC::MicroAPI::UnalignReg u0;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        uint32_t sregChannel = num;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<T>(sregChannel);
            auto srcAddr = xLocalAddr + i * repeatElm;
            auto dstAddr = dstLocalAddr + i * repeatElm;
            DuplicateReg<T>(res, maskAll);
            for (uint16_t j = 0; j < loopNum; j++) {
                MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(j, channelStride);
                MicroAPI::DataCopy(vd0, srcAddr, offset);
                MicroAPI::Add(res, vd0, res, p0);
            }
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeAvgParaRes<T>(res, dstAddr, repeatElm);
            }
            if constexpr (IS_LAST_LOOP) {
                MicroAPI::Muls(res, res, mulsFactor_, p0);
            }
            MicroAPI::DataCopyUnAlign(dstAddr, res, u0, repeatElm);
            MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
        }
        DuplicateValue<T>(dstLocalAddr, padNum, dataCount);
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::ComputeSingleNormForAvgNotFp32(int32_t localCurIdx,
    int64_t loop, int64_t dataCount)
{
    LocalTensor<float> sumLocal = sumBuf_.Get<float>();
    LocalTensor<T> outLocal = outputBuf_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    auto xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    auto sumLocalAddr = (__local_mem__ float*)sumLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    auto dstLocalAddr = (__local_mem__ T*)outLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(dataCount, repeatElm);
    uint32_t num = dataCount;
    uint32_t channelStride = Ops::Base::CeilAlign(dataCount, eleBlockSize_);
    uint16_t loopNum = loop;
    uint32_t padNum = tilingData_->onceOutNum > 1 ? repeatTimes * repeatElm - dataCount : 0;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> in;
        MicroAPI::RegTensor<float> inFp32;
        MicroAPI::RegTensor<float> resFp32;
        MicroAPI::UnalignReg u0;
        uint32_t sregChannel = num;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<float>(sregChannel);
            auto srcAddr = xLocalAddr + i * repeatElm;
            auto sumAddr = sumLocalAddr + i * repeatElm;
            auto dstAddr = dstLocalAddr + i * repeatElm;
            MicroAPI::Duplicate(resFp32, 0);
            for (uint16_t j = 0; j < loopNum; j++) {
                MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(j, channelStride);
                MicroAPI::DataCopy(in, srcAddr, offset);
                MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t>&)in, (MicroAPI::RegTensor<uint16_t>&)in);
                MicroAPI::Cast<float, T, castTraitT2Fp32>(inFp32, in, p0);
                MicroAPI::Add(resFp32, inFp32, resFp32, p0);
            }
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeAvgParaRes<float>(resFp32, sumAddr, repeatElm);
            }
            if constexpr (!IS_LAST_LOOP) {
                MicroAPI::DataCopyUnAlign(sumAddr, resFp32, u0, repeatElm);
                MicroAPI::DataCopyUnAlignPost(sumAddr, u0, 0);
            } else {
                MicroAPI::Muls(resFp32, resFp32, mulsFactor_, p0);
                MicroAPI::Cast<T, float, castTraitFp322T>(in, resFp32, p0);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)in, (MicroAPI::RegTensor<uint32_t>&)in);
                MicroAPI::DataCopyUnAlign(dstAddr, in, u0, repeatElm);
                MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
            }
        }
        DuplicateValue<float>(sumLocalAddr, padNum, dataCount);
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::ComputeSingleWithGatherForAvgNotFp32(int32_t localCurIdx,
    int64_t loop, int64_t dataCount)
{
    LocalTensor<float> sumLocal = sumBuf_.Get<float>();
    LocalTensor<T> outLocal = outputBuf_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    auto xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    auto dstLocalAddr = (__local_mem__ T*)outLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    auto sumLocalAddr = (__local_mem__ float*)sumLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t repeatTimes = loop / repeatElm;
    uint16_t tailLoop = loop - repeatTimes * repeatElm;
    uint32_t channelNum = dataCount;
    uint32_t channelStride = dataCount * repeatElm;
    uint16_t loopNum = dataCount;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> res;
        MicroAPI::RegTensor<T> in;
        MicroAPI::RegTensor<float> inFp32;
        MicroAPI::RegTensor<uint16_t> v0;
        AscendC::MicroAPI::UnalignReg u0;
        MicroAPI::Arange((MicroAPI::RegTensor<int16_t>&)v0, 0);
        uint32_t tailSreg = tailLoop;
        uint32_t mainSreg = repeatElm;
        uint32_t tailLen = tailLoop;
        uint32_t mainLen = repeatElm;
        MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<T>(tailSreg);
        MicroAPI::MaskReg pMain = MicroAPI::UpdateMask<T>(mainSreg);
        MicroAPI::MaskReg maskFp32 = MicroAPI::UpdateMask<float>(mainLen);
        MicroAPI::MaskReg tailMaskFp32 = MicroAPI::UpdateMask<float>(tailLen);
        MicroAPI::Muls(v0, v0, channelNum, pMain);
        for (uint16_t i = 0; i < loopNum; i++) {
            auto srcAddr = xLocalAddr + i;
            auto dstAddr = dstLocalAddr + i;
            auto sumAddr = sumLocalAddr + i;
            MicroAPI::Duplicate(res, 0);
            for (uint16_t j = 0; j < repeatTimes; j++) {
                MicroAPI::DataCopyGather(in, srcAddr + j * channelStride, v0, pMain);
                MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t>&)in, (MicroAPI::RegTensor<uint16_t>&)in);
                MicroAPI::Cast<float, T, castTraitT2Fp32>(inFp32, in, maskFp32);
                MicroAPI::Add(res, inFp32, res, maskFp32);
            }
            MicroAPI::RegTensor<float> tmp;
            MicroAPI::DataCopyGather(in, srcAddr + repeatTimes * channelStride, v0, pTail);
            MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t>&)in, (MicroAPI::RegTensor<uint16_t>&)in);
            MicroAPI::Cast<float, T, castTraitT2Fp32>(inFp32, in, tailMaskFp32);
            MicroAPI::Add(tmp, inFp32, res, tailMaskFp32);
            MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(res, tmp, tailMaskFp32);
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::ReduceSum(res, res, maskAll);
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeSumRes<float>(res, sumAddr, 0);
            }
            if constexpr (!IS_LAST_LOOP) {
                MicroAPI::DataCopyUnAlign(sumAddr, res, u0, 1);
                MicroAPI::DataCopyUnAlignPost(sumAddr, u0, 0);
            } else {
                MicroAPI::Muls(res, res, mulsFactor_, maskAll);
                MicroAPI::Cast<T, float, castTraitFp322T>(in, res, maskAll);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)in, (MicroAPI::RegTensor<uint32_t>&)in);
                MicroAPI::DataCopyUnAlign(dstAddr, in, u0, 1);
                MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
            }
        }
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::ComputeSingleWithGather(int32_t localCurIdx, int64_t loop,
    int64_t dataCount)
{
    LocalTensor<T> maxOutLocal = outputBuf_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    using U = typename IndexTypeGet<T>::type;
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* dstLocalAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t repeatTimes = loop / repeatElm;
    uint16_t tailLoop = loop - repeatTimes * repeatElm;
    uint32_t channelNum = dataCount;
    uint32_t channelStride = dataCount * repeatElm;
    uint16_t loopNum = dataCount;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<T> res;
        MicroAPI::RegTensor<U> v0;
        AscendC::MicroAPI::UnalignReg u0;
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        uint32_t tailSreg = tailLoop;
        uint32_t mainSreg = repeatElm;
        MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailSreg);
        MicroAPI::MaskReg pMain = MicroAPI::UpdateMask<U>(mainSreg);
        MicroAPI::MaskReg maskAllForT = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Muls(v0, v0, channelNum, pMain);
        for (uint16_t i = 0; i < loopNum; i++) {
            auto srcAddr = xLocalAddr + i;
            auto dstAddr = dstLocalAddr + i;
            DuplicateReg<T>(res, maskAllForT);
            for (uint16_t j = 0; j < repeatTimes; j++) {
                SumWithGather<false>(res, srcAddr + j * channelStride, v0, pMain);
            }
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
            SumWithGather<true>(res, srcAddr + repeatTimes * channelStride, v0, pTail);
            MicroAPI::ReduceSum(res, res, maskAll);
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeSumRes<T>(res, dstAddr, 0);
            }
            if constexpr (IS_LAST_LOOP) {
                MicroAPI::Muls(res, res, mulsFactor_, maskAll);
            }
            MicroAPI::DataCopyUnAlign(dstAddr, res, u0, 1);
            MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
        }
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T>
template <bool MERGE, bool IS_LAST_LOOP>
__aicore__ inline void AvgPoolNhwcBigKernel<T>::ComputeSingle(int32_t localCurIdx, int64_t loop,
    int64_t dataCount)
{
    if (tilingData_->channel * sizeof(T) <= GATHER_THRES) {
        if constexpr (std::is_same<T, float>::value) {
            ComputeSingleWithGather<MERGE, IS_LAST_LOOP>(localCurIdx, loop, dataCount);
        } else {
            ComputeSingleWithGatherForAvgNotFp32<MERGE, IS_LAST_LOOP>(localCurIdx, loop, dataCount);
        }
    } else {
        if constexpr (std::is_same<T, float>::value) {
            ComputeSingleNorm<MERGE, IS_LAST_LOOP>(localCurIdx, loop, dataCount);
        } else {
            ComputeSingleNormForAvgNotFp32<MERGE, IS_LAST_LOOP>(localCurIdx, loop, dataCount);
        }
    }
}

}  // namespace Pool3D
#endif  // POOL_3D_NDHWC_BIG_KERNEL_H_
