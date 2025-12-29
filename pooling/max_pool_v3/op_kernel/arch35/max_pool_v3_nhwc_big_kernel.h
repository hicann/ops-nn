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
 * \file max_pool_v3_nhwc_big_kernel.h
 * \brief
 */
#ifndef MAX_POOL_V3_NHWC_BIG_KERNEL_H_
#define MAX_POOL_V3_NHWC_BIG_KERNEL_H_

#include "max_pool_v3_common.h"

namespace MaxPoolV3 {
using namespace AscendC;

static constexpr int32_t NO_SPLIT_KERNEL = 0;
static constexpr int32_t SPLIT_KERNEL_H = 1;
static constexpr int32_t SPLIT_KERNEL_W = 2;
static constexpr int32_t SPLIT_KERNEL_C = 3;
static constexpr int64_t GATHER_THRES = 32;

template <typename T>
__aicore__ inline void MergeMaxParaRes(MicroAPI::RegTensor<T>& res, __local_mem__ T* dstLocalAddr, int32_t num)
{
    // merge cur result with pre result
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<T> lastRes;
    AscendC::MicroAPI::UnalignReg u0;
    auto curSrcAddr = dstLocalAddr;
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    AscendC::MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
    AscendC::MicroAPI::DataCopyUnAlign(lastRes, u0, curSrcAddr, num);
    MicroAPI::Max(res, res, lastRes, pregAll);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

template <typename T>
class MaxPoolV3NHWCBigKernel {
public:
    __aicore__ inline MaxPoolV3NHWCBigKernel(TPipe* pipe, const MaxPoolV3NHWCBigKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcKernelSize(int64_t curIdx, int64_t& curkH, int64_t& curkW, int64_t& curInOffset);
    template <int32_t SPLIT_MODE>
    __aicore__ inline void BaseCompute(int64_t beginIdx, int64_t endIdx);
    __aicore__ inline void CopyInSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInMultiRows(int64_t offset, int64_t rows, int64_t cols, int64_t blockLen);
    __aicore__ inline void CopyInMultiRowsContiguous(int64_t offset, int64_t rows, int64_t cols);
    __aicore__ inline void CopyMaxOut(int64_t curIdx);
    __aicore__ inline void CopyOutSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void NoSplitKernelProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset);
    __aicore__ inline void SplitKernelHProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset);
    __aicore__ inline void SplitKernelWProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset);
    __aicore__ inline void SplitChannelProcess(int32_t curIdx, int64_t curkH, int64_t curkW, int64_t curInOffset);
    template <bool MERGE>
    __aicore__ inline void ComputeSingle(int32_t localCurIdx, int64_t loop, int64_t dataCount);
    template <bool MERGE>
    __aicore__ inline void ComputeSingleWithGather(int32_t localCurIdx, int64_t loop, int64_t dataCount);
    template <bool MERGE>
    __aicore__ inline void ComputeSingleNorm(int32_t localCurIdx, int64_t loop, int64_t dataCount);
    template <bool CLEAR>
    __aicore__ inline void InitOutLocal(int32_t localCurIdx);
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }

    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // 输出ub
    TBuf<> maxUBOutput_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;

    const MaxPoolV3NHWCBigKernelTilingData* tilingData_;

    int64_t inHW_ = 1;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;
    int64_t channelAlign_ = 0;
    int64_t vRegLen_ = Ops::Base::GetVRegSize() / sizeof(T);
    int64_t eleBlockSize_ = Ops::Base::GetUbBlockSize() / sizeof(T);
};

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    inHW_ = tilingData_->hInDim * tilingData_->wInDim;
    channelAlign_ = Ops::Base::CeilAlign(tilingData_->channel, eleBlockSize_);
    if (GetBlockIdx() < tilingData_->blockTail) {
        beginIdx_ = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx_ = beginIdx_ + tilingData_->blockFactor + 1;
    } else {
        beginIdx_ = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx_ = beginIdx_ + tilingData_->blockFactor;
    }
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, tilingData_->outUbSize * sizeof(T));
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::Process()
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
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::CalcKernelSize(
    int64_t curIdx, int64_t& curkH, int64_t& curkW, int64_t& curInOffset)
{
    if (tilingData_->isSigOut) {
        curInOffset = curIdx * inHW_ * tilingData_->channel;
        curOriginIndex_ = 0;
        curkH = min(tilingData_->kH - tilingData_->tPad, tilingData_->hInDim);
        curkW = min(tilingData_->kW - tilingData_->lPad, tilingData_->wInDim);
        return;
    }
    int64_t outHW = tilingData_->hOutDim * tilingData_->wOutDim;
    int64_t cur2D = curIdx % outHW;
    int64_t curNc = curIdx / outHW;
    int64_t curHo = cur2D / tilingData_->wOutDim;
    int64_t curWo = cur2D % tilingData_->wOutDim;

    curOriginH_ = tilingData_->sH * curHo - tilingData_->tPad;
    if (curOriginH_ < 0) {
        curkH = min(tilingData_->kH + curOriginH_, tilingData_->hInDim);
        curOriginH_ = 0;
    } else {
        curkH = min(tilingData_->hInDim - curOriginH_, tilingData_->kH);
    }

    curOriginW_ = tilingData_->sW * curWo - tilingData_->lPad;
    if (curOriginW_ < 0) {
        curkW = min(tilingData_->kW + curOriginW_, tilingData_->wInDim);
        curOriginW_ = 0;
    } else {
        curkW = min(tilingData_->wInDim - curOriginW_, tilingData_->kW);
    }

    curOriginIndex_ = (curOriginH_ * tilingData_->wInDim + curOriginW_) * tilingData_->channel;
    curInOffset = curNc * inHW_ * tilingData_->channel + curOriginIndex_;
}

template <typename T>
template <int32_t SPLIT_MODE>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::BaseCompute(int64_t beginIdx, int64_t endIdx)
{
    int64_t curkH = 1;
    int64_t curkW = 1;
    int64_t curInOffset = 0;
    // current blockdim range
    for (int64_t idx = beginIdx; idx < endIdx; idx++) {
        CalcKernelSize(idx, curkH, curkW, curInOffset);
        int32_t localCurIdx = (idx - beginIdx) % tilingData_->onceOutNum;
        if constexpr (SPLIT_MODE == NO_SPLIT_KERNEL) {
            InitOutLocal<false>(localCurIdx);
            NoSplitKernelProcess(localCurIdx, curkH, curkW, curInOffset);
            CopyMaxOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_H) {
            InitOutLocal<true>(localCurIdx);
            SplitKernelHProcess(localCurIdx, curkH, curkW, curInOffset);
            CopyMaxOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_W) {
            InitOutLocal<true>(localCurIdx);
            SplitKernelWProcess(localCurIdx, curkH, curkW, curInOffset);
            CopyMaxOut(idx);
        } else if constexpr (SPLIT_MODE == SPLIT_KERNEL_C) {
            SplitChannelProcess(idx, curkH, curkW, curInOffset);
        }
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::CopyInSingleRow(int64_t offset, int64_t blockLen)
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
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::CopyInMultiRows(
    int64_t offset, int64_t rows, int64_t cols, int64_t blockLen)
{
    if (tilingData_->channel * sizeof(T) <= GATHER_THRES) {
        CopyInMultiRowsContiguous(offset, rows, cols * tilingData_->channel);
    } else {
        LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = rows;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->wInDim * tilingData_->channel * sizeof(T);
        loopParams.loop1DstStride = cols * channelAlign_ * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPadExtParams<T> padExtParams;
        padExtParams.isPad = false;
        padExtParams.leftPadding = 0;
        padExtParams.rightPadding = 0;
        padExtParams.paddingValue = 0;

        DataCopyExtParams extParams;
        extParams.blockCount = cols;
        extParams.blockLen = blockLen * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        inputQue_.EnQue(xLocal);
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::CopyInMultiRowsContiguous(int64_t offset, int64_t rows, int64_t cols)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();

    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    DataCopyExtParams extParams;
    extParams.blockCount = rows;
    extParams.blockLen = cols * sizeof(T);
    extParams.srcStride = (tilingData_->wInDim * tilingData_->channel - cols) * sizeof(T);
    extParams.dstStride = 0;
    DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);

    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::CopyOutSingleRow(int64_t offset, int64_t blockLen)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.Get<T>();
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

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::CopyMaxOut(int64_t curIdx)
{
    int32_t localCurIdx = (curIdx - beginIdx_) % tilingData_->onceOutNum;

    if (localCurIdx == tilingData_->onceOutNum - 1 || curIdx == endIdx_ - 1) {
        CopyOutSingleRow((curIdx - localCurIdx) * tilingData_->channel, (localCurIdx + 1) * tilingData_->channel);
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::NoSplitKernelProcess(
    int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset)
{
    CopyInMultiRows(curInOffset, curkH, curkW, tilingData_->channel);
    ComputeSingle<false>(localCurIdx, curkW * curkH, tilingData_->channel);
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::SplitKernelHProcess(
    int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset)
{
    int64_t realIndex = 0;
    int64_t inputOffset = curInOffset;
    int64_t maxIndex = 0;

    // 整行搬入
    int64_t hFactor = tilingData_->inUbSize / channelAlign_ / curkW;
    int64_t hLoops = (curkH + hFactor - 1) / hFactor;
    int64_t hTail = curkH - (hLoops - 1) * hFactor;

    for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
        int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
        CopyInMultiRows(inputOffset, curhFactor, curkW, tilingData_->channel);
        ComputeSingle<true>(localCurIdx, curkW * curhFactor, tilingData_->channel);
        inputOffset += curhFactor * tilingData_->wInDim * tilingData_->channel;
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::SplitKernelWProcess(
    int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset)
{
    // 单行很大，单行循环搬
    int64_t hLoops = curkH;
    int64_t wFactor = tilingData_->inUbSize / channelAlign_;
    int64_t wLoops = (curkW + wFactor - 1) / wFactor;
    int64_t wTail = curkW - (wLoops - 1) * wFactor;

    for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
        int64_t inputOffset = curInOffset + hLoop * tilingData_->wInDim * tilingData_->channel;
        for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
            int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
            CopyInMultiRows(inputOffset, 1, curFactor, tilingData_->channel);
            ComputeSingle<true>(localCurIdx, curFactor, tilingData_->channel);
            inputOffset += curFactor * tilingData_->channel;
        }
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::SplitChannelProcess(
    int32_t curIdx, int64_t curkH, int64_t curkW, int64_t curInOffset)
{
    // 单行很大，单行循环搬
    int64_t hLoops = curkH;
    int64_t wLoops = curkW;
    int64_t cFactor = tilingData_->inUbSize / vRegLen_ * vRegLen_;
    int64_t cLoops = (tilingData_->channel + cFactor - 1) / cFactor;
    int64_t cTail = tilingData_->channel - (cLoops - 1) * cFactor;
    for (int64_t cLoop = 0; cLoop < cLoops; cLoop++) {
        int32_t curFactor = cLoop == cLoops - 1 ? cTail : cFactor;
        InitOutLocal<true>(0);
        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            int64_t inputOffset = curInOffset + hLoop * tilingData_->wInDim * tilingData_->channel + cLoop * cFactor;
            for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                CopyInSingleRow(inputOffset, curFactor);
                LocalTensor<T> maxOutLocal = maxUBOutput_.Get<T>();
                LocalTensor<T> xLocal = inputQue_.DeQue<T>();
                Max(maxOutLocal, xLocal, maxOutLocal, curFactor);
                inputQue_.FreeTensor<T>(xLocal);
                inputOffset += tilingData_->channel;
            }
        }
        CopyOutSingleRow(curIdx * tilingData_->channel + cLoop * cFactor, curFactor);
    }
}

template <typename T>
template <bool CLEAR>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::InitOutLocal(int32_t localCurIdx)
{
    if (localCurIdx != 0) {
        return;
    }

    int32_t maxLocalLen = tilingData_->outUbSize;
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);

    if constexpr (!CLEAR) { // kerel 全载场景无需merge，因此无需初始化output
        return;
    }
    LocalTensor<T> maxOutLocal = maxUBOutput_.Get<T>();
    __local_mem__ T* dstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
    uint16_t repeatTimes = CeilDivision(maxLocalLen, repeatElm);
    uint32_t num = maxLocalLen;
    if constexpr (sizeof(T) == B64) {
        constexpr uint32_t twoTraitElm = 2 * Ops::Base::GetVRegSize() / sizeof(T);
        repeatTimes = CeilDivision(maxLocalLen, twoTraitElm);
    }
    __local_mem__ T* addr = (__local_mem__ T*)dstAddr;
    __VEC_SCOPE__
    {
        CustomDuplicate(addr, num, repeatTimes);
    }
}

template <typename T>
template <bool MERGE>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::ComputeSingleNorm(
    int32_t localCurIdx, int64_t loop, int64_t dataCount)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    T negInf = GetNegInf<T>();
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
            if constexpr (IsSame<T, bfloat16_t>::value) {
                MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
            } else {
                MicroAPI::Duplicate(res, negInf);
            }
            for (uint16_t j = 0; j < loopNum; j++) {
                if constexpr (sizeof(T) == B64) {
                    MicroAPI::DataCopy(vd0, srcAddr + j * channelStride);
                } else {
                    MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(j, channelStride);
                    MicroAPI::DataCopy(vd0, srcAddr, offset);
                }
                MicroAPI::Max(res, vd0, res, p0);
            }
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeMaxParaRes<T>(res, dstAddr, repeatElm);
            }
            MicroAPI::DataCopyUnAlign(dstAddr, res, u0, repeatElm);
            MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
        }
        DuplicateNegInf<T>(dstLocalAddr, padNum, dataCount);
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T>
template <bool MERGE>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::ComputeSingleWithGather(
    int32_t localCurIdx, int64_t loop, int64_t dataCount)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    using U = typename IndexTypeGet<T>::type;
    using M = typename GetComputeType<T>::type;
    T negInf = GetNegInf<T>();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr() + localCurIdx * tilingData_->channel;
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t repeatTimes = loop / repeatElm;
    uint16_t tailLoop = loop - repeatTimes * repeatElm;
    uint32_t channelNum = dataCount;
    uint32_t channelStride = dataCount * repeatElm;
    uint16_t loopNum = dataCount;
    __VEC_SCOPE__
    {
        using RegDstT = typename std::conditional<
            sizeof(M) == B64, MicroAPI::RegTensor<M, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<M>>::type;
        using regType = typename VciTypeGet<U>::type;
        using gatherType = typename GetGatherType<M>::type;
        RegDstT res;
        MicroAPI::RegTensor<U> v0;
        AscendC::MicroAPI::UnalignReg u0;
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        uint32_t tailSreg = tailLoop;
        uint32_t mainSreg = repeatElm;
        MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailSreg);
        MicroAPI::MaskReg pMain = MicroAPI::UpdateMask<U>(mainSreg);
        MicroAPI::Muls(v0, v0, channelNum, pMain);
        for (uint16_t i = 0; i < loopNum; i++) {
            auto srcAddr = xLocalAddr + i;
            auto dstAddr = dstLocalAddr + i;
            if constexpr (IsSame<T, bfloat16_t>::value) {
                MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
            } else {
                MicroAPI::Duplicate(res, negInf);
            }
            for (uint16_t j = 0; j < repeatTimes; j++) {
                MaxWithGather<false>(res, srcAddr + j * channelStride, v0, pMain);
            }
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
            MaxWithGather<true>(res, srcAddr + repeatTimes * channelStride, v0, pTail);
            if constexpr (sizeof(M) == 1) {
                MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(maskAll, maskAll);
            }
            ReduceMaxAll<M>(res, res, maskAll);
            if constexpr (MERGE) {
                // merge cur result with last result
                MergeMaxRes<M>(res, dstAddr, 0);
            }
            MicroAPI::DataCopyUnAlign(dstAddr, res, u0, 1);
            MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
        }
    }
    inputQue_.FreeTensor<T>(xLocal);
}

template <typename T>
template <bool MERGE>
__aicore__ inline void MaxPoolV3NHWCBigKernel<T>::ComputeSingle(int32_t localCurIdx, int64_t loop, int64_t dataCount)
{
    if (tilingData_->channel * sizeof(T) <= GATHER_THRES) {
        ComputeSingleWithGather<MERGE>(localCurIdx, loop, dataCount);
    } else {
        ComputeSingleNorm<MERGE>(localCurIdx, loop, dataCount);
    }
}

} // namespace MaxPoolV3
#endif // MAX_POOL_V3_NHWC_BIG_KERNEL_H_
