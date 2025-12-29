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
 * \file max_pool_v3_big_kernel.h
 * \brief
 */
#ifndef MAX_POOL_V3_BIG_KERNEL_H_
#define MAX_POOL_V3_BIG_KERNEL_H_

#include "max_pool_v3_common.h"

namespace MaxPoolV3 {
using namespace AscendC;

constexpr int32_t OUT_BUFFER_LEN = 1024;

template <typename T>
class MaxPoolV3BigKernel {
public:
    __aicore__ inline MaxPoolV3BigKernel(TPipe* pipe, const MaxPoolV3BigKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcKernelSize(int64_t curIdx, int64_t& curkH, int64_t& curkW, int64_t& curInOffset);
    template <bool SPLIT_KERNEL>
    __aicore__ inline void BaseCompute(int64_t beginIdx, int64_t endIdx, int64_t maxCount);
    __aicore__ inline void CopyInSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInMultiRows(int64_t offset, int64_t blockLen, int64_t blockCount);
    __aicore__ inline void CopyMaxOut(int64_t curIdx);
    __aicore__ inline void NoSplitKernelProcess(
        int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t maxCount);
    __aicore__ inline void SplitKernelProcess(
        int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t maxCount);
    template <bool MERGE, bool SPLITKW>
    __aicore__ inline void ComputeSingle(int32_t localCurIdx, int64_t dataCount);
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

    const MaxPoolV3BigKernelTilingData* tilingData_;

    int64_t inHW_ = 1;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;
};

template <typename T>
__aicore__ inline void MaxPoolV3BigKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    inHW_ = tilingData_->hInDim * tilingData_->wInDim;
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

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->maxCount * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, OUT_BUFFER_LEN * sizeof(T));
}

template <typename T>
__aicore__ inline void MaxPoolV3BigKernel<T>::Process()
{
    if (tilingData_->kH * tilingData_->kW <= tilingData_->maxCount) {
        BaseCompute<false>(beginIdx_, endIdx_, tilingData_->maxCount);
    } else {
        BaseCompute<true>(beginIdx_, endIdx_, tilingData_->maxCount);
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3BigKernel<T>::CalcKernelSize(
    int64_t curIdx, int64_t& curkH, int64_t& curkW, int64_t& curInOffset)
{
    if (tilingData_->isSigOut) {
        curInOffset = curIdx * inHW_;
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

    curOriginIndex_ = curOriginH_ * tilingData_->wInDim + curOriginW_;
    curInOffset = curNc * inHW_ + curOriginIndex_;
}

template <typename T>
template <bool SPLIT_KERNEL>
__aicore__ inline void MaxPoolV3BigKernel<T>::BaseCompute(int64_t beginIdx, int64_t endIdx, int64_t maxCount)
{
    int64_t curkH = 1;
    int64_t curkW = 1;
    int64_t curInOffset = 0;
    // current blockdim range
    for (int64_t idx = beginIdx; idx < endIdx; idx++) {
        CalcKernelSize(idx, curkH, curkW, curInOffset);
        constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
        int32_t localCurIdx = (idx - beginIdx) % maxLocalLen;
        if constexpr (SPLIT_KERNEL) {
            InitOutLocal<true>(localCurIdx);
            SplitKernelProcess(localCurIdx, curkH, curkW, curInOffset, maxCount);
        } else {
            InitOutLocal<false>(localCurIdx);
            NoSplitKernelProcess(localCurIdx, curkH, curkW, curInOffset, maxCount);
        }
        CopyMaxOut(idx);
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3BigKernel<T>::CopyInSingleRow(int64_t offset, int64_t blockLen)
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
__aicore__ inline void MaxPoolV3BigKernel<T>::CopyInMultiRows(int64_t offset, int64_t blockLen, int64_t blockCount)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();

    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    DataCopyExtParams extParams;
    extParams.blockCount = blockCount;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = (tilingData_->wInDim - blockLen) * sizeof(T);
    extParams.dstStride = 0;
    DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
    inputQue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MaxPoolV3BigKernel<T>::CopyMaxOut(int64_t curIdx)
{
    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
    int32_t localCurIdx = (curIdx - beginIdx_) % maxLocalLen;

    if (localCurIdx == maxLocalLen - 1 || curIdx == endIdx_ - 1) {
        LocalTensor<T> maxOutLocal = maxUBOutput_.Get<T>();

        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = (localCurIdx + 1) * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        DataCopyPad(maxGm_[curIdx - localCurIdx], maxOutLocal, extParams);
    }
}

template <typename T>
__aicore__ inline void MaxPoolV3BigKernel<T>::NoSplitKernelProcess(
    int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t maxCount)
{
    CopyInMultiRows(curInOffset, curkW, curkH);
    ComputeSingle<false, false>(localCurIdx, curkW * curkH);
}

template <typename T>
__aicore__ inline void MaxPoolV3BigKernel<T>::SplitKernelProcess(
    int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset, int64_t maxCount)
{
    int64_t realIndex = 0;
    int64_t inputOffset = curInOffset;
    int64_t kernelOffset = curOriginIndex_;
    int64_t maxIndex = 0;

    if (curkW <= maxCount) {
        // 整行搬入
        int64_t hFactor = maxCount / curkW;
        int64_t hLoops = (curkH + hFactor - 1) / hFactor;
        int64_t hTail = curkH - (hLoops - 1) * hFactor;

        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
            CopyInMultiRows(inputOffset, curkW, curhFactor);
            ComputeSingle<true, false>(localCurIdx, curkW * curhFactor);
            inputOffset += curhFactor * tilingData_->wInDim;
            kernelOffset += curhFactor * tilingData_->wInDim;
        }
    } else {
        // 单行很大，单行循环搬
        int64_t hLoops = curkH;
        int64_t wFactor = maxCount;
        int64_t wLoops = (curkW + wFactor - 1) / wFactor;
        int64_t wTail = curkW - (wLoops - 1) * wFactor;

        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            inputOffset = curInOffset + hLoop * tilingData_->wInDim;
            kernelOffset = curOriginIndex_ + hLoop * tilingData_->wInDim;
            for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                int32_t curFactor = wLoop == wLoops - 1 ? wTail : wFactor;
                CopyInSingleRow(inputOffset, curFactor);
                ComputeSingle<true, true>(localCurIdx, curFactor);
                inputOffset += curFactor;
                kernelOffset += curFactor;
            }
        }
    }
}

template <typename T>
template <bool CLEAR>
__aicore__ inline void MaxPoolV3BigKernel<T>::InitOutLocal(int32_t localCurIdx)
{
    if (localCurIdx != 0) {
        return;
    }

    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
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

    T negInf = GetNegInf<T>();
    __local_mem__ T* addr = (__local_mem__ T*)dstAddr;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> v0;
        if constexpr (IsSame<T, bfloat16_t>::value) {
            MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF);
        } else {
            MicroAPI::Duplicate(v0, negInf);
        }
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<T>(num);
            if constexpr (sizeof(T) == B64) {
                MicroAPI::DataCopy(addr + i * repeatElm, v0, p0);
            } else {
                MicroAPI::AddrReg offsetReg = MicroAPI::CreateAddrReg<T>(i, repeatElm);
                MicroAPI::DataCopy(addr, v0, offsetReg, p0);
            }
        }
    }
}

template <typename T>
template <bool MERGE, bool SPLITKW>
__aicore__ inline void MaxPoolV3BigKernel<T>::ComputeSingle(int32_t localCurIdx, int64_t dataCount)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.Get<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    T negInf = GetNegInf<T>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* dstLocalAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
    uint16_t repeatTimes = CeilDivision(dataCount, repeatElm);
    uint32_t num = repeatTimes * repeatElm; // 需要vreg_len对齐
    uint32_t padNum = num - dataCount;
    __VEC_SCOPE__
    {
        DuplicateNegInf<T>(xLocalAddr, padNum, dataCount);
        MicroAPI::RegTensor<T> vd0;
        MicroAPI::RegTensor<T> res;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        if constexpr (IsSame<T, bfloat16_t>::value) {
            MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
        } else {
            MicroAPI::Duplicate(res, negInf);
        }
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<T>(num);
            if constexpr (sizeof(T) == B64) {
                MicroAPI::DataCopy(vd0, xLocalAddr + i * repeatElm);
            } else {
                MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatElm);
                MicroAPI::DataCopy(vd0, xLocalAddr, offset);
            }
            MicroAPI::Max(res, vd0, res, maskAll);
        }
        ReduceMaxAll<T>(res, res, maskAll);
        if constexpr (MERGE) {
            // merge cur result with last result
            MergeMaxRes<T>(res, dstLocalAddr, localCurIdx);
        }
        StoreElement<T>(dstLocalAddr, res, localCurIdx, 1);
    }
    inputQue_.FreeTensor<T>(xLocal);
}

} // namespace MaxPoolV3
#endif // MAX_POOL_V3_BIG_KERNEL_H_
