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
 * \file avg_pool_big_kernel.h
 * \brief
 */
#ifndef AVG_POOL_BIG_KERNEL_H_
#define AVG_POOL_BIG_KERNEL_H_

#include "avg_pool_common.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "avg_pool_struct.h"

namespace AvgPool
{
using namespace AscendC;

constexpr int32_t OUT_BUFFER_LEN = 1024;
constexpr int32_t NUM128 = 128;

template <typename T>
class AvgPoolBigKernel
{
public:
    __aicore__ inline AvgPoolBigKernel(TPipe* pipe, const AvgPoolBigKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcKernelSize(int64_t curIdx, int64_t& curkH, int64_t& curkW, int64_t& curInOffset);
    template <bool SPLIT_KERNEL>
    __aicore__ inline void BaseCompute(int64_t beginIdx, int64_t endIdx, int64_t maxCount);
    __aicore__ inline void CopyInSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInMultiRows(int64_t offset, int64_t blockLen, int64_t blockCount);
    __aicore__ inline void CopyResultToUb(int64_t curIdx);
    __aicore__ inline void CopyAvgOut(int64_t curIdx);
    __aicore__ inline void NoSplitKernelProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset,
                                                int64_t maxCount);
    __aicore__ inline void SplitKernelProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW, int64_t curInOffset,
                                              int64_t maxCount);
    template <bool CLEAR>
    __aicore__ inline void InitOutLocal(int32_t localCurIdx);
    __aicore__ inline void ComputeSum(int64_t length);
    __aicore__ inline void ComputeAvg();
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }

    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // cast 输入转换
    TBuf<> ubSizeCast_;
    // cast 循环累加结果
    TBuf<> ubLoopReduce_;
    // cast 最终结果
    TBuf<> ubLoopResult_;
    // 输出ub
    TBuf<> uBOutput_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> avgGm_;

    const AvgPoolBigKernelTilingData* tilingData_;

    int64_t inHW_ = 1;
    int64_t outHW_ = 1;
    int64_t curOriginH_ = 0;
    int64_t curOriginW_ = 0;
    int64_t curOriginIndex_ = 0;
    int64_t beginIdx_ = 0;
    int64_t endIdx_ = 0;

    float mulsFactor_ = 0;
};

template <typename T>
__aicore__ inline void AvgPoolBigKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    inHW_ = tilingData_->hInDim * tilingData_->wInDim;
    outHW_ = tilingData_->hOutDim * tilingData_->wOutDim;
    if (GetBlockIdx() < tilingData_->blockTail) {
        beginIdx_ = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx_ = beginIdx_ + tilingData_->blockFactor + 1;
    } else {
        beginIdx_ = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx_ = beginIdx_ + tilingData_->blockFactor;
    }
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    avgGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->maxCount * sizeof(T));
    pipe_->InitBuffer(uBOutput_, OUT_BUFFER_LEN * sizeof(T));
    pipe_->InitBuffer(ubLoopReduce_, NUM128);
    pipe_->InitBuffer(ubLoopResult_, NUM128);

    if constexpr (!std::is_same<T, float>::value) {
        pipe_->InitBuffer(ubSizeCast_, tilingData_->maxCount * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void AvgPoolBigKernel<T>::Process()
{
    if (tilingData_->kH * tilingData_->kW <= tilingData_->maxCount) {
        BaseCompute<false>(beginIdx_, endIdx_, tilingData_->maxCount);
    } else {
        BaseCompute<true>(beginIdx_, endIdx_, tilingData_->maxCount);
    }
}

template <typename T>
__aicore__ inline void AvgPoolBigKernel<T>::CalcKernelSize(int64_t curIdx, int64_t& curkH, int64_t& curkW,
                                                             int64_t& curInOffset)
{
    int64_t cur2D = curIdx % outHW_;
    int64_t curNc = curIdx / outHW_;
    int64_t curHo = cur2D / tilingData_->wOutDim;
    int64_t curWo = cur2D % tilingData_->wOutDim;

    int64_t curkPadH = 0;
    CalcKernelSizeCore(PoolParamsForDim{tilingData_->hInDim, curHo, tilingData_->kH, tilingData_->sH,
        tilingData_->tPad, tilingData_->bPad}, curkH, curkPadH, curOriginH_);
    int64_t curkPadW = 0;
    CalcKernelSizeCore(PoolParamsForDim{tilingData_->wInDim, curWo, tilingData_->kW, tilingData_->sW,
        tilingData_->lPad, tilingData_->rPad}, curkW, curkPadW, curOriginW_);

    curOriginIndex_ = curOriginH_ * tilingData_->wInDim + curOriginW_;
    curInOffset = curNc * inHW_ + curOriginIndex_;

    if (tilingData_->divisorOverride > 0) {
        mulsFactor_ = 1.0f / static_cast<float>(tilingData_->divisorOverride);
    } else if (tilingData_->countIncludePad == 0) {
        mulsFactor_ = curkH * curkW == 0 ? 0 : 1.0f / static_cast<float>(curkH * curkW);
    } else {
        mulsFactor_ = 1.0f / static_cast<float>(curkPadH * curkPadW);
    }
}

template <typename T>
template <bool SPLIT_KERNEL>
__aicore__ inline void AvgPoolBigKernel<T>::BaseCompute(int64_t beginIdx, int64_t endIdx, int64_t maxCount)
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
        CopyResultToUb(localCurIdx);
        CopyAvgOut(idx);
    }
}

template <typename T>
__aicore__ inline void AvgPoolBigKernel<T>::CopyInSingleRow(int64_t offset, int64_t blockLen)
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
__aicore__ inline void AvgPoolBigKernel<T>::CopyInMultiRows(int64_t offset, int64_t blockLen, int64_t blockCount)
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
__aicore__ inline void AvgPoolBigKernel<T>::CopyResultToUb(int64_t curIdx)
{
    LocalTensor<T> uboutLocal = uBOutput_.Get<T>();
    __local_mem__ T* dstAddr = (__local_mem__ T*)uboutLocal.GetPhyAddr() + curIdx;

    LocalTensor<T> ubResult = ubLoopResult_.Get<T>();
    __local_mem__ T* srcAddr = (__local_mem__ T*)ubResult.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> res;
        MicroAPI::UnalignReg u0;
        MicroAPI::DataCopyUnAlignPre(u0, srcAddr);
        MicroAPI::DataCopyUnAlign(res, u0, srcAddr, ONE);

        MicroAPI::UnalignReg u1;
        MicroAPI::DataCopyUnAlign(dstAddr, res, u1, ONE);
        MicroAPI::DataCopyUnAlignPost(dstAddr, u1, 0);
    }
}

template <typename T>
__aicore__ inline void AvgPoolBigKernel<T>::CopyAvgOut(int64_t curIdx)
{
    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
    int32_t localCurIdx = (curIdx - beginIdx_) % maxLocalLen;

    if (localCurIdx == maxLocalLen - 1 || curIdx == endIdx_ - 1) {
        LocalTensor<T> avgOutLocal = uBOutput_.Get<T>();
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = (localCurIdx + 1) * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        DataCopyPad(avgGm_[curIdx - localCurIdx], avgOutLocal, extParams);
    }
}

template <typename T>
__aicore__ inline void AvgPoolBigKernel<T>::NoSplitKernelProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW,
                                                                   int64_t curInOffset, int64_t maxCount)
{
    CopyInMultiRows(curInOffset, curkW, curkH);
    ComputeSum(curkW * curkH);
    ComputeAvg();
}

template <typename T>
__aicore__ inline void AvgPoolBigKernel<T>::SplitKernelProcess(int32_t localCurIdx, int64_t curkH, int64_t curkW,
                                                                 int64_t curInOffset, int64_t maxCount)
{
    int64_t realIndex = 0;
    int64_t inputOffset = curInOffset;
    int64_t kernelOffset = curOriginIndex_;
    int64_t maxIndex = 0;
    // 单行可搬入，一次搬多行
    if (curkW <= maxCount) {
        int64_t hFactor = maxCount / curkW;
        int64_t hLoops = (curkH + hFactor - 1) / hFactor;
        int64_t hTail = curkH - (hLoops - 1) * hFactor;
        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            int32_t curhFactor = hLoop == hLoops - 1 ? hTail : hFactor;
            CopyInMultiRows(inputOffset, curkW, curhFactor);
            ComputeSum(curkW * curhFactor);
            inputOffset += curhFactor * tilingData_->wInDim;
            kernelOffset += curhFactor * tilingData_->wInDim;
        }
    // 单行很大，单行循环搬
    } else {
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
                ComputeSum(curFactor);
                inputOffset += curFactor;
                kernelOffset += curFactor;
            }
        }
    }
    ComputeAvg();
}

template <typename T>
template <bool CLEAR>
__aicore__ inline void AvgPoolBigKernel<T>::InitOutLocal(int32_t localCurIdx)
{
    // 清零ubLoopReduce_的第一个值，确保累加结果正确
    float Float32Zero = 0.0f;
    LocalTensor<float> loopReduce = ubLoopReduce_.Get<float>();
    Duplicate<float>(loopReduce, Float32Zero, ONE);

    if (localCurIdx != 0) {
        return;
    }

    constexpr int32_t maxLocalLen = OUT_BUFFER_LEN;
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);

    if constexpr (!CLEAR) {
        return;
    }
    LocalTensor<T> avgOutLocal = uBOutput_.Get<T>();
    __local_mem__ T* dstAddr = (__local_mem__ T*)avgOutLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
    uint16_t repeatTimes = CeilDivision(maxLocalLen, repeatElm);
    uint32_t num = maxLocalLen;

    T zero = T(0);
    __local_mem__ T* addr = (__local_mem__ T*)dstAddr;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> v0;
        MicroAPI::Duplicate(v0, zero);
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
__aicore__ inline void AvgPoolBigKernel<T>::ComputeAvg()
{
    if constexpr (std::is_same<T, float>::value) {  
        LocalTensor<T> loopReduce = ubLoopReduce_.Get<T>();
        LocalTensor<T> resultLocal = ubLoopResult_.Get<T>();
        Muls(resultLocal, loopReduce, mulsFactor_, ONE);
    } else {
        LocalTensor<float> loopReduce = ubLoopReduce_.Get<float>();
        Muls(loopReduce, loopReduce, mulsFactor_, ONE);
        LocalTensor<T> resultLocal = ubLoopResult_.Get<T>();
        Cast(resultLocal, loopReduce, RoundMode::CAST_ROUND, ONE);
    }
}

template <typename T>
__aicore__ inline void AvgPoolBigKernel<T>::ComputeSum(int64_t length)
{
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    if constexpr (std::is_same<T, float>::value) {
        LocalTensor<T> loopReduce = ubLoopReduce_.Get<T>();
        ReduceSum<T>(xLocal, xLocal, xLocal, length);
        Add(loopReduce, loopReduce, xLocal, ONE);
    } else {
        LocalTensor<float> ubCast = ubSizeCast_.Get<float>();
        Cast(ubCast, xLocal, RoundMode::CAST_NONE, length);
        ReduceSum<float>(ubCast, ubCast, ubCast, length);
        LocalTensor<float> loopReduce = ubLoopReduce_.Get<float>();
        Add(loopReduce, loopReduce, ubCast, ONE);
    }
    inputQue_.FreeTensor<T>(xLocal);
}

}  // namespace AvgPool
#endif  // AVG_POOL_BIG_KERNEL_H_
