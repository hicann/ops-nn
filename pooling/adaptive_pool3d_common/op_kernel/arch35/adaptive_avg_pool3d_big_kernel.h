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
 * \file adaptive_avg_pool3d_big_kernel.h
 * \brief
 */
#ifndef ADAPTIVE_AVG_POOL3D_BIG_KERNEL_H_
#define ADAPTIVE_AVG_POOL3D_BIG_KERNEL_H_

#include "adaptive_pool3d_big_kernel.h"
#include "pool_utils/arch35/compute/adaptive_avg_pool_big_kernel_compute.h"

namespace AdaptivePool3d {
using namespace AscendC;
constexpr int32_t STORE_ADD_BUFFER = 1024;

template <typename T>
class AdaptiveAvgPool3dBigKernel : public AdaptivePool3dBigKernel<T> {
public:
    __aicore__ inline AdaptiveAvgPool3dBigKernel(
        const AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData& tilingData, TPipe& pipe)
        : AdaptivePool3dBigKernel<T>(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOutputBuffer();
    template <typename U>
    __aicore__ inline void InitStoreOutBuffer();
    __aicore__ inline void BaseCompute(int64_t curIdx);
    __aicore__ inline void NoSplitProcess(int64_t curIdx);
    __aicore__ inline void SplitProcess(int64_t curIdx);
    __aicore__ inline void ComputeSplitD(int64_t curIdx);
    __aicore__ inline void ComputeSplitH(int64_t curIdx);
    __aicore__ inline void ComputeSplitW(int64_t curIdx);
    template <typename U>
    __aicore__ inline void ComputeAvg(LocalTensor<U> storeAddLocal, int64_t curIdx);

protected:
    TBuf<QuePosition::VECCALC> storeAddUB_;
};

template <typename T>
template <typename U>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::InitStoreOutBuffer()
{
    LocalTensor<U> avgStoreOutLocal = this->storeAddUB_.template Get<U>();
    __ubuf__ U* avgStoreOutAddr = (__ubuf__ U*)avgStoreOutLocal.GetPhyAddr();

    uint32_t maxOutCount = BATCH_COPYOUT_COUNT;
    uint32_t maxVfCount = platform::GetVRegSize() / sizeof(T);
    uint16_t repeatMaxTimes = ops::CeilDiv(static_cast<uint32_t>(maxOutCount), maxVfCount);

    __VEC_SCOPE__
    {
        Reg::RegTensor<U> avgStoreOutReg;
        Reg::Duplicate(avgStoreOutReg, static_cast<U>(0));
        for (uint16_t i = 0; i < repeatMaxTimes; i++) {
            Reg::MaskReg avgStoreOutMask = Reg::UpdateMask<U>(maxOutCount);
            Reg::AddrReg offsetStoreReg = Reg::CreateAddrReg<U>(i, maxVfCount);
            Reg::StoreAlign(avgStoreOutAddr, avgStoreOutReg, offsetStoreReg, avgStoreOutMask);
        }
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::InitOutputBuffer()
{
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    LocalTensor<T> avgOutLocal = this->outputUB_.template Get<T>();
    __ubuf__ T* avgOutAddr = (__ubuf__ T*)avgOutLocal.GetPhyAddr();

    uint32_t maxOutCount = BATCH_COPYOUT_COUNT;
    uint32_t maxVfCount = platform::GetVRegSize() / sizeof(T);
    uint16_t repeatMaxTimes = ops::CeilDiv(static_cast<uint32_t>(maxOutCount), maxVfCount);

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> avgOutReg;
        Reg::Duplicate(avgOutReg, static_cast<T>(0));
        for (uint16_t i = 0; i < repeatMaxTimes; i++) {
            Reg::MaskReg avgOutMask = Reg::UpdateMask<T>(maxOutCount);
            Reg::AddrReg offsetReg = Reg::CreateAddrReg<T>(i, maxVfCount);
            Reg::StoreAlign(avgOutAddr, avgOutReg, offsetReg, avgOutMask);
        }
    }
}

template <typename T>
template <typename U>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::ComputeAvg(LocalTensor<U> storeAddLocal, int64_t curIdx)
{
    LocalTensor<T> outputLocal = this->outputUB_.template Get<T>();
    __ubuf__ U* storeLocalAddr = (__ubuf__ U*)storeAddLocal.GetPhyAddr();
    __ubuf__ T* dstLocalAddr = (__ubuf__ T*)outputLocal.GetPhyAddr();
    U divNum = static_cast<U>(this->curkDHW_);

    __VEC_SCOPE__
    {
        Reg::MaskReg pregOne = Reg::CreateMask<U, Reg::MaskPattern::VL1>();
        Reg::RegTensor<U> disiv;
        Reg::RegTensor<U> lastRes;

        Reg::Duplicate(disiv, divNum);
        PoolUtils::Compute::LoadOneValue<U>(storeLocalAddr, lastRes, pregOne, 0);
        Reg::Div(lastRes, lastRes, disiv, pregOne);

        PoolUtils::Compute::StoreOneValue<T, U>(dstLocalAddr, lastRes, pregOne, curIdx);
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::ComputeSplitD(int64_t curIdx)
{
    int64_t dFactor = this->tilingData_.maxCount / this->AlignHW;
    int64_t dLoops = ops::CeilDiv(this->curkD_, dFactor);
    int64_t dTail = this->curkD_ - (dLoops - DIGHT1) * dFactor;
    int64_t inputOffset = this->curInOffset_;
    for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
        int32_t curDFactor = dLoop == (dLoops - 1) ? dTail : dFactor;
        AdaptivePool3dBigKernel<T>::CopyIn(inputOffset, this->curkW_, this->curkH_, curDFactor);
        LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
        PoolUtils::Compute::ComputeSum<T, float, SPLIT_D != NO_SPLIT>(xLocal, this->storeAddUB_.template Get<float>(),
                                                                      this->AlignHW * curDFactor);
        inputOffset += curDFactor * this->inHW_;
        this->inputQue_.template FreeTensor<T>(xLocal);
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::ComputeSplitH(int64_t curIdx)
{
    int64_t hFactor = this->tilingData_.maxCount / this->AlignW;
    int64_t hLoops = ops::CeilDiv(this->curkH_, hFactor);
    int64_t hTail = this->curkH_ - (hLoops - DIGHT1) * hFactor;
    for (int64_t dLoop = 0; dLoop < this->curkD_; dLoop++) {
        int64_t inputOffset = this->curInOffset_ + dLoop * this->inHW_;
        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            int64_t curHFactor = hLoop == (hLoops - 1) ? hTail : hFactor;
            AdaptivePool3dBigKernel<T>::CopyIn(inputOffset, this->curkW_, curHFactor, DIGHT1);
            LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
            PoolUtils::Compute::ComputeSum<T, float, SPLIT_H != NO_SPLIT>(
                xLocal, this->storeAddUB_.template Get<float>(), this->AlignW * curHFactor);
            inputOffset += hFactor * this->tilingData_.wInDim;
            this->inputQue_.template FreeTensor<T>(xLocal);
        }
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::ComputeSplitW(int64_t curIdx)
{
    int64_t wFactor = this->tilingData_.maxCount;
    int64_t wLoops = ops::CeilDiv(this->curkW_, wFactor);
    int64_t wTail = this->curkW_ - (wLoops - DIGHT1) * wFactor;
    for (int64_t dLoop = 0; dLoop < this->curkD_; dLoop++) {
        int64_t dOffset = this->curInOffset_ + dLoop * this->inHW_;
        for (int64_t hLoop = 0; hLoop < this->curkH_; hLoop++) {
            int64_t inputOffset = dOffset + hLoop * this->tilingData_.wInDim;
            for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
                int64_t curWFactor = wLoop == (wLoops - 1) ? wTail : wFactor;
                AdaptivePool3dBigKernel<T>::CopyIn(inputOffset, curWFactor, DIGHT1, DIGHT1);
                LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
                PoolUtils::Compute::ComputeSum<T, float, SPLIT_W != NO_SPLIT>(
                    xLocal, this->storeAddUB_.template Get<float>(), curWFactor);
                inputOffset += curWFactor;
                this->inputQue_.template FreeTensor<T>(xLocal);
            }
        }
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::NoSplitProcess(int64_t curIdx)
{
    AdaptivePool3dBigKernel<T>::CopyIn(this->curInOffset_, this->curkW_, this->curkH_, this->curkD_);
    LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
    PoolUtils::Compute::ComputeSum<T, float, NO_SPLIT != NO_SPLIT>(xLocal, this->storeAddUB_.template Get<float>(),
                                                                   this->AlignDHW);
    this->inputQue_.template FreeTensor<T>(xLocal);
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::SplitProcess(int64_t curIdx)
{
    InitStoreOutBuffer<float>();
    if (this->AlignHW <= this->tilingData_.maxCount) {
        ComputeSplitD(curIdx);
    } else if (this->AlignW <= this->tilingData_.maxCount) {
        ComputeSplitH(curIdx);
    } else {
        ComputeSplitW(curIdx);
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::BaseCompute(int64_t curIdx)
{
    LocalTensor<float> storeAddLocal = this->storeAddUB_.template Get<float>();
    if (this->AlignDHW <= this->tilingData_.maxCount) {
        NoSplitProcess(curIdx);
    } else {
        SplitProcess(curIdx);
    }
    ComputeAvg<float>(storeAddLocal, curIdx);
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::Init(GM_ADDR x, GM_ADDR y)
{
    // AdaptivePool3dBigKernel init
    AdaptivePool3dBigKernel<T>::Init(x, y);
    this->pipe_.InitBuffer(storeAddUB_, STORE_ADD_BUFFER);
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::Process()
{
    int64_t beginIdx = 0;
    int64_t endIdx = 0;
    if (GetBlockIdx() < this->tilingData_.blockTail) {
        beginIdx = GetBlockIdx() * (this->tilingData_.blockFactor + 1);
        endIdx = beginIdx + this->tilingData_.blockFactor + 1;
    } else {
        beginIdx = GetBlockIdx() * this->tilingData_.blockFactor + this->tilingData_.blockTail;
        endIdx = beginIdx + this->tilingData_.blockFactor;
    }

    InitOutputBuffer();
    InitStoreOutBuffer<float>();
    int64_t curLocalIdx = 0;
    int64_t outputOffset = beginIdx;
    for (int64_t outIdx = beginIdx; outIdx < endIdx; outIdx++) {
        AdaptivePool3dBigKernel<T>::CalcWindowSize(outIdx);
        BaseCompute(curLocalIdx);
        curLocalIdx++;
        if (curLocalIdx == BATCH_COPYOUT_COUNT) {
            PoolUtils::DataMove::CopyOut<T>(this->outputUB_, this->yGm_, curLocalIdx, outputOffset);
            InitOutputBuffer();
            outputOffset = outIdx + 1;
            curLocalIdx = 0;
        }
    }
    if (curLocalIdx != 0) {
        PoolUtils::DataMove::CopyOut<T>(this->outputUB_, this->yGm_, curLocalIdx, outputOffset);
    }
}
} // namespace AdaptivePool3d
#endif // ADAPTIVE_AVG_POOL3D_BIG_KERNEL_H
