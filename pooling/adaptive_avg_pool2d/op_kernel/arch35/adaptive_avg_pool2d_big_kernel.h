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
 * \file adaptive_avg_pool2d_big_kernel.h
 * \brief
 */
#ifndef ADAPTIVE_AVG_POOL2D_BIG_KERNEL_H_
#define ADAPTIVE_AVG_POOL2D_BIG_KERNEL_H_

#include "adaptive_pool2d_big_kernel.h"
#include "pool_utils/arch35/compute/adaptive_avg_pool_big_kernel_compute.h"

namespace AdaptiveAvgPool2dOp {
using namespace AscendC;
using namespace AdaptivePool2dOp;
static constexpr int64_t STORE_ADD_BUFFER = 1024;

template <typename T, uint64_t COPY_MODE>
__aicore__ inline void PadZeroToLocalMem(const __ubuf__ void* dstAddr, uint32_t padNum, uint32_t offset, T padValue)
{
    Reg::RegTensor<T> vReg;
    Reg::UnalignRegForStore uReg;
    Reg::Duplicate(vReg, padValue);
    auto addr = (__ubuf__ T*)dstAddr + offset;
    Reg::StoreUnAlign(addr, vReg, uReg, padNum);
    Reg::StoreUnAlignPost(addr, uReg, 0);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T, uint64_t COPY_MODE>
class AdaptiveAvgPool2dBigKernel : public AdaptivePool2dBigKernel<T> {
public:
    __aicore__ inline AdaptiveAvgPool2dBigKernel(const AdaptivePool2dBigKernelTilingData& tilingData, TPipe& pipe)
        : AdaptivePool2dBigKernel<T>(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOutputBuffer();
    template <typename U>
    __aicore__ inline void InitStoreOutBuffer();
    __aicore__ inline void BaseCompute(int64_t curIdx);
    __aicore__ inline void NoSplitProcess(int64_t curIdx);
    __aicore__ inline void SplitProcess(int64_t curIdx);
    __aicore__ inline void ComputeSplitH(int64_t curIdx);
    __aicore__ inline void ComputeSplitW(int64_t curIdx);
    template <typename U>
    __aicore__ inline void ComputeAvg(LocalTensor<U> storeAddLocal, int64_t curIdx);
    __aicore__ inline int64_t GetCalW();
    __aicore__ inline int64_t GetCalHW();

protected:
    TBuf<QuePosition::VECCALC> storeAddUB_;
};

template <typename T, uint64_t COPY_MODE>
template <typename U>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::InitStoreOutBuffer()
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

template <typename T, uint64_t COPY_MODE>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::InitOutputBuffer()
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

template <typename T, uint64_t COPY_MODE>
template <typename U>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::ComputeAvg(LocalTensor<U> storeAddLocal,
                                                                            int64_t curIdx)
{
    LocalTensor<T> outputLocal = this->outputUB_.template Get<T>();
    __ubuf__ U* storeLocalAddr = (__ubuf__ U*)storeAddLocal.GetPhyAddr();
    __ubuf__ T* dstLocalAddr = (__ubuf__ T*)outputLocal.GetPhyAddr();
    U divNum_ = static_cast<U>(this->curkHW_);

    __VEC_SCOPE__
    {
        Reg::MaskReg pregOne = Reg::CreateMask<U, Reg::MaskPattern::VL1>();
        Reg::RegTensor<U> disiv;
        Reg::RegTensor<U> lastRes;

        Reg::Duplicate(disiv, divNum_);
        PoolUtils::Compute::LoadOneValue<U>(storeLocalAddr, lastRes, pregOne, 0);
        Reg::Div(lastRes, lastRes, disiv, pregOne);

        PoolUtils::Compute::StoreOneValue<T, U>(dstLocalAddr, lastRes, pregOne, curIdx);
    }
}

template <typename T, uint64_t COPY_MODE>
__aicore__ inline int64_t AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::GetCalW()
{
    if constexpr (COPY_MODE == TPL_BIG_KERNEL_NDDMA) {
        return this->curkW_;
    } else {
        return this->alignW_;
    }
}

template <typename T, uint64_t COPY_MODE>
__aicore__ inline int64_t AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::GetCalHW()
{
    if constexpr (COPY_MODE == TPL_BIG_KERNEL_NDDMA) {
        return this->curkHW_;
    } else {
        return this->alignHW_;
    }
}

template <typename T, uint64_t COPY_MODE>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::ComputeSplitH(int64_t curIdx)
{
    int64_t hFactor = this->tilingData_.maxCount / GetCalW();
    int64_t hLoops = ops::CeilDiv(this->curkH_, hFactor);
    int64_t hTail = this->curkH_ - (hLoops - DIGHT1) * hFactor;
    int64_t inputOffset = this->curInOffset_;
    for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
        int64_t curHFactor = hLoop == (hLoops - 1) ? hTail : hFactor;
        if constexpr (COPY_MODE == TPL_BIG_KERNEL_NDDMA) {
            AdaptivePool2dBigKernel<T>::UnAlignCopyIn(inputOffset, this->curkW_, curHFactor);
        } else {
            AdaptivePool2dBigKernel<T>::CopyIn(inputOffset, this->curkW_, curHFactor);
        }
        LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
        PoolUtils::Compute::ComputeSum<T, float, SPLIT_H != NO_SPLIT>(xLocal, this->storeAddUB_.template Get<float>(),
                                                                      GetCalW() * curHFactor);
        inputOffset += hFactor * this->tilingData_.wInDim;
        this->inputQue_.template FreeTensor<T>(xLocal);
    }
}

template <typename T, uint64_t COPY_MODE>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::ComputeSplitW(int64_t curIdx)
{
    int64_t wFactor = this->tilingData_.maxCount;
    int64_t wLoops = ops::CeilDiv(this->curkW_, wFactor);
    int64_t wTail = this->curkW_ - (wLoops - DIGHT1) * wFactor;
    int64_t dOffset = this->curInOffset_;
    for (int64_t hLoop = 0; hLoop < this->curkH_; hLoop++) {
        int64_t inputOffset = dOffset + hLoop * this->tilingData_.wInDim;
        for (int64_t wLoop = 0; wLoop < wLoops; wLoop++) {
            int64_t curWFactor = wLoop == (wLoops - 1) ? wTail : wFactor;
            if constexpr (COPY_MODE == TPL_BIG_KERNEL_NDDMA) {
                AdaptivePool2dBigKernel<T>::UnAlignCopyIn(inputOffset, curWFactor, DIGHT1);
            } else {
                AdaptivePool2dBigKernel<T>::CopyIn(inputOffset, curWFactor, DIGHT1);
            }
            LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
            PoolUtils::Compute::ComputeSum<T, float, SPLIT_W != NO_SPLIT>(
                xLocal, this->storeAddUB_.template Get<float>(), curWFactor);
            inputOffset += curWFactor;
            this->inputQue_.template FreeTensor<T>(xLocal);
        }
    }
}

template <typename T, uint64_t COPY_MODE>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::NoSplitProcess(int64_t curIdx)
{
    if constexpr (COPY_MODE == TPL_BIG_KERNEL_NDDMA) {
        AdaptivePool2dBigKernel<T>::UnAlignCopyIn(this->curInOffset_, this->curkW_, this->curkH_);
    } else {
        AdaptivePool2dBigKernel<T>::CopyIn(this->curInOffset_, this->curkW_, this->curkH_);
    }
    LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
    PoolUtils::Compute::ComputeSum<T, float, NO_SPLIT != NO_SPLIT>(xLocal, this->storeAddUB_.template Get<float>(),
                                                                   GetCalHW());
    this->inputQue_.template FreeTensor<T>(xLocal);
}

template <typename T, uint64_t COPY_MODE>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::SplitProcess(int64_t curIdx)
{
    InitStoreOutBuffer<float>();
    if (GetCalW() <= this->tilingData_.maxCount) {
        ComputeSplitH(curIdx);
    } else {
        ComputeSplitW(curIdx);
    }
}

template <typename T, uint64_t COPY_MODE>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::BaseCompute(int64_t curIdx)
{
    LocalTensor<float> storeAddLocal = this->storeAddUB_.template Get<float>();
    if (GetCalHW() <= this->tilingData_.maxCount) {
        NoSplitProcess(curIdx);
    } else {
        SplitProcess(curIdx);
    }
    ComputeAvg<float>(storeAddLocal, curIdx);
}

template <typename T, uint64_t COPY_MODE>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::Init(GM_ADDR x, GM_ADDR y)
{
    // AdaptivePool2dBigKernel init
    AdaptivePool2dBigKernel<T>::Init(x, y);
    this->pipe_.InitBuffer(storeAddUB_, STORE_ADD_BUFFER);
}

template <typename T, uint64_t COPY_MODE>
__aicore__ inline void AdaptiveAvgPool2dBigKernel<T, COPY_MODE>::Process()
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
        AdaptivePool2dBigKernel<T>::CalcWindowSize(outIdx);
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
} // namespace AdaptiveAvgPool2dOp
#endif // ADAPTIVE_AVG_POOL2D_BIG_KERNEL_H
