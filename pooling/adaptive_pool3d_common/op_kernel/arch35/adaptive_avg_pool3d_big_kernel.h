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

namespace AdaptivePool3d{
using namespace AscendC;
constexpr int32_t STORE_ADD_BUFFER = 1024;

template <typename T, typename U>
__aicore__ inline void StoreOneValue(const __local_mem__ void* dstAddr, MicroAPI::RegTensor<U>& srcReg,
                                       MicroAPI::MaskReg& maskReg, uint32_t offset)
{
    auto addr = (__local_mem__ T*)dstAddr + offset;
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> regfp16;
        MicroAPI::Cast<half, float, CASTB4TOB2>(regfp16, srcReg, maskReg);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(addr, regfp16, maskReg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> regBf16;
        MicroAPI::Cast<bfloat16_t, float, CASTB4TOB2>(regBf16, srcReg, maskReg);
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(addr, regBf16, maskReg);
    } else if constexpr (sizeof(T) == DIGHT4) {
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(addr, (MicroAPI::RegTensor<T>&)srcReg, maskReg);
    } else {
        MicroAPI::UnalignReg uReg;
        MicroAPI::DataCopyUnAlign(addr, srcReg, uReg, 1);
        MicroAPI::DataCopyUnAlignPost(addr, uReg, 0);
    }
}

template <typename U>
__aicore__ inline void LoadOneValue(const __local_mem__ void* srcAddr, MicroAPI::RegTensor<U>& dstReg,
                                      MicroAPI::MaskReg& preg, uint32_t offset)
{
    auto addr = (__local_mem__ U*)srcAddr + offset;
    if constexpr (sizeof(U) == DIGHT4) {
        MicroAPI::DataCopy<U, MicroAPI::LoadDist::DIST_BRC_B32>(dstReg, addr);
    } else {
        MicroAPI::UnalignReg ureg;
        MicroAPI::DataCopyUnAlignPre(ureg, addr);
        MicroAPI::DataCopyUnAlign(dstReg, ureg, addr, 1);
    }
}

template <typename T, typename U>
__aicore__ inline void LoadXLocalToReg(const __local_mem__ void* srcAddr, MicroAPI::RegTensor<U>& dstReg,
                                     MicroAPI::MaskReg& preg, MicroAPI::AddrReg& offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> regfp16;
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(regfp16, (__local_mem__ half*)srcAddr, offset);
        MicroAPI::Cast<float, half, CASTB2TOB4>(dstReg, regfp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> regBf16;
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(regBf16, (__local_mem__ bfloat16_t*)srcAddr, offset);
        MicroAPI::Cast<float, bfloat16_t, CASTB2TOB4>(dstReg, regBf16, preg);
    } else {
        MicroAPI::DataCopy(dstReg, (__local_mem__ float*)srcAddr, offset);
    }
}

template <typename U>
__aicore__ inline void UpdateSum(MicroAPI::RegTensor<U>& res, const __local_mem__ U* storeLocalAddr, int32_t offset)
{
    // get data from local mem
    MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<U> lastRes;

    // get last res from local mem
    LoadOneValue<U>(storeLocalAddr, lastRes, pregOne, offset);

    //calc sum
    MicroAPI::Add(res, res, lastRes, pregOne);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

template <typename T>
class AdaptiveAvgPool3dBigKernel : public AdaptivePool3dBigKernel<T>
{
public:
    __aicore__ inline AdaptiveAvgPool3dBigKernel(const AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData &tilingData, TPipe &pipe) :
        AdaptivePool3dBigKernel<T>(tilingData, pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOutputBuffer();
    __aicore__ inline void BaseCompute(int64_t curIdx);
    __aicore__ inline void NoSplitProcess(int64_t curIdx);
    __aicore__ inline void SplitProcess(int64_t curIdx);
    __aicore__ inline void ComputeSplitD(int64_t curIdx);
    __aicore__ inline void ComputeSplitH(int64_t curIdx);
    __aicore__ inline void ComputeSplitW(int64_t curIdx);
    template <int32_t SPLIT_MODE, typename U>
    __aicore__ inline void ComputeSum(LocalTensor<T> xLocal, int64_t localCurIdx, int64_t dataCount);
    template <typename U>
    __aicore__ inline void ComputeAvg(LocalTensor<U> storeAddLocal, int64_t curIdx);

protected:
    TBuf<QuePosition::VECCALC> storeAddUB_;
};

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::InitOutputBuffer()
{
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    LocalTensor<T> avgOutLocal = this->outputUB_.template Get<T>();
    LocalTensor<float> avgStoreOutLocal = this->storeAddUB_.template Get<float>();
    __local_mem__ T* avgOutAddr = (__local_mem__ T*)avgOutLocal.GetPhyAddr();
    __local_mem__ float* avgStoreOutAddr = (__local_mem__ float*)avgStoreOutLocal.GetPhyAddr();

    uint32_t maxOutCount = BATCH_COPYOUT_COUNT;
    uint32_t maxVfCount = platform::GetVRegSize() / sizeof(T);
    uint16_t repeatMaxTimes = ops::CeilDiv(static_cast<uint32_t>(maxOutCount), maxVfCount);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> avgOutReg;
        MicroAPI::RegTensor<float> avgStoreOutReg;
        MicroAPI::Duplicate(avgOutReg, static_cast<T>(0));
        MicroAPI::Duplicate(avgStoreOutReg, static_cast<float>(0));
        for (uint16_t i = 0; i < repeatMaxTimes; i++) {
            MicroAPI::MaskReg avgOutMask = MicroAPI::UpdateMask<T>(maxOutCount);
            MicroAPI::MaskReg avgStoreOutMask = MicroAPI::UpdateMask<float>(maxOutCount);
            MicroAPI::AddrReg offsetReg = MicroAPI::CreateAddrReg<T>(i, maxVfCount);
            MicroAPI::AddrReg offsetStoreReg = MicroAPI::CreateAddrReg<float>(i, maxVfCount);
            MicroAPI::DataCopy(avgOutAddr, avgOutReg, offsetReg, avgOutMask);
            MicroAPI::DataCopy(avgStoreOutAddr, avgStoreOutReg, offsetStoreReg, avgStoreOutMask);
        }
    }
}

template<typename T>
template<typename U>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::ComputeAvg(LocalTensor<U> storeAddLocal, int64_t curIdx)
{
    LocalTensor<T> outputLocal = this->outputUB_.template Get<T>();
    __local_mem__ U* storeLocalAddr = (__local_mem__ U*)storeAddLocal.GetPhyAddr();
    __local_mem__ T* dstLocalAddr = (__local_mem__ T*)outputLocal.GetPhyAddr();
    U divNum = static_cast<U>(this->curkDHW_);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::VL1>();
        MicroAPI::RegTensor<U> disiv;
        MicroAPI::RegTensor<U> lastRes;

        MicroAPI::Duplicate(disiv, divNum);
        LoadOneValue<U>(storeLocalAddr, lastRes, pregOne, curIdx);
        MicroAPI::Div(lastRes, lastRes, disiv, pregOne);

        StoreOneValue<T, U>(dstLocalAddr, lastRes, pregOne, curIdx);
    }
}

template <typename T>
template <int32_t SPLIT_MODE, typename U>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::ComputeSum(
    LocalTensor<T> xLocal, int64_t localCurIdx, int64_t dataCount)
{
    LocalTensor<U> storeAddLocal = this->storeAddUB_.template Get<U>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ U* storeLocalAddr = (__local_mem__ U*)storeAddLocal.GetPhyAddr();
    
    uint32_t repeatCount = platform::GetVRegSize() / sizeof(U); //一个vf需要的次数
    uint16_t repeatTimes = ops::CeilDiv(static_cast<uint32_t>(dataCount), repeatCount); //上取整，获取repeatCount的整数倍
    uint32_t dataCount_ = dataCount;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> res;
        MicroAPI::MaskReg sumMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate(res, static_cast<U>(0));
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(dataCount_); //一次处理数量
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatCount); //搬运偏移
            LoadXLocalToReg<T, U>(xLocalAddr, vd0, p0, offset);
            MicroAPI::ReduceSum(vd1, vd0, p0);
            MicroAPI::Add(res, res, vd1, sumMask);
        }
        if constexpr (SPLIT_MODE != NO_SPLIT) {
            UpdateSum<U>(res, storeLocalAddr, localCurIdx);
        }
        StoreOneValue<U, U>(storeLocalAddr, res, sumMask, localCurIdx);
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::ComputeSplitD(int64_t curIdx)
{
    int64_t dFactor = this->tilingData_.maxCount / this->curkHW_;
    int64_t dLoops = ops::CeilDiv(this->curkD_, dFactor);
    int64_t dTail = this->curkD_ - (dLoops - DIGHT1) * dFactor;
    int64_t inputOffset = this->curInOffset_;
    for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
        int32_t curDFactor = dLoop == (dLoops - 1) ? dTail : dFactor;
        AdaptivePool3dBigKernel<T>::CopyIn(inputOffset, this->curkW_, this->curkH_, curDFactor);
        LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
        ComputeSum<SPLIT_D, float>(xLocal, curIdx, this->curkHW_ * curDFactor);
        inputOffset += curDFactor * this->inHW_;
        this->inputQue_.template FreeTensor<T>(xLocal);
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::ComputeSplitH(int64_t curIdx)
{
    int64_t hFactor = this->tilingData_.maxCount / this->curkW_;
    int64_t hLoops = ops::CeilDiv(this->curkH_, hFactor);
    int64_t hTail = this->curkH_ - (hLoops - DIGHT1) * hFactor;
    for (int64_t dLoop = 0; dLoop < this->curkD_; dLoop++) {
        int64_t inputOffset = this->curInOffset_ + dLoop * this->inHW_;
        for (int64_t hLoop = 0; hLoop < hLoops; hLoop++) {
            int64_t curHFactor = hLoop == (hLoops - 1) ? hTail : hFactor;
            AdaptivePool3dBigKernel<T>::CopyIn(inputOffset, this->curkW_, curHFactor, DIGHT1);
            LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
            ComputeSum<SPLIT_H, float>(xLocal, curIdx, this->curkW_ * curHFactor);
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
                ComputeSum<SPLIT_W, float>(xLocal, curIdx, curWFactor);
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
    ComputeSum<NO_SPLIT, float>(xLocal, curIdx, this->curkDHW_);
    this->inputQue_.template FreeTensor<T>(xLocal);
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::SplitProcess(int64_t curIdx)
{
    if (this->curkHW_ <= this->tilingData_.maxCount) {
        ComputeSplitD(curIdx);
    } else if (this->curkW_ <= this->tilingData_.maxCount) {
        ComputeSplitH(curIdx);
    } else {
        ComputeSplitW(curIdx);
    }
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dBigKernel<T>::BaseCompute(int64_t curIdx)
{
    LocalTensor<float> storeAddLocal = this->storeAddUB_.template DeQue<float>();
    if (this->curkDHW_ <= this->tilingData_.maxCount) {
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
    // set half overflow
    if constexpr (IsSameType<T, half>::value) {
        SetCtrlSpr<HALF_OVERFLOW_MODE_CTRL, HALF_OVERFLOW_MODE_CTRL>(1);
    }
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
    int64_t curLocalIdx = 0;
    int64_t outputOffset = beginIdx;
    for (int64_t outIdx = beginIdx; outIdx < endIdx; outIdx++) {
        AdaptivePool3dBigKernel<T>::CalcWindowSize(outIdx);
        BaseCompute(curLocalIdx);
        curLocalIdx++;
        if (curLocalIdx == BATCH_COPYOUT_COUNT) {
            AdaptivePool3dBigKernel<T>::CopyOut(curLocalIdx, outputOffset);
            InitOutputBuffer();
            outputOffset = outIdx + 1;
            curLocalIdx = 0;
        }
    }
    if (curLocalIdx != 0) {
        AdaptivePool3dBigKernel<T>::CopyOut(curLocalIdx, outputOffset);
    }
}
} // namespace AdaptivePool3d
#endif // ADAPTIVE_AVG_POOL3D_BIG_KERNEL_H