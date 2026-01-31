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
 * \file adaptive_max_pool3d_big_kernel.h
 * \brief
 */
#ifndef ADAPTIVE_MAX_POOL3D_BIG_KERNEL_H_
#define ADAPTIVE_MAX_POOL3D_BIG_KERNEL_H_

#include "adaptive_pool3d_big_kernel.h"

namespace AdaptivePool3d{
using namespace AscendC;

template <typename T, typename U>
__aicore__ inline void StoreOneValue(const __local_mem__ void* dstAddr, MicroAPI::RegTensor<U>& srcReg,
                                       MicroAPI::MaskReg& maskReg, uint32_t offset)
{
    auto addr = (__local_mem__ T*)dstAddr + offset;
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(addr, srcReg, maskReg);
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

template <typename T, typename U>
__aicore__ inline void LoadOneValue(const __local_mem__ void* srcAddr, MicroAPI::RegTensor<U>& dstReg,
                                      MicroAPI::MaskReg& preg, uint32_t offset)
{
    auto addr = (__local_mem__ T*)srcAddr + offset;
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> regBf16;
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(regBf16, addr);
        MicroAPI::Cast<float, bfloat16_t, CASTB2TOB4>(dstReg, regBf16, preg);
    } else if constexpr (IsSameType<T, half>::value) {
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(dstReg, addr);
    } else if constexpr (sizeof(T) == DIGHT4) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(dstReg, addr);
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
        MicroAPI::DataCopy(dstReg, (__local_mem__ half*)srcAddr, offset);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> regBf16;
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(regBf16, (__local_mem__ bfloat16_t*)srcAddr, offset);
        MicroAPI::Cast<float, bfloat16_t, CASTB2TOB4>(dstReg, regBf16, preg);
    } else {
        MicroAPI::DataCopy(dstReg, (__local_mem__ float*)srcAddr, offset);
    }
}

template <typename T>
__aicore__ inline void PadInfToLocalMem(const __local_mem__ void* dstAddr, uint32_t padNum, uint32_t offset, T padValue)
{
    MicroAPI::RegTensor<T> vReg;
    MicroAPI::UnalignReg uReg;
    MicroAPI::Duplicate(vReg, padValue);
    auto addr = (__local_mem__ T*)dstAddr + offset;
    MicroAPI::DataCopyUnAlign(addr, vReg, uReg, padNum);
    MicroAPI::DataCopyUnAlignPost(addr, uReg, 0);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename U, typename UINDEX>
__aicore__ inline void CalcMaxAndIndex(MicroAPI::RegTensor<U>& dstMax, MicroAPI::RegTensor<UINDEX>& dstIndex,
    MicroAPI::RegTensor<U>& srcMax, MicroAPI::RegTensor<UINDEX>& srcIndex, UINDEX idxDefaultValue)
{
    // select first max value or last nan from one reg
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRegNotNan;
    MicroAPI::MaskReg maskRegNan;
    MicroAPI::RegTensor<U> vRegMaxNotNan;
    MicroAPI::RegTensor<U> vRegMaxNotNanAll;
    MicroAPI::RegTensor<UINDEX> uIndexReg;
    // calc last nan index
    MicroAPI::Duplicate(uIndexReg, idxDefaultValue);
    MicroAPI::Compare<U, CMPMODE::NE>(maskRegNan, srcMax, srcMax, maskAll); // nan mask
    MicroAPI::Select(uIndexReg, srcIndex, uIndexReg, maskRegNan);
    MicroAPI::ReduceMax(uIndexReg, uIndexReg, maskAll);

    // calc not nan max value and index
    MicroAPI::Compare<U, CMPMODE::EQ>(maskRegNotNan, srcMax, srcMax, maskAll);
    MicroAPI::ReduceMax(vRegMaxNotNan, srcMax, maskRegNotNan); 
    MicroAPI::Duplicate(vRegMaxNotNanAll, vRegMaxNotNan, maskAll);
    MicroAPI::Compare<U, CMPMODE::EQ>(maskRegNotNan, srcMax, vRegMaxNotNanAll, maskAll);
    MicroAPI::ReduceMin(dstIndex, srcIndex, maskRegNotNan);

    // compare nanIndex and -1 to get mask, select index
    MicroAPI::CompareScalar<UINDEX, CMPMODE::NE>(maskRegNan, uIndexReg, idxDefaultValue, maskAll);
    MicroAPI::Select(dstIndex, uIndexReg, dstIndex, maskRegNan);
    MicroAPI::ReduceMax(dstMax, srcMax, maskAll);

    // all value in the kernel is -inf
    MicroAPI::CompareScalar<UINDEX, CMPMODE::EQ>(maskRegNotNan, dstIndex, idxDefaultValue, maskAll);
    MicroAPI::Duplicate(uIndexReg, static_cast<UINDEX>(0));
    MicroAPI::Select(dstIndex, uIndexReg, dstIndex, maskRegNotNan);
}

template <typename TINDEX>
__aicore__ inline void CalcDivAndMod(MicroAPI::RegTensor<TINDEX>& totalReg, MicroAPI::RegTensor<TINDEX>& divisorReg,
    MicroAPI::RegTensor<TINDEX>& divReg, MicroAPI::RegTensor<TINDEX>& modReg)
{
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<TINDEX, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<TINDEX> tmpReg;

    MicroAPI::Div(divReg, totalReg, divisorReg, maskOne);
    MicroAPI::Mul(tmpReg, divReg, divisorReg, maskOne);
    MicroAPI::Sub(modReg, totalReg, tmpReg, maskOne);
}

template <typename TINDEX, typename UINDEX, int32_t SPLIT_MODE>
__aicore__ inline void CalcRealIndex(MicroAPI::RegTensor<TINDEX>& resIndex, MicroAPI::RegTensor<UINDEX>& index,
    TINDEX wInDim, TINDEX inHW, TINDEX curkW, TINDEX curkHW, TINDEX curOffset)
{
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<UINDEX, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<int32_t> regTmp;

    // calc index type use int16, need change to int32
    if constexpr (IsSameType<UINDEX, int16_t>::value) {
        MicroAPI::Cast<int32_t, int16_t, CASTB2TOB4>(regTmp, index, maskOne);
    } else {
        MicroAPI::Move(regTmp, index, maskOne);
    }

    // output index type use int64, need change to int64
    if constexpr (IsSameType<TINDEX, int64_t>::value) {
        MicroAPI::Cast<int64_t, int32_t, CASTB4TOB8>(resIndex, regTmp, maskOne);
    } else {
        MicroAPI::Move(resIndex, regTmp, maskOne);
    }

    if constexpr (SPLIT_MODE == SPLIT_W) {
        MicroAPI::Adds(resIndex, resIndex, curOffset, maskOne);
    } else if constexpr (SPLIT_MODE == SPLIT_H) {
        MicroAPI::RegTensor<TINDEX> regOffset;
        MicroAPI::RegTensor<TINDEX> regCurkW;
        MicroAPI::RegTensor<TINDEX> regHOffset;
        MicroAPI::Duplicate(regOffset, curOffset, maskOne);
        MicroAPI::Duplicate(regCurkW, curkW, maskOne);
        CalcDivAndMod<TINDEX>(resIndex, regCurkW, regHOffset, resIndex);
        MicroAPI::Muls(regHOffset, regHOffset, wInDim, maskOne);
        MicroAPI::Add(resIndex, resIndex, regHOffset, maskOne);
        MicroAPI::Add(resIndex, resIndex, regOffset, maskOne);
    } else {
        MicroAPI::RegTensor<TINDEX> regOffset;
        MicroAPI::RegTensor<TINDEX> regCurkW;
        MicroAPI::RegTensor<TINDEX> regCurkHW;
        MicroAPI::RegTensor<TINDEX> regDOffset;
        MicroAPI::RegTensor<TINDEX> regHOffset;
        MicroAPI::Duplicate(regOffset, curOffset, maskOne);
        MicroAPI::Duplicate(regCurkHW, curkHW, maskOne);
        MicroAPI::Duplicate(regCurkW, curkW, maskOne);
        CalcDivAndMod<TINDEX>(resIndex, regCurkHW, regDOffset, resIndex);
        CalcDivAndMod<TINDEX>(resIndex, regCurkW, regHOffset, resIndex);
        MicroAPI::Muls(regDOffset, regDOffset, inHW, maskOne);
        MicroAPI::Muls(regHOffset, regHOffset, wInDim, maskOne);
        MicroAPI::Add(resIndex, resIndex, regDOffset, maskOne);
        MicroAPI::Add(resIndex, resIndex, regHOffset, maskOne);
        MicroAPI::Add(resIndex, resIndex, regOffset, maskOne);
    }
}

template <typename T1, typename T2, typename TINDEX>
__aicore__ inline void UpdateMaxAndIndex(MicroAPI::RegTensor<T2>& res, MicroAPI::RegTensor<TINDEX>& realResIndex,
    const __local_mem__ T1* dstLocalAddr, const __local_mem__ TINDEX* indexLocalAddr, int32_t offset, int32_t isPadValue)
{
    // get data from local mem
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg notNanMaskReg;
    MicroAPI::MaskReg nanMaskReg;
    MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<TINDEX, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<T2> lastRes;
    MicroAPI::RegTensor<TINDEX> lastResIndex;

    // get last res from local mem
    LoadOneValue<T1, T2>(dstLocalAddr, lastRes, pregOne, offset);
    LoadOneValue<TINDEX, TINDEX>(indexLocalAddr, lastResIndex, maskOne, offset);

    // calc last max value and mask
    MicroAPI::Compare<T2, CMPMODE::NE>(nanMaskReg, res, res, maskAll);
    MicroAPI::Compare<T2, CMPMODE::GT>(notNanMaskReg, res, lastRes, maskAll);
    MicroAPI::MaskXor(notNanMaskReg, notNanMaskReg, nanMaskReg, maskAll);
    MicroAPI::Select(res, res, lastRes, notNanMaskReg);

    // calc last index
    MicroAPI::CompareScalar<TINDEX, CMPMODE::EQ>(nanMaskReg, lastResIndex, isPadValue, maskAll);
    MicroAPI::Select(lastResIndex, realResIndex, lastResIndex, nanMaskReg);
    MicroAPI::Select(realResIndex, realResIndex, lastResIndex, notNanMaskReg); 
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

template <typename T, typename TINDEX>
class AdaptiveMaxPool3dBigKernel : public AdaptivePool3dBigKernel<T>
{
public:
    __aicore__ inline AdaptiveMaxPool3dBigKernel(const AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData &tilingData, TPipe &pipe) :
        AdaptivePool3dBigKernel<T>(tilingData, pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOutputBuffer();
    __aicore__ inline void BaseCompute(int64_t curIdx);
    __aicore__ inline void NoSplitProcess(int64_t curIdx);
    __aicore__ inline void SplitProcess(int64_t curIdx);
    __aicore__ inline void ComputeSplitD(int64_t curIdx);
    __aicore__ inline void ComputeSplitH(int64_t curIdx);
    __aicore__ inline void ComputeSplitW(int64_t curIdx);
    template <int32_t SPLIT_MODE, typename U, typename UINDEX>
    __aicore__ inline void ComputeMax(LocalTensor<T> xLocal, int64_t localCurIdx, int64_t dataCount,
                                      int64_t curOffset);
    __aicore__ inline void CopyOutIndices(int64_t copyCount, int64_t offset);

    // indices need
    TBuf<> indicesUB_;
    GlobalTensor<TINDEX> indicesGm_;
};

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::InitOutputBuffer()
{
    event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
    LocalTensor<T> maxOutLocal = this->maxUB_.template Get<T>();
    __local_mem__ T* maxOutAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
    LocalTensor<TINDEX> idxOutLocal = indicesUB_.Get<TINDEX>();
    __local_mem__ TINDEX* idxOutAddr = (__local_mem__ TINDEX*)idxOutLocal.GetPhyAddr();

    uint32_t maxOutCount = BATCH_COPYOUT_COUNT;
    uint32_t maxVfCount = platform::GetVRegSize() / sizeof(T);
    uint16_t repeatMaxTimes = ops::CeilDiv(static_cast<uint32_t>(maxOutCount), maxVfCount);
    uint32_t idxVfCount = platform::GetVRegSize() / sizeof(TINDEX);
    uint32_t idxOutCount = BATCH_COPYOUT_COUNT;
    uint16_t repeatIdxTimes = ops::CeilDiv(static_cast<uint32_t>(maxOutCount), maxVfCount);

    T maxDefaultValue = this->minValue_;
    TINDEX idxDefaultValue = -1;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> maxReg;
        MicroAPI::Duplicate(maxReg, maxDefaultValue);
        for (uint16_t i = 0; i < repeatMaxTimes; i++) {
            MicroAPI::MaskReg maxMask = MicroAPI::UpdateMask<T>(maxOutCount);
            MicroAPI::AddrReg offsetReg = MicroAPI::CreateAddrReg<T>(i, maxVfCount);
            MicroAPI::DataCopy(maxOutAddr, maxReg, offsetReg, maxMask);
        }
        MicroAPI::RegTensor<TINDEX> idxReg;
        MicroAPI::Duplicate(idxReg, idxDefaultValue);
        for (uint16_t i = 0; i < repeatIdxTimes; i++) {
            MicroAPI::MaskReg idxMask = MicroAPI::UpdateMask<TINDEX>(idxOutCount);
            MicroAPI::AddrReg offsetReg = MicroAPI::CreateAddrReg<TINDEX>(i, idxVfCount);
            MicroAPI::DataCopy(idxOutAddr, idxReg, offsetReg, idxMask);
        }
    }
}

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::CopyOutIndices(int64_t copyCount, int64_t offset)
{
    event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    LocalTensor<TINDEX> indicesLocal = indicesUB_.Get<TINDEX>();
    DataCopyExtParams indicesExtParams;
    indicesExtParams.blockCount = DIGHT1;
    indicesExtParams.blockLen = copyCount * sizeof(TINDEX);
    indicesExtParams.srcStride = 0;
    indicesExtParams.dstStride = 0;
    DataCopyPad(indicesGm_[offset], indicesLocal, indicesExtParams);
}

template <typename T, typename TINDEX>
template <int32_t SPLIT_MODE, typename U, typename UINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::ComputeMax(
    LocalTensor<T> xLocal, int64_t localCurIdx, int64_t dataCount, int64_t curOffset)
{
    LocalTensor<T> maxLocal = this->maxUB_.template Get<T>();
    LocalTensor<TINDEX> indicesLocal = indicesUB_.Get<TINDEX>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* dstLocalAddr = (__local_mem__ T*)maxLocal.GetPhyAddr();
    __local_mem__ TINDEX* indicesLocalAddr = (__local_mem__ TINDEX*)indicesLocal.GetPhyAddr();
    
    T minValue = this->minValue_;
    U uMinValue = AdaptivePool3dBigKernel<T>::template GetDtypeMinValue<U>();
    UINDEX idxDefaultValue = -1;
    uint32_t repeatCount = platform::GetVRegSize() / sizeof(U);
    uint16_t repeatTimes = ops::CeilDiv(static_cast<uint32_t>(dataCount), repeatCount);
    uint32_t totalNum = repeatTimes * repeatCount;
    uint32_t fillNum = totalNum - dataCount;
    curOffset = curOffset - this->curNc_ * this->inDHW_;
    TINDEX inHW = this->inHW_;
    TINDEX wInDim = this->tilingData_.wInDim;
    TINDEX curkHW = this->curkHW_;
    TINDEX curkW = this->curkW_;

    __VEC_SCOPE__
    {
        PadInfToLocalMem<T>(xLocalAddr, fillNum, dataCount, minValue);
        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> res;
        MicroAPI::RegTensor<UINDEX> resIndex;
        MicroAPI::RegTensor<UINDEX> dstIndex;
        MicroAPI::RegTensor<TINDEX> realResIndex;
        MicroAPI::MaskReg cmpMaskNanReg;
        MicroAPI::MaskReg cmpMaskReg;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegMax = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg maskRegIdx = MicroAPI::CreateMask<UINDEX, MicroAPI::MaskPattern::VL1>();  

        MicroAPI::Duplicate(resIndex, idxDefaultValue);
        MicroAPI::Duplicate(res, uMinValue);
        MicroAPI::Arange(dstIndex, 0);
        // copyin xLocal to reg, calc max reg and index
        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(totalNum);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatCount);
            LoadXLocalToReg<T, U>(xLocalAddr, vd0, p0, offset);
            MicroAPI::Compare<U, CMPMODE::NE>(cmpMaskNanReg, vd0, vd0, maskRegAll);
            MicroAPI::Compare<U, CMPMODE::GT>(cmpMaskReg, vd0, res, maskRegAll);
            MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg, cmpMaskNanReg, maskRegAll);
            MicroAPI::Select(res, vd0, res, cmpMaskReg);
            MicroAPI::Select(resIndex, dstIndex, resIndex, cmpMaskReg);
            MicroAPI::Adds(dstIndex, dstIndex, repeatCount, maskRegAll);
        }
        // calc max value and temp index
        CalcMaxAndIndex<U, UINDEX>(res, dstIndex, res, resIndex, idxDefaultValue);
        // calc real indexa
        CalcRealIndex<TINDEX, UINDEX, SPLIT_MODE>(realResIndex, dstIndex, wInDim, inHW, curkW, curkHW, curOffset);
        // no split need compare maxUB value and current value
        if constexpr (SPLIT_MODE != NO_SPLIT) {
            UpdateMaxAndIndex<T, U, TINDEX>(res, realResIndex, dstLocalAddr, indicesLocalAddr, localCurIdx, idxDefaultValue);
        }

        StoreOneValue<T, U>(dstLocalAddr, res, maskRegMax, localCurIdx);
        StoreOneValue<TINDEX, TINDEX>(indicesLocalAddr, realResIndex, maskRegIdx, localCurIdx);
    }
}

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::ComputeSplitD(int64_t curIdx)
{
    int64_t dFactor = this->tilingData_.maxCount / this->curkHW_;
    int64_t dLoops = ops::CeilDiv(this->curkD_, dFactor);
    int64_t dTail = this->curkD_ - (dLoops - DIGHT1) * dFactor;
    int64_t inputOffset = this->curInOffset_;
    for (int64_t dLoop = 0; dLoop < dLoops; dLoop++) {
        int32_t curDFactor = dLoop == (dLoops - 1) ? dTail : dFactor;
        AdaptivePool3dBigKernel<T>::CopyIn(inputOffset, this->curkW_, this->curkH_, curDFactor);
        LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
        if constexpr (IsSameType<T, half>::value) {
            ComputeMax<SPLIT_D, half, int16_t>(xLocal, curIdx, this->curkHW_ * curDFactor, inputOffset);
        } else {
            ComputeMax<SPLIT_D, float, int32_t>(xLocal, curIdx, this->curkHW_ * curDFactor, inputOffset);
        }
        inputOffset += curDFactor * this->inHW_;
        this->inputQue_.template FreeTensor<T>(xLocal);
    }
}

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::ComputeSplitH(int64_t curIdx)
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
            if constexpr (IsSameType<T, half>::value) {
                ComputeMax<SPLIT_H, half, int16_t>(xLocal, curIdx, this->curkW_ * curHFactor, inputOffset);
            } else {
                ComputeMax<SPLIT_H, float, int32_t>(xLocal, curIdx, this->curkW_ * curHFactor, inputOffset);
            }
            inputOffset += hFactor * this->tilingData_.wInDim;
            this->inputQue_.template FreeTensor<T>(xLocal);
        }
    }
}

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::ComputeSplitW(int64_t curIdx)
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
                if constexpr (IsSameType<T, half>::value) {
                    ComputeMax<SPLIT_W, half, int16_t>(xLocal, curIdx, curWFactor, inputOffset);
                } else {
                    ComputeMax<SPLIT_W, float, int32_t>(xLocal, curIdx, curWFactor, inputOffset);
                }
                inputOffset += curWFactor;
                this->inputQue_.template FreeTensor<T>(xLocal);
            }
        }
    }
}

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::NoSplitProcess(int64_t curIdx)
{
    AdaptivePool3dBigKernel<T>::CopyIn(this->curInOffset_, this->curkW_, this->curkH_, this->curkD_);
    LocalTensor<T> xLocal = this->inputQue_.template DeQue<T>();
    if constexpr (IsSameType<T, half>::value) {
        ComputeMax<NO_SPLIT, half, int16_t>(xLocal, curIdx, this->curkDHW_, this->curInOffset_);
    } else {
        ComputeMax<NO_SPLIT, float, int32_t>(xLocal, curIdx, this->curkDHW_, this->curInOffset_);
    }
    this->inputQue_.template FreeTensor<T>(xLocal);
}

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::SplitProcess(int64_t curIdx)
{
    if (this->curkHW_ <= this->tilingData_.maxCount) {
        ComputeSplitD(curIdx);
    } else if (this->curkW_ <= this->tilingData_.maxCount) {
        ComputeSplitH(curIdx);
    } else {
        ComputeSplitW(curIdx);
    }
}

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::BaseCompute(int64_t curIdx)
{
    if (this->curkDHW_ <= this->tilingData_.maxCount) {
        NoSplitProcess(curIdx);
    } else {
        SplitProcess(curIdx);
    }
}

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices)
{
    // AdaptivePool3dBigKernel init
    AdaptivePool3dBigKernel<T>::Init(x, y);

    // indices init
    indicesGm_.SetGlobalBuffer((__gm__ TINDEX*)indices);
    this->pipe_.InitBuffer(indicesUB_, BATCH_COPYOUT_COUNT * sizeof(TINDEX));

    // set half overflow
    if constexpr (IsSameType<T, half>::value) {
        SetCtrlSpr<HALF_OVERFLOW_MODE_CTRL, HALF_OVERFLOW_MODE_CTRL>(1);
    }
}

template <typename T, typename TINDEX>
__aicore__ inline void AdaptiveMaxPool3dBigKernel<T, TINDEX>::Process()
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
            CopyOutIndices(curLocalIdx, outputOffset);
            InitOutputBuffer();
            outputOffset = outIdx + 1;
            curLocalIdx = 0;
        }
    }
    if (curLocalIdx != 0) {
        AdaptivePool3dBigKernel<T>::CopyOut(curLocalIdx, outputOffset);
        CopyOutIndices(curLocalIdx, outputOffset);
    }
}
} // namespace AdaptivePool3d
#endif // ADAPTIVE_MAX_POOL3D_BIG_KERNEL_H