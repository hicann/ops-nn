/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file non_zero_full_load_base.h
 * \brief
 */
#ifndef CANN_NON_ZERO_FULL_LOAD_BASE_H
#define CANN_NON_ZERO_FULL_LOAD_BASE_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"

namespace NonZero {
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MaskUnPack;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnPack;
using AscendC::MicroAPI::UpdateMask;

template <typename T1, typename T2, int TILING_KEY>
class NonZeroFullLoadBase {
public:
    __aicore__ inline NonZeroFullLoadBase(){};

protected:
    const NonZeroTilingData* tilingData_;
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueX_;
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> yGm_;
    GlobalTensor<uint64_t> shapeGm_;
    TBuf<QuePosition::VECCALC> addUb_;
    TBuf<QuePosition::VECCALC> maskUb_;
    DataCopyExtParams copyParams_;

    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    uint64_t allNum_ = 0;
    uint32_t vfLenT1_ = GetVRegSize() / sizeof(T1);

    constexpr static AscendC::MicroAPI::CastTrait castTraitB322T2 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR y, GM_ADDR outShape, const NonZeroTilingData* tilingData);
    __aicore__ inline void BaseProcess();
    __aicore__ inline void VfCleanUb(LocalTensor<uint32_t>& addUbSize);
    __aicore__ inline void CopyInBefore(
        int32_t loopNum, int32_t beforeNum, int32_t tailNum, LocalTensor<uint32_t>& addUbSize,
        LocalTensor<uint32_t>& maskUbSize);
    __aicore__ inline void VfReduceSum(LocalTensor<uint32_t>& addUbSize);
    __aicore__ inline void ComputeNonZeroNum(LocalTensor<uint32_t>& addUbSize);
    __aicore__ inline void ComputeBefore(
        uint64_t& loopGm, int32_t loopCore, int32_t beforeNumOut, LocalTensor<uint32_t>& maskUbSize);
    __aicore__ inline void VfComputeIds(
        int32_t loopIndex, int32_t loopNum, LocalTensor<int32_t>& outUbSize, LocalTensor<uint32_t>& maskUbSize,
        uint64_t& arNum);
    __aicore__ inline void VfComputeIdsB64(int32_t loopIndex, int32_t loopNum, LocalTensor<int32_t> &outUbSize,
        LocalTensor<uint32_t> &maskUbSize, uint64_t &arNum);
    __aicore__ inline void VfPerCoreNonZeroNum1(
        int32_t loop, int32_t computeNum, LocalTensor<T1>& xUbSize, LocalTensor<uint32_t>& maskUbSize,
        LocalTensor<uint32_t>& addUbSize);
    __aicore__ inline void VfPerCoreNonZeroNum2(
        int32_t loop, int32_t computeNum, LocalTensor<T1>& xUbSize, LocalTensor<uint32_t>& maskUbSize,
        LocalTensor<uint32_t>& addUbSize);
    __aicore__ inline void VfPerCoreNonZeroNum3(
        int32_t loop, int32_t computeNum, LocalTensor<T1>& xUbSize, LocalTensor<uint32_t>& maskUbSize,
        LocalTensor<uint32_t>& addUbSize);
    __aicore__ inline void VfPerCoreNonZeroNum4(int32_t loop, int32_t computeNum, LocalTensor<T1> &xUbSize,
        LocalTensor<uint32_t> &maskUbSize, LocalTensor<uint32_t> &addUbSize);
    __aicore__ inline void CopyOut(uint64_t arNum, int32_t loopOffset, int32_t alignInt32, LocalTensor<T2>& outUbSize);
    __aicore__ inline void CopyOutTrans(uint64_t arNum, uint64_t looOffset, LocalTensor<T2>& outUbSize);

    __aicore__ inline void TransComputeOut2(int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32);
    __aicore__ inline void TransComputeOut3(int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32);
    __aicore__ inline void TransComputeOut4(int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32);
    __aicore__ inline void NoTrans2(int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32);
    __aicore__ inline void NoTrans3(int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32);
    __aicore__ inline void NoTrans4(int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32);
    __aicore__ inline void ComputOut2(
        uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize);
    __aicore__ inline void ComputOut3(
        uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize);
    __aicore__ inline void ComputOut4(
        uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize);
    __aicore__ inline void ComputOutNo2(
        uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize);
    __aicore__ inline void ComputOutNo3(
        uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize);
    __aicore__ inline void ComputOutNo4(
        uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize);
    __aicore__ inline void CopyInTail(
        int32_t loopNum, int32_t tailNum, LocalTensor<uint32_t>& addUbSize, LocalTensor<uint32_t>& maskUbSize);
};

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::VfCleanUb(LocalTensor<uint32_t>& addUbSize)
{
    auto cleanUbPtr = (__ubuf__ uint32_t*)addUbSize.GetPhyAddr();
    uint32_t mask = VF_LEN_INT32;
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> cleanReg;
        MaskReg maskClean;
        maskClean = UpdateMask<uint32_t>(mask);
        Duplicate(cleanReg, (uint32_t)0);
        DataCopy(cleanUbPtr, cleanReg, maskClean);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::InitBase(
    GM_ADDR x, GM_ADDR y, GM_ADDR outShape, const NonZeroTilingData* tilingData)
{
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    yGm_.SetGlobalBuffer((__gm__ T2*)y);
    shapeGm_.SetGlobalBuffer((__gm__ uint64_t*)outShape);
    tilingData_ = tilingData;

    pipe.InitBuffer(inQueX_, BUFFER_NUM, tilingData_->xInputSize);
    pipe.InitBuffer(addUb_, ADD_UB_SIZE);
    pipe.InitBuffer(maskUb_, tilingData_->maskSize);
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::CopyInTail(
    int32_t loopNum, int32_t tailNum, LocalTensor<uint32_t>& addUbSize, LocalTensor<uint32_t>& maskUbSize)
{
    LocalTensor<T1> xUb = inQueX_.AllocTensor<T1>();
    copyParams_.blockCount = 1;
    copyParams_.blockLen = tailNum * sizeof(T1);
    copyParams_.srcStride = 0;
    copyParams_.dstStride = 0;
    DataCopyPad(xUb, xGm_[loopNum * tilingData_->ubFactorNum], copyParams_, padParams_);
    inQueX_.EnQue<QuePosition::VECIN, QuePosition::VECCALC>(xUb);
    LocalTensor<T1> xUbSize = inQueX_.DeQue<QuePosition::VECIN, QuePosition::VECCALC, T1>();
    if constexpr (sizeof(T1) == sizeof(int8_t)) {
        VfPerCoreNonZeroNum1(loopNum, tailNum, xUbSize, maskUbSize, addUbSize);
    } else if constexpr (sizeof(T1) == sizeof(int16_t)) {
        VfPerCoreNonZeroNum2(loopNum, tailNum, xUbSize, maskUbSize, addUbSize);
    } else if constexpr (sizeof(T1) == sizeof(int32_t)) {
        VfPerCoreNonZeroNum3(loopNum, tailNum, xUbSize, maskUbSize, addUbSize);
    } else if constexpr (sizeof(T1) == sizeof(int64_t)) {
        VfPerCoreNonZeroNum4(loopNum, tailNum, xUbSize, maskUbSize, addUbSize);
    }
    inQueX_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(xUbSize);
    LocalTensor<T1> outUbSize = inQueX_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T1>();
    inQueX_.FreeTensor(outUbSize);
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::CopyInBefore(
    int32_t loopNum, int32_t beforeNum, int32_t tailNum, LocalTensor<uint32_t>& addUbSize,
    LocalTensor<uint32_t>& maskUbSize)
{
    for (int32_t loopCore = 0; loopCore < loopNum; loopCore++) {
        LocalTensor<T1> xUb = inQueX_.AllocTensor<T1>();
        copyParams_.blockCount = 1;
        copyParams_.blockLen = beforeNum * sizeof(T1);
        copyParams_.srcStride = 0;
        copyParams_.dstStride = 0;
        DataCopyPad(xUb, xGm_[loopCore * tilingData_->ubFactorNum], copyParams_, padParams_);
        inQueX_.EnQue<QuePosition::VECIN, QuePosition::VECCALC>(xUb);
        LocalTensor<T1> xUbSize = inQueX_.DeQue<QuePosition::VECIN, QuePosition::VECCALC, T1>();

        if constexpr (sizeof(T1) == sizeof(int8_t)) {
            VfPerCoreNonZeroNum1(loopCore, beforeNum, xUbSize, maskUbSize, addUbSize);
        } else if constexpr (sizeof(T1) == sizeof(int16_t)) {
            VfPerCoreNonZeroNum2(loopCore, beforeNum, xUbSize, maskUbSize, addUbSize);
        } else if constexpr (sizeof(T1) == sizeof(int32_t)) {
            VfPerCoreNonZeroNum3(loopCore, beforeNum, xUbSize, maskUbSize, addUbSize);
        } else if constexpr (sizeof(T1) == sizeof(int64_t)) {
            VfPerCoreNonZeroNum4(loopCore, beforeNum, xUbSize, maskUbSize, addUbSize);
        }
        inQueX_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(xUbSize);
        LocalTensor<T1> outUbSize = inQueX_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T1>();
        inQueX_.FreeTensor(outUbSize);
    }
    CopyInTail(loopNum, tailNum, addUbSize, maskUbSize);
    ComputeNonZeroNum(addUbSize);
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::ComputeNonZeroNum(LocalTensor<uint32_t>& addUbSize)
{
    VfReduceSum(addUbSize);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    allNum_ = addUbSize.GetValue(0);
    LocalTensor<uint64_t> tensorNew = addUbSize.ReinterpretCast<uint64_t>();
    if (tilingData_->needTranspose) {
        tensorNew.SetValue(DIM16, NON_ZERO_OUTSHAPE_DIM_BASE);
        tensorNew.SetValue(DIM17, allNum_);
        tensorNew.SetValue(DIM18, tilingData_->inputDims);
    } else {
        tensorNew.SetValue(DIM16, NON_ZERO_OUTSHAPE_DIM_BASE);
        tensorNew.SetValue(DIM17, tilingData_->inputDims);
        tensorNew.SetValue(DIM18, allNum_);
    }
    DataCopyPad(shapeGm_, tensorNew[DIM16], {1, 24, 0, 0});
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::VfReduceSum(LocalTensor<uint32_t>& addUbSize)
{
    auto addUbPtr = (__ubuf__ uint32_t*)addUbSize.GetPhyAddr();
    uint32_t mask = VF_LEN_INT32;
    uint32_t oneMask = 1;
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> addReg;
        RegTensor<uint32_t> dstReg;
        MaskReg addMask;
        MaskReg oneMaskReg;
        addMask = UpdateMask<uint32_t>(mask);
        oneMaskReg = UpdateMask<uint32_t>(oneMask);
        DataCopy(addReg, addUbPtr);
        ReduceSum(dstReg, addReg, addMask);
        DataCopy(addUbPtr, dstReg, oneMaskReg);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::VfComputeIds(
    int32_t loopIndex, int32_t loopNum, LocalTensor<int32_t>& outUbSize, LocalTensor<uint32_t>& maskUbSize,
    uint64_t& arNum)
{
    uint32_t sreg = (uint32_t)loopNum;
    uint16_t repeatTimes = CeilDivision(loopNum, VF_LEN_INT32);
    auto dstPtr = (__ubuf__ int32_t*)outUbSize.GetPhyAddr() + tilingData_->offsetInt32Trans;
    auto maskUbPtr = (__ubuf__ uint32_t*)maskUbSize.GetPhyAddr() + loopIndex * tilingData_->maskLoopNumO;
    int32_t numPerCore1 = tilingData_->numPerCore;
    int32_t beforeNumO1 = tilingData_->beforeNumO;
    int32_t scalar = loopIndex * beforeNumO1;
    int32_t vfLenInt32 = VF_LEN_INT32;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
        AscendC::MicroAPI::UnalignReg ureg0;
        RegTensor<int32_t> vsqzReg;
        MaskReg maskReg;
        MaskReg preg;
        RegTensor<int32_t> idsReg;
        Arange(idsReg, scalar);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = UpdateMask<uint32_t>(sreg);
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(i, 8);
            AscendC::MicroAPI::DataCopy(maskReg, maskUbPtr, offset);
            AscendC::MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, maskReg);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstPtr, vsqzReg, ureg0);
            Adds(idsReg, idsReg, vfLenInt32, preg);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(dstPtr, ureg0);
    }
    arNum = (AscendC::MicroAPI::GetSpr<SpecialPurposeReg::AR>()) / 4;
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::VfComputeIdsB64(int32_t loopIndex, int32_t loopNum,
    LocalTensor<int32_t> &outUbSize, LocalTensor<uint32_t> &maskUbSize, uint64_t &arNum)
{
    uint32_t sreg = (uint32_t)loopNum;
    uint32_t repeatElm = (uint32_t)VF_LEN_INT64;
    uint16_t repeatTimes = CeilDivision(loopNum, repeatElm);
    auto dstPtr = (__ubuf__ int32_t *)outUbSize.GetPhyAddr() + tilingData_->offsetInt32Trans;
    auto maskUbPtr = (__ubuf__ uint64_t *)maskUbSize.GetPhyAddr() + loopIndex * tilingData_->maskLoopNumO;
    int32_t numPerCore1 = tilingData_->numPerCore;
    int32_t beforeNumO1 = tilingData_->beforeNumO;
    int32_t scalar = loopIndex * beforeNumO1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
        AscendC::MicroAPI::UnalignReg ureg0;
        RegTensor<int32_t> vsqzReg;
        RegTensor<int32_t> idsReg;
        MaskReg maskReg, maskHalf;
        MaskReg preg;
        preg = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        Arange(idsReg, scalar);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<int64_t>(i, 4);
            AscendC::MicroAPI::DataCopy(maskReg, maskUbPtr, offset);
            AscendC::MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(maskHalf, maskReg);
            
            AscendC::MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::STORE_REG>(vsqzReg, idsReg, maskHalf);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstPtr, vsqzReg, ureg0);

            AscendC::MicroAPI::Adds(idsReg, idsReg, repeatElm, preg);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(dstPtr, ureg0);
    }
    arNum = (AscendC::MicroAPI::GetSpr<SpecialPurposeReg::AR>()) / 4;
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::VfPerCoreNonZeroNum1(
    int32_t loop, int32_t computeNum, LocalTensor<T1>& xUbSize, LocalTensor<uint32_t>& maskUbSize,
    LocalTensor<uint32_t>& addUbSize)
{
    auto xUbPtr = (__ubuf__ T1*)xUbSize.GetPhyAddr();
    auto maskUbPtr = (__ubuf__ uint32_t*)maskUbSize.GetPhyAddr() + loop * tilingData_->maskLoopNum;
    auto maskUbPtr2 = (__ubuf__ uint32_t*)maskUbSize.GetPhyAddr() + loop * tilingData_->maskLoopNum + DIM8;
    auto maskUbPtr3 = (__ubuf__ uint32_t*)maskUbSize.GetPhyAddr() + loop * tilingData_->maskLoopNum + DIM8 * DIM2;
    auto maskUbPtr4 = (__ubuf__ uint32_t*)maskUbSize.GetPhyAddr() + loop * tilingData_->maskLoopNum + DIM8 * DIM3;
    auto dstUbPtr = (__ubuf__ uint32_t*)addUbSize.GetPhyAddr();
    uint32_t sreg = (uint32_t)computeNum;
    uint16_t repeatElm = (uint16_t)vfLenT1_;
    uint16_t repeatTimes = CeilDivision(computeNum, repeatElm);
    uint32_t addMask = VF_LEN_INT32;
    __VEC_SCOPE__
    {
        MaskReg preg, addComReg, pregUnpack, pregUnpackH, pregUnpackL, pregUnpackH1, pregUnpackL1, pregUnpackH2;
        RegTensor<T1> xSrcReg;
        RegTensor<uint8_t> src1Reg, src0Reg;
        RegTensor<uint8_t> selectReg;
        RegTensor<uint8_t> maskUbReg;
        MaskReg cmpReg;
        RegTensor<uint32_t> addReg;
        RegTensor<uint32_t> dstReg;
        RegTensor<uint16_t> uint16Reg;
        RegTensor<uint16_t> uint16Reg1;
        RegTensor<uint32_t> uint32Reg;
        RegTensor<uint32_t> uint32Reg1;
        RegTensor<uint32_t> uint32Reg2;
        RegTensor<uint32_t> uint32Reg3;
        DataCopy(addReg, dstUbPtr);
        Duplicate(src1Reg, (uint8_t)1);
        Duplicate(src0Reg, (uint8_t)0);
        addComReg = UpdateMask<uint32_t>(addMask);
        for (uint16_t maskI = 0; maskI < repeatTimes; maskI++) {
            preg = UpdateMask<T1>(sreg);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<uint8_t>(maskI, repeatElm);
            AscendC::MicroAPI::AddrReg srcOffset1 = AscendC::MicroAPI::CreateAddrReg<uint32_t>(maskI, 32);
            DataCopy(xSrcReg, xUbPtr, srcOffset);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            Select(selectReg, src1Reg, src0Reg, cmpReg);
            MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(pregUnpack, cmpReg);
            MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(pregUnpackH, cmpReg);

            MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(pregUnpackL, pregUnpack);
            MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(pregUnpackH1, pregUnpack);
            MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(pregUnpackL1, pregUnpackH);
            MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(pregUnpackH2, pregUnpackH);

            AscendC::MicroAPI::DataCopy(maskUbPtr, pregUnpackL, srcOffset1);
            AscendC::MicroAPI::DataCopy(maskUbPtr2, pregUnpackH1, srcOffset1);
            AscendC::MicroAPI::DataCopy(maskUbPtr3, pregUnpackL1, srcOffset1);
            AscendC::MicroAPI::DataCopy(maskUbPtr4, pregUnpackH2, srcOffset1);

            UnPack<uint16_t, uint8_t, AscendC::MicroAPI::HighLowPart::LOWEST>(uint16Reg, selectReg);
            UnPack<uint16_t, uint8_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(uint16Reg1, selectReg);
            UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(uint32Reg, uint16Reg);
            UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(uint32Reg1, uint16Reg);
            UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(uint32Reg2, uint16Reg1);
            UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(uint32Reg3, uint16Reg1);
            Add(addReg, uint32Reg, addReg, addComReg);
            Add(addReg, uint32Reg1, addReg, addComReg);
            Add(addReg, uint32Reg2, addReg, addComReg);
            Add(addReg, uint32Reg3, addReg, addComReg);
        }
        DataCopy(dstUbPtr, addReg, addComReg);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::VfPerCoreNonZeroNum2(
    int32_t loop, int32_t computeNum, LocalTensor<T1>& xUbSize, LocalTensor<uint32_t>& maskUbSize,
    LocalTensor<uint32_t>& addUbSize)
{
    auto xUbPtr = (__ubuf__ T1*)xUbSize.GetPhyAddr();
    auto maskUbPtr = (__ubuf__ uint32_t*)maskUbSize.GetPhyAddr() + loop * tilingData_->maskLoopNum;
    auto maskUbPtr2 = (__ubuf__ uint32_t*)maskUbSize.GetPhyAddr() + loop * tilingData_->maskLoopNum + DIM8;
    auto dstUbPtr = (__ubuf__ uint32_t*)addUbSize.GetPhyAddr();
    uint32_t sreg = (uint32_t)computeNum;
    uint16_t repeatElm = (uint16_t)vfLenT1_;
    uint16_t repeatTimes = CeilDivision(computeNum, repeatElm);
    uint32_t addMask = VF_LEN_INT32;
    __VEC_SCOPE__
    {
        MaskReg preg, addComReg, pregUnpackL, pregUnpackH, cmpReg;
        RegTensor<T1> xSrcReg;
        RegTensor<uint16_t> src1Reg;
        RegTensor<uint16_t> src0Reg;
        RegTensor<uint16_t> selectReg;
        RegTensor<uint8_t> maskUbReg;
        RegTensor<uint32_t> addReg;
        RegTensor<uint32_t> dstReg;
        RegTensor<uint32_t> uint16Reg;
        RegTensor<uint32_t> uint16Reg1;
        RegTensor<uint32_t> uint32Reg;
        RegTensor<uint32_t> uint32Reg1;
        RegTensor<uint32_t> uint32Reg2;
        RegTensor<uint32_t> uint32Reg3;
        DataCopy(addReg, dstUbPtr);
        Duplicate(src1Reg, (uint16_t)1);
        Duplicate(src0Reg, (uint16_t)0);
        addComReg = UpdateMask<uint32_t>(addMask);

        for (uint16_t maskI = 0; maskI < repeatTimes; maskI++) {
            preg = UpdateMask<T1>(sreg);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<uint16_t>(maskI, repeatElm);
            AscendC::MicroAPI::AddrReg srcOffset1 = AscendC::MicroAPI::CreateAddrReg<uint32_t>(maskI, 16);
            DataCopy(xSrcReg, xUbPtr, srcOffset);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            Select(selectReg, src1Reg, src0Reg, cmpReg);
            MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(pregUnpackL, cmpReg);
            MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(pregUnpackH, cmpReg);

            AscendC::MicroAPI::DataCopy(maskUbPtr, pregUnpackL, srcOffset1);
            AscendC::MicroAPI::DataCopy(maskUbPtr2, pregUnpackH, srcOffset1);

            UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(uint16Reg, selectReg);
            UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(uint16Reg1, selectReg);

            Add(addReg, uint16Reg, addReg, addComReg);
            Add(addReg, uint16Reg1, addReg, addComReg);
        }
        DataCopy(dstUbPtr, addReg, addComReg);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::VfPerCoreNonZeroNum3(
    int32_t loop, int32_t computeNum, LocalTensor<T1>& xUbSize, LocalTensor<uint32_t>& maskUbSize,
    LocalTensor<uint32_t>& addUbSize)
{
    auto xUbPtr = (__ubuf__ T1*)xUbSize.GetPhyAddr();
    auto maskUbPtr = (__ubuf__ uint32_t*)maskUbSize.GetPhyAddr() + loop * tilingData_->maskLoopNum;
    auto dstUbPtr = (__ubuf__ uint32_t*)addUbSize.GetPhyAddr();
    uint32_t sreg = (uint32_t)computeNum;
    uint16_t repeatElm = (uint16_t)vfLenT1_;
    uint16_t repeatTimes = CeilDivision(computeNum, repeatElm);
    uint32_t addMask = VF_LEN_INT32;
    __VEC_SCOPE__
    {
        MaskReg preg, addComReg, cmpReg;
        RegTensor<T1> xSrcReg;
        RegTensor<uint32_t> src1Reg;
        RegTensor<uint32_t> src0Reg;
        RegTensor<uint32_t> selectReg;
        RegTensor<uint8_t> maskUbReg;
        RegTensor<uint32_t> addReg;
        RegTensor<uint32_t> dstReg;
        RegTensor<uint32_t> uint16Reg;
        RegTensor<uint32_t> uint16Reg1;
        RegTensor<uint32_t> uint32Reg;
        RegTensor<uint32_t> uint32Reg1;
        RegTensor<uint32_t> uint32Reg2;
        RegTensor<uint32_t> uint32Reg3;
        DataCopy(addReg, dstUbPtr);
        Duplicate(src1Reg, (uint32_t)1);
        Duplicate(src0Reg, (uint32_t)0);
        addComReg = UpdateMask<uint32_t>(addMask);
        for (uint16_t maskI = 0; maskI < repeatTimes; maskI++) {
            preg = UpdateMask<T1>(sreg);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(maskI, repeatElm);
            AscendC::MicroAPI::AddrReg srcOffset1 = AscendC::MicroAPI::CreateAddrReg<uint32_t>(maskI, 8);
            DataCopy(xSrcReg, xUbPtr, srcOffset);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            Select(selectReg, src1Reg, src0Reg, cmpReg);
            AscendC::MicroAPI::DataCopy(maskUbPtr, cmpReg, srcOffset1);
            Add(addReg, selectReg, addReg, addComReg);
        }
        DataCopy(dstUbPtr, addReg, addComReg);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::VfPerCoreNonZeroNum4(int32_t loop, int32_t computeNum,
    LocalTensor<T1> &xUbSize, LocalTensor<uint32_t> &maskUbSize, LocalTensor<uint32_t> &addUbSize)
{
    auto xUbPtr = (__ubuf__ T1 *)xUbSize.GetPhyAddr();
    auto maskUbPtr = (__ubuf__ uint64_t *)maskUbSize.GetPhyAddr() + static_cast<uint64_t>(loop) * tilingData_->maskLoopNum;
    auto dstUbPtr = (__ubuf__ uint64_t *)addUbSize.GetPhyAddr();
    uint32_t sreg = (uint32_t)computeNum;
    uint32_t repeatElm = (uint32_t)vfLenT1_;
    uint16_t repeatTimes = CeilDivision(computeNum, repeatElm);
    uint32_t addMask = VF_LEN_INT64;
    __VEC_SCOPE__
    {
        MaskReg preg, addComReg, cmpReg;
        RegTensor<T1> xSrcReg;
        RegTensor<uint64_t> src1Reg;
        RegTensor<uint64_t> src0Reg;
        RegTensor<uint64_t> selectReg;
        RegTensor<uint64_t> addReg;

        DataCopy(addReg, dstUbPtr);
        Duplicate(src1Reg, (uint64_t)1);
        Duplicate(src0Reg, (uint64_t)0);
        addComReg = UpdateMask<uint64_t>(addMask);
        for (uint16_t maskI = 0; maskI < repeatTimes; maskI++) {
            preg = UpdateMask<uint64_t>(sreg);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<uint64_t>(maskI, repeatElm);
            AscendC::MicroAPI::AddrReg srcOffset1 = AscendC::MicroAPI::CreateAddrReg<uint64_t>(maskI, 4);
            DataCopy(xSrcReg, xUbPtr, srcOffset);
            CompareScalar<T1, CMPMODE::NE>(cmpReg, xSrcReg, (T1)0, preg);
            Select(selectReg, src1Reg, src0Reg, cmpReg);
            AscendC::MicroAPI::DataCopy(maskUbPtr, cmpReg, srcOffset1);
            Add(addReg, selectReg, addReg, addComReg);
        }
        DataCopy(dstUbPtr, addReg, addComReg);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::CopyOut(
    uint64_t arNum, int32_t loopOffset, int32_t alignInt32, LocalTensor<T2>& outUbSize)
{
    int32_t alignInt64 = sizeof(T2) == DIM8 ? ((arNum + DIM4 - 1) / DIM4) * DIM4 : alignInt32;
    int32_t srcStride = ((alignInt32 - alignInt64) * sizeof(T2)) / ONE_BLOCK;
    copyParams_.blockCount = tilingData_->inputDims;
    copyParams_.blockLen = arNum * sizeof(T2);
    copyParams_.srcStride = srcStride;
    copyParams_.dstStride = (allNum_ - arNum) * sizeof(T2);
    DataCopyPad(yGm_[loopOffset - arNum], outUbSize, copyParams_);
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::CopyOutTrans(
    uint64_t arNum, uint64_t looOffset, LocalTensor<T2>& outUbSize)
{
    copyParams_.blockCount = 1;
    copyParams_.blockLen = tilingData_->inputDims * arNum * sizeof(T2);
    copyParams_.srcStride = 0;
    copyParams_.dstStride = 0;
    DataCopyPad(yGm_[(looOffset - arNum) * tilingData_->inputDims], outUbSize, copyParams_);
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::NoTrans2(
    int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32)
{
    uint16_t repeatTimes = CeilDivision(alignInt32 * tilingData_->inputDims, VF_LEN_INT64);
    uint16_t repeatTimes1 = CeilDivision(arNum, VF_LEN_INT32);
    uint32_t sreg = (int32_t)arNum;
    auto dstInt64Ptr = (__ubuf__ uint32_t*)dstUbInt32.GetPhyAddr();
    auto dstInt32Ptr = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    auto dstInt32Ptr2 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + alignInt32].GetPhyAddr();
    auto dstInt64Ptr1 = (__ubuf__ int32_t*)dstUbInt32.GetPhyAddr();
    auto dstInt32Ptr1 = (__ubuf__ int32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    uint32_t sregInt32 = alignInt32 * tilingData_->inputDims * DIM2;
    uint32_t m0 = tilingData_->quickDivRMList[0];
    uint32_t shape0 = tilingData_->mulInDimRList[0];
    int16_t k0 = tilingData_->quickDivRKList[0];
    int32_t vfLenInt32 = VF_LEN_INT32;
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<uint32_t> srcReg;
        RegTensor<uint32_t> qmulReg;
        RegTensor<uint32_t> subReg;
        RegTensor<uint32_t> dst0;
        RegTensor<uint32_t> dst1;
        RegTensor<T2> castReg;
        Duplicate(qmulReg, m0);
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = UpdateMask<uint32_t>(sreg);
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(j, vfLenInt32);
            DataCopy(srcReg, dstInt64Ptr, offset);

            ComputeOutIds(dst0, dst1, srcReg, qmulReg, k0, shape0, subReg, offset, dstInt32Ptr, preg);

            DataCopy(dstInt32Ptr2, subReg, offset, preg);
        }
        if constexpr (IsSameType<T2, int64_t>::value) {
            mem_bar(VST_VLD);
            CastInt32(repeatTimes, preg, sregInt32, dst1, dstInt32Ptr1, dstInt64Ptr1);
        }
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::NoTrans3(
    int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32)
{
    uint16_t repeatTimes = CeilDivision(alignInt32 * tilingData_->inputDims, VF_LEN_INT64);
    uint16_t repeatTimes1 = CeilDivision(arNum, VF_LEN_INT32);
    uint32_t sreg = (int32_t)arNum;
    int32_t vfLenInt32 = VF_LEN_INT32;
    auto dstInt64Ptr = (__ubuf__ uint32_t*)dstUbInt32.GetPhyAddr();
    auto dstInt32Ptr = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    auto dstInt32Ptr2 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + alignInt32].GetPhyAddr();
    auto dstInt32Ptr3 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + alignInt32 * 2].GetPhyAddr();
    auto dstInt64Ptr1 = (__ubuf__ int32_t*)dstUbInt32.GetPhyAddr();
    auto dstInt32Ptr1 = (__ubuf__ int32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    uint32_t sregInt32 = alignInt32 * tilingData_->inputDims * DIM2;
    uint32_t m1 = tilingData_->quickDivRMList[0];
    uint32_t m0 = tilingData_->quickDivRMList[1];
    uint32_t shape1 = tilingData_->mulInDimRList[0];
    uint32_t shape0 = tilingData_->mulInDimRList[1];
    int16_t k0 = tilingData_->quickDivRKList[1];
    int16_t k1 = tilingData_->quickDivRKList[0];
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<uint32_t> srcReg, qmulReg, q1mulReg, subReg, dst0, dst1;
        RegTensor<T2> castReg;
        Duplicate(qmulReg, m0);
        Duplicate(q1mulReg, m1);
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = UpdateMask<uint32_t>(sreg);
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(j, vfLenInt32);
            DataCopy(srcReg, dstInt64Ptr, offset);

            ComputeOutIds(dst0, dst1, srcReg, q1mulReg, k1, shape1, subReg, offset, dstInt32Ptr, preg);

            ComputeOutIds(dst0, dst1, subReg, qmulReg, k0, shape0, srcReg, offset, dstInt32Ptr2, preg);

            DataCopy(dstInt32Ptr3, srcReg, offset, preg);
        }
        if constexpr (IsSameType<T2, int64_t>::value) {
            mem_bar(VST_VLD);
            CastInt32(repeatTimes, preg, sregInt32, dst1, dstInt32Ptr1, dstInt64Ptr1);
        }
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::NoTrans4(
    int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32)
{
    uint16_t repeatTimes = CeilDivision(alignInt32 * tilingData_->inputDims, VF_LEN_INT64);
    uint16_t repeatTimes1 = CeilDivision(arNum, VF_LEN_INT32);
    uint32_t sreg = (int32_t)arNum;
    auto dstInt64Ptr = (__ubuf__ uint32_t*)dstUbInt32.GetPhyAddr();
    auto dstInt32Ptr = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    auto dstInt32Ptr2 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + alignInt32].GetPhyAddr();
    auto dstInt32Ptr3 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + alignInt32 * 2].GetPhyAddr();
    auto dstInt32Ptr4 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + alignInt32 * 3].GetPhyAddr();
    auto dstInt64Ptr1 = (__ubuf__ int32_t*)dstUbInt32.GetPhyAddr();
    auto dstInt32Ptr1 = (__ubuf__ int32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    uint32_t sregInt32 = alignInt32 * tilingData_->inputDims * DIM2;
    uint32_t m2 = tilingData_->quickDivRMList[0];
    uint32_t m1 = tilingData_->quickDivRMList[1];
    uint32_t m0 = tilingData_->quickDivRMList[2];
    uint32_t shape2 = tilingData_->mulInDimRList[0];
    uint32_t shape1 = tilingData_->mulInDimRList[1];
    uint32_t shape0 = tilingData_->mulInDimRList[2];
    int16_t k2 = tilingData_->quickDivRKList[0];
    int16_t k1 = tilingData_->quickDivRKList[1];
    int16_t k0 = tilingData_->quickDivRKList[2];
    int32_t vfLenInt32 = VF_LEN_INT32;
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<uint32_t> srcReg, qmulReg, q1mulReg, q2mulReg, subReg, dst0, dst1;
        RegTensor<T2> castReg;
        Duplicate(qmulReg, m0);
        Duplicate(q1mulReg, m1);
        Duplicate(q2mulReg, m2);
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = UpdateMask<uint32_t>(sreg);
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(j, vfLenInt32);
            DataCopy(srcReg, dstInt64Ptr, offset);

            ComputeOutIds(dst0, dst1, srcReg, q2mulReg, k2, shape2, subReg, offset, dstInt32Ptr, preg);

            ComputeOutIds(dst0, dst1, subReg, q1mulReg, k1, shape1, srcReg, offset, dstInt32Ptr2, preg);
            ComputeOutIds(dst0, dst1, srcReg, qmulReg, k0, shape0, subReg, offset, dstInt32Ptr3, preg);

            DataCopy(dstInt32Ptr4, subReg, offset, preg);
        }
        if constexpr (IsSameType<T2, int64_t>::value) {
            mem_bar(VST_VLD);
            CastInt32(repeatTimes, preg, sregInt32, dst1, dstInt32Ptr1, dstInt64Ptr1);
        }
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::TransComputeOut2(
    int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32)
{
    uint16_t repeatTimes = CeilDivision(alignInt32 * tilingData_->inputDims, VF_LEN_INT64);
    int32_t vfLenInt32 = VF_LEN_INT32;
    uint16_t repeatTimes1 = arNum / vfLenInt32;
    uint16_t tail = arNum % vfLenInt32;
    uint32_t sreg1 = tail;
    uint32_t sreg = (int32_t)arNum;
    uint32_t sreg2 = tail <= ONE_BLOCK ? tail * DIM2 : vfLenInt32;
    int32_t kk = tail * DIM2 - vfLenInt32;
    uint32_t sreg3 = kk <= 0 ? 0 : (uint32_t)kk;
    auto dstInt64Ptr = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt32Trans].GetPhyAddr();
    auto dstInt32Ptr = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    auto dstInt32Ptr2 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + vfLenInt32].GetPhyAddr();
    auto dstInt64Ptr1 = (__ubuf__ int32_t*)dstUbInt32.GetPhyAddr();
    auto dstInt32Ptr1 = (__ubuf__ int32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    uint32_t sregInt32 = alignInt32 * tilingData_->inputDims * DIM2;
    uint32_t m0 = tilingData_->quickDivRMList[0];
    int16_t k0 = tilingData_->quickDivRKList[0];
    uint32_t shape0 = tilingData_->mulInDimRList[0];

    __VEC_SCOPE__
    {
        MaskReg preg, preg1, preg2, preg3;
        RegTensor<uint32_t> srcReg, qmulReg, subReg, dst0, dst1;
        RegTensor<T2> castReg;
        Duplicate(qmulReg, m0);
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = UpdateMask<uint32_t>(sreg);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(j, vfLenInt32);
            DataCopy(srcReg, dstInt64Ptr, srcOffset);

            ComputeOutIdsTrans24(dst0, dst1, srcReg, qmulReg, preg, shape0, k0, subReg);

            Interleave(dst0, srcReg, dst1, subReg);
            AscendC::MicroAPI::AddrReg dstOffset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(j, 128);
            DataCopy(dstInt32Ptr, dst0, dstOffset, preg);
            DataCopy(dstInt32Ptr2, srcReg, dstOffset, preg);
        }
        if (tail != 0) {
            preg1 = UpdateMask<uint32_t>(sreg1);
            preg2 = UpdateMask<uint32_t>(sreg2);
            preg3 = UpdateMask<uint32_t>(sreg3);
            DataCopy(srcReg, dstInt64Ptr + repeatTimes1 * 64);
            ComputeOutIdsTrans24(dst0, dst1, srcReg, qmulReg, preg1, shape0, k0, subReg);

            Interleave(dst0, srcReg, dst1, subReg);
            DataCopy(dstInt32Ptr + 128 * repeatTimes1, dst0, preg2);
            DataCopy(dstInt32Ptr + 128 * repeatTimes1 + vfLenInt32, srcReg, preg3);
        }
        if constexpr (IsSameType<T2, int64_t>::value) {
            mem_bar(VST_VLD);
            CastInt32(repeatTimes, preg, sregInt32, subReg, dstInt32Ptr1, dstInt64Ptr1);
        }
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::TransComputeOut3(
    int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32)
{
    uint16_t repeatTimes = CeilDivision(alignInt32 * tilingData_->inputDims, VF_LEN_INT64);
    uint16_t repeatTimes1 = CeilDivision(arNum, VF_LEN_INT32);
    uint32_t sreg = (int32_t)arNum;
    auto dstInt64Ptr = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt32Trans].GetPhyAddr();
    auto dstInt32Ptr = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    auto dstInt64Ptr1 = (__ubuf__ int32_t*)dstUbInt32.GetPhyAddr();
    auto dstInt32Ptr1 = (__ubuf__ int32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    uint32_t sregInt32 = alignInt32 * tilingData_->inputDims * DIM2;
    uint32_t m1 = tilingData_->quickDivRMList[0];
    uint32_t m0 = tilingData_->quickDivRMList[1];
    int16_t k1 = tilingData_->quickDivRKList[0];
    int16_t k0 = tilingData_->quickDivRKList[1];
    uint32_t shape1 = tilingData_->mulInDimRList[0];
    uint32_t shape0 = tilingData_->mulInDimRList[1];
    int32_t vfLenInt32 = VF_LEN_INT32;
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<uint32_t> srcReg, qmulReg, q1mulReg, subReg, dst0, dst1;
        RegTensor<int32_t> transReg, transReg1, transReg2, transReg8;
        RegTensor<T2> castReg;
        Duplicate(qmulReg, m0);
        Duplicate(q1mulReg, m1);
        Arange(transReg, 0);
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = UpdateMask<uint32_t>(sreg);
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(j, vfLenInt32);
            DataCopy(srcReg, dstInt64Ptr, offset);
            Muls(transReg2, transReg, 3, preg);
            ComputeOutIdsTrans(dst0, dst1, srcReg, q1mulReg, subReg, transReg2, dstInt32Ptr, k1, shape1, preg);
            ComputeOutIdsTrans1(dst0, dst1, subReg, qmulReg, transReg2, transReg1, dstInt32Ptr, k0, shape0, 1, preg);
            Adds(transReg8, transReg2, (int32_t)2, preg);
            Adds(transReg, transReg, vfLenInt32, preg);
            DataCopyScatter(dstInt32Ptr, subReg, (RegTensor<uint32_t>&)transReg8, preg);
        }
        if constexpr (IsSameType<T2, int64_t>::value) {
            mem_bar(VST_VLD);
            CastInt32(repeatTimes, preg, sregInt32, dst1, dstInt32Ptr1, dstInt64Ptr1);
        }
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::TransComputeOut4(
    int32_t alignInt32, uint64_t arNum, LocalTensor<int32_t>& dstUbInt32)
{
    uint16_t repeatTimes = CeilDivision(alignInt32 * tilingData_->inputDims, VF_LEN_INT64);
    int32_t vfLenInt32 = VF_LEN_INT32;
    uint16_t repeatTimes1 = arNum / vfLenInt32;
    uint16_t tail = arNum % vfLenInt32;
    uint32_t sreg1 = tail;
    uint32_t sreg = (int32_t)arNum;
    auto dstInt64Ptr = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt32Trans].GetPhyAddr();
    auto dstInt32Ptr = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    auto dstInt32Ptr2 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + vfLenInt32].GetPhyAddr();
    auto dstInt32Ptr3 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + vfLenInt32 * DIM2].GetPhyAddr();
    auto dstInt32Ptr4 = (__ubuf__ uint32_t*)dstUbInt32[tilingData_->offsetInt64 + vfLenInt32 * DIM3].GetPhyAddr();
    auto dstInt64Ptr1 = (__ubuf__ int32_t*)dstUbInt32.GetPhyAddr();
    auto dstInt32Ptr1 = (__ubuf__ int32_t*)dstUbInt32[tilingData_->offsetInt64].GetPhyAddr();
    uint32_t sregInt32 = alignInt32 * tilingData_->inputDims * DIM2;
    uint32_t sreg3 = 0;
    uint32_t sreg4 = 0;
    uint32_t sreg5 = 0;
    uint32_t sreg2 = 0;
    uint32_t m2 = tilingData_->quickDivRMList[0];
    uint32_t m1 = tilingData_->quickDivRMList[1];
    uint32_t m0 = tilingData_->quickDivRMList[2];
    uint32_t shape2 = tilingData_->mulInDimRList[0];
    uint32_t shape1 = tilingData_->mulInDimRList[1];
    uint32_t shape0 = tilingData_->mulInDimRList[2];
    int16_t k2 = tilingData_->quickDivRKList[0];
    int16_t k1 = tilingData_->quickDivRKList[1];
    int16_t k0 = tilingData_->quickDivRKList[2];
    if (tail <= (ONE_BLOCK / DIM2)) {
        sreg2 = (uint32_t)tail * DIM4;
    } else if (tail <= ONE_BLOCK) {
        sreg2 = (uint32_t)vfLenInt32;
        sreg3 = (uint32_t)tail * DIM4 - vfLenInt32;
    } else if (tail <= (ONE_BLOCK + ONE_BLOCK / DIM2)) {
        sreg2 = (uint32_t)vfLenInt32;
        sreg3 = (uint32_t)vfLenInt32;
        sreg4 = (uint32_t)tail * DIM4 - vfLenInt32 * DIM2;
    } else {
        sreg2 = (uint32_t)vfLenInt32;
        sreg3 = (uint32_t)vfLenInt32;
        sreg4 = (uint32_t)vfLenInt32;
        sreg5 = (uint32_t)tail * DIM4 - vfLenInt32 * DIM3;
    }

    __VEC_SCOPE__
    {
        MaskReg preg, preg1, preg2, preg3, preg4, preg5;
        RegTensor<uint32_t> srcReg;
        RegTensor<uint32_t> qmulReg;
        RegTensor<uint32_t> q2mulReg;
        RegTensor<uint32_t> q1mulReg;
        RegTensor<uint32_t> subReg;
        RegTensor<uint32_t> sub2Reg;
        RegTensor<uint32_t> sub3Reg;
        RegTensor<uint32_t> trans1Reg;
        RegTensor<uint32_t> trans3Reg;
        RegTensor<uint32_t> trans4Reg;
        RegTensor<uint32_t> dst0;
        RegTensor<uint32_t> dst1;
        RegTensor<T2> castReg;
        Duplicate(qmulReg, m0);
        Duplicate(q1mulReg, m1);
        Duplicate(q2mulReg, m2);
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = UpdateMask<uint32_t>(sreg);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(j, vfLenInt32);
            DataCopy(srcReg, dstInt64Ptr, srcOffset);
            ComputeOutIdsTrans24(dst0, dst1, srcReg, q2mulReg, preg, shape2, k2, subReg);

            ComputeOutIdsTrans24(dst0, srcReg, subReg, q1mulReg, preg, shape1, k1, sub2Reg);

            ComputeOutIdsTrans24(dst0, sub3Reg, sub2Reg, qmulReg, preg, shape0, k0, subReg);

            Interleave(trans1Reg, dst0, dst1, sub3Reg);
            Interleave(trans3Reg, trans4Reg, srcReg, subReg);

            Interleave(dst1, subReg, trans1Reg, trans3Reg);
            Interleave(sub2Reg, sub3Reg, dst0, trans4Reg);
            AscendC::MicroAPI::AddrReg dstOffset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(j, vfLenInt32 * 4);
            DataCopy(dstInt32Ptr, dst1, dstOffset, preg);
            DataCopy(dstInt32Ptr2, subReg, dstOffset, preg);
            DataCopy(dstInt32Ptr3, sub2Reg, dstOffset, preg);
            DataCopy(dstInt32Ptr4, sub3Reg, dstOffset, preg);
        }
        if (tail != 0) {
            preg1 = UpdateMask<uint32_t>(sreg1);
            DataCopy(srcReg, dstInt64Ptr + repeatTimes1 * vfLenInt32);
            ComputeOutIdsTrans24(dst0, dst1, srcReg, q2mulReg, preg1, shape2, k2, subReg);

            ComputeOutIdsTrans24(dst0, srcReg, subReg, q1mulReg, preg1, shape1, k1, sub2Reg);

            ComputeOutIdsTrans24(dst0, sub3Reg, sub2Reg, qmulReg, preg1, shape0, k0, subReg);

            Interleave(trans1Reg, dst0, dst1, sub3Reg);
            Interleave(trans3Reg, trans4Reg, srcReg, subReg);

            Interleave(dst1, subReg, trans1Reg, trans3Reg);
            Interleave(sub2Reg, sub3Reg, dst0, trans4Reg);
            preg2 = UpdateMask<uint32_t>(sreg2);
            preg3 = UpdateMask<uint32_t>(sreg3);
            preg4 = UpdateMask<uint32_t>(sreg4);
            preg5 = UpdateMask<uint32_t>(sreg5);
            DataCopy(dstInt32Ptr + vfLenInt32 * 4 * repeatTimes1, dst1, preg2);
            DataCopy(dstInt32Ptr + vfLenInt32 * 4 * repeatTimes1 + vfLenInt32, subReg, preg3);
            DataCopy(dstInt32Ptr + vfLenInt32 * 4 * repeatTimes1 + vfLenInt32 * 2, sub2Reg, preg4);
            DataCopy(dstInt32Ptr + vfLenInt32 * 4 * repeatTimes1 + vfLenInt32 * 3, sub3Reg, preg5);
        }
        if constexpr (IsSameType<T2, int64_t>::value) {
            mem_bar(VST_VLD);
            CastInt32(repeatTimes, preg, sregInt32, subReg, dstInt32Ptr1, dstInt64Ptr1);
        }
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::ComputOut2(
    uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize)
{
    if (!tilingData_->needTranspose) {
        ComputOutNo2(arNum, loopGm, dstUbInt32, xUbSize);
        return;
    }
    if (arNum > 0) {
        int32_t alignInt32 = ((arNum + 8 - 1) / 8) * 8;
        TransComputeOut2(alignInt32, arNum, dstUbInt32);
        inQueX_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(xUbSize);
        LocalTensor<T2> outUbSize = inQueX_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T2>();
        CopyOutTrans(arNum, loopGm, outUbSize);
        inQueX_.FreeTensor(outUbSize);
    } else {
        inQueX_.FreeTensor(xUbSize);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::ComputOut3(
    uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize)
{
    if (!tilingData_->needTranspose) {
        ComputOutNo3(arNum, loopGm, dstUbInt32, xUbSize);
        return;
    }
    if (arNum > 0) {
        int32_t alignInt32 = ((arNum + 8 - 1) / 8) * 8;
        TransComputeOut3(alignInt32, arNum, dstUbInt32);
        inQueX_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(xUbSize);
        LocalTensor<T2> outUbSize = inQueX_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T2>();
        CopyOutTrans(arNum, loopGm, outUbSize);
        inQueX_.FreeTensor(outUbSize);
    } else {
        inQueX_.FreeTensor(xUbSize);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::ComputOut4(
    uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize)
{
    if (!tilingData_->needTranspose) {
        ComputOutNo4(arNum, loopGm, dstUbInt32, xUbSize);
        return;
    }
    if (arNum > 0) {
        int32_t alignInt32 = ((arNum + 8 - 1) / 8) * 8;
        TransComputeOut4(alignInt32, arNum, dstUbInt32);
        inQueX_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(xUbSize);
        LocalTensor<T2> outUbSize = inQueX_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T2>();
        CopyOutTrans(arNum, loopGm, outUbSize);
        inQueX_.FreeTensor(outUbSize);
    } else {
        inQueX_.FreeTensor(xUbSize);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::ComputOutNo2(
    uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize)
{
    if (arNum > 0) {
        int32_t alignInt32 = ((arNum + 8 - 1) / 8) * 8;
        NoTrans2(alignInt32, arNum, dstUbInt32);
        inQueX_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(xUbSize);
        LocalTensor<T2> outUbSize = inQueX_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T2>();
        CopyOut(arNum, loopGm, alignInt32, outUbSize);
        inQueX_.FreeTensor(outUbSize);
    } else {
        inQueX_.FreeTensor(xUbSize);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::ComputOutNo3(
    uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize)
{
    if (arNum > 0) {
        int32_t alignInt32 = ((arNum + 8 - 1) / 8) * 8;
        NoTrans3(alignInt32, arNum, dstUbInt32);
        inQueX_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(xUbSize);
        LocalTensor<T2> outUbSize = inQueX_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T2>();
        CopyOut(arNum, loopGm, alignInt32, outUbSize);
        inQueX_.FreeTensor(outUbSize);
    } else {
        inQueX_.FreeTensor(xUbSize);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::ComputOutNo4(
    uint64_t arNum, uint64_t loopGm, LocalTensor<int32_t>& dstUbInt32, LocalTensor<T2>& xUbSize)
{
    if (arNum > 0) {
        int32_t alignInt32 = ((arNum + 8 - 1) / 8) * 8;
        NoTrans4(alignInt32, arNum, dstUbInt32);
        inQueX_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(xUbSize);
        LocalTensor<T2> outUbSize = inQueX_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T2>();
        CopyOut(arNum, loopGm, alignInt32, outUbSize);
        inQueX_.FreeTensor(outUbSize);
    } else {
        inQueX_.FreeTensor(xUbSize);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::ComputeBefore(
    uint64_t& loopGm, int32_t loopCore, int32_t beforeNumOut, LocalTensor<uint32_t>& maskUbSize)
{
    LocalTensor<T2> dstUbInt64 = inQueX_.AllocTensor<T2>();
    inQueX_.EnQue<QuePosition::VECIN, QuePosition::VECCALC>(dstUbInt64);
    LocalTensor<T2> xUbSize = inQueX_.DeQue<QuePosition::VECIN, QuePosition::VECCALC, T2>();
    LocalTensor<int32_t> dstUbInt32 = xUbSize.template ReinterpretCast<int32_t>();
    uint64_t arNum = 0;
    if constexpr (sizeof(T1) == sizeof(int64_t)) {
        VfComputeIdsB64(loopCore, beforeNumOut, dstUbInt32, maskUbSize, arNum);
    } else {
        VfComputeIds(loopCore, beforeNumOut, dstUbInt32, maskUbSize, arNum);
    }
    loopGm = loopGm + arNum;
    if constexpr (TILING_KEY == DIM2) {
        ComputOut2(arNum, loopGm, dstUbInt32, xUbSize);
    } else if constexpr (TILING_KEY == DIM3) {
        ComputOut3(arNum, loopGm, dstUbInt32, xUbSize);
    } else if constexpr (TILING_KEY == DIM4) {
        ComputOut4(arNum, loopGm, dstUbInt32, xUbSize);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroFullLoadBase<T1, T2, TILING_KEY>::BaseProcess()
{
    LocalTensor<uint32_t> addUbSize = addUb_.template Get<uint32_t>();
    LocalTensor<uint32_t> maskUbSize = maskUb_.template Get<uint32_t>();
    VfCleanUb(addUbSize);
    CopyInBefore(
        tilingData_->loopNumPerCore, tilingData_->ubFactorNum, tilingData_->loopTailPerCore, addUbSize, maskUbSize);
    uint64_t loopGm = 0;
    for (int32_t loopCore = 0; loopCore < tilingData_->loopNumO; loopCore++) {
        ComputeBefore(loopGm, loopCore, tilingData_->beforeNumO, maskUbSize);
    }
    if (tilingData_->loopTailO != 0) {
        ComputeBefore(loopGm, tilingData_->loopNumO, tilingData_->loopTailO, maskUbSize);
    }
}
} // namespace NonZero
#endif
