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
 * \file unique_consecutive_helper.h
 * \brief unique consecutive helper
 */
#ifndef UNIQUE_CONSECUTIVE_HELPER_H
#define UNIQUE_CONSECUTIVE_HELPER_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "unique_consecutive_constant.h"

using namespace AscendC;

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {
};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {
};

__aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    return (y > 0) ? (x + y - 1) / y : 0;
}

template <typename T>
__aicore__ inline constexpr uint32_t GetVLEleNums()
{
    return GetVecLen() / sizeof(T);
}

template <typename T>
__aicore__ inline void Copy2GmEx(const GlobalTensor<T>& dst, const LocalTensor<T>& src, int64_t burst, int64_t copyNums,
                                 int64_t srcStride, int64_t dstStride)
{
    if (copyNums > 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = burst;
        dataCopyParams.blockLen = copyNums * sizeof(T);
        dataCopyParams.srcStride = srcStride;
        dataCopyParams.dstStride = dstStride;
        DataCopyPad(dst, src, dataCopyParams);
    }
}

template <HardEvent ent>
__aicore__ inline void SimpleNativePipeSync()
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(ent));
    SetFlag<ent>(event);
    WaitFlag<ent>(event);
}

template <typename T>
__aicore__ inline void CollectAndCopy2Ub(__ubuf__ T* dstUbAddr, MicroAPI::RegTensor<T>& srcReg,
                                         MicroAPI::RegTensor<T>& tmpReg, MicroAPI::MaskReg& cmpMask,
                                         MicroAPI::UnalignReg& ureg)
{
    MicroAPI::GatherMask<T, MicroAPI::GatherMaskMode::STORE_REG>(tmpReg, srcReg, cmpMask);
    MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbAddr, tmpReg, ureg);
}

template <int IDX_INC_NUMS>
__aicore__ inline void CollectIdxWithUpdateAndCopy2Ub(__ubuf__ int32_t* dstUbAddr, MicroAPI::RegTensor<int32_t>& srcReg,
                                                      MicroAPI::RegTensor<int32_t>& tmpReg, MicroAPI::MaskReg& cmpMask,
                                                      MicroAPI::MaskReg& pregAll, MicroAPI::UnalignReg& ureg)
{
    CollectAndCopy2Ub<int32_t>(dstUbAddr, srcReg, tmpReg, cmpMask, ureg);
    MicroAPI::Adds(srcReg, srcReg, (int32_t)IDX_INC_NUMS, pregAll);
}

// process 64 elements only
template <typename T, typename T1, int REP_LENGTH>
static __aicore__ inline void VFCollectPostUniqueIdx(__ubuf__ int32_t* dstIdxAddr, __ubuf__ T* srcValueAddr,
                                                     int32_t startCount, uint32_t repeatTimes, uint32_t totalNums)
{
    MicroAPI::RegTensor<T> xPrev;
    MicroAPI::RegTensor<T> xNext;

    MicroAPI::RegTensor<int32_t> out;
    MicroAPI::RegTensor<int32_t> idx;

    MicroAPI::UnalignReg uregIn;
    MicroAPI::UnalignReg uregOut;

    MicroAPI::MaskReg cmpRet;
    MicroAPI::MaskReg pregLoop;
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    uint32_t sreg0 = totalNums;

    MicroAPI::Arange(idx, startCount);
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        pregLoop = MicroAPI::UpdateMask<T>(sreg0);
        auto curtSrcAddr = srcValueAddr + REP_LENGTH * i + 1;

        DataCopy(xPrev, srcValueAddr + REP_LENGTH * i);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(xNext, uregIn, curtSrcAddr);

        MicroAPI::Compare<T1, CMPMODE::NE>(cmpRet, (MicroAPI::RegTensor<T1>&)xPrev, (MicroAPI::RegTensor<T1>&)xNext, pregLoop);

        if constexpr (sizeof(T) == 1) {
            MicroAPI::MaskReg maskQ1;
            MicroAPI::MaskReg maskQ2;
            MicroAPI::MaskReg maskQ3;
            MicroAPI::MaskReg maskQ4;
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(maskQ2, cmpRet);
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::HIGHEST>(maskQ4, cmpRet);
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(maskQ1, maskQ2);
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::HIGHEST>(maskQ2, maskQ2);
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(maskQ3, maskQ4);
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::HIGHEST>(maskQ4, maskQ4);

            constexpr uint32_t idxIncNums = GetVLEleNums<int32_t>();
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, maskQ1, pregAll, uregOut);
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, maskQ2, pregAll, uregOut);
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, maskQ3, pregAll, uregOut);
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, maskQ4, pregAll, uregOut);
        } else if constexpr (sizeof(T) == 2) {
            MicroAPI::MaskReg maskLowHalf;
            MicroAPI::MaskReg maskHighHalf;
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(maskLowHalf, cmpRet);
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::HIGHEST>(maskHighHalf, cmpRet);

            constexpr uint32_t idxIncNums = GetVLEleNums<int32_t>();
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, maskLowHalf, pregAll, uregOut);
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, maskHighHalf, pregAll, uregOut);
        } else if constexpr (sizeof(T) == 4) {
            constexpr uint32_t idxIncNums = GetVLEleNums<int32_t>();
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, cmpRet, pregAll, uregOut);
        } else {
            AscendC::MicroAPI::MaskReg maskHalf;
            AscendC::MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpRet);
            constexpr uint32_t idxIncNums = GetVLEleNums<int64_t>();
            CollectIdxWithUpdateAndCopy2Ub<idxIncNums>(dstIdxAddr, idx, out, maskHalf, pregAll, uregOut);
        }
    }
    MicroAPI::DataCopyUnAlignPost(dstIdxAddr, uregOut);
}

// support B8~B32 only, B64 use other implement
template <typename T, typename T1, int REP_LENGTH>
static __aicore__ inline void VFCollectPostUniqueValue(__ubuf__ T* dstValueAddr, __ubuf__ T* srcValueAddr,
                                                       uint32_t repeatTimes, uint32_t totalNums)
{
    MicroAPI::RegTensor<T> xPrev;
    MicroAPI::RegTensor<T> xNext;
    MicroAPI::RegTensor<T> out;

    MicroAPI::UnalignReg uregIn;
    MicroAPI::UnalignReg uregOut;

    MicroAPI::MaskReg cmpRet;
    MicroAPI::MaskReg pregLoop;

    uint32_t sreg0 = totalNums;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        pregLoop = MicroAPI::UpdateMask<T>(sreg0);
        auto curtSrcAddr = srcValueAddr + REP_LENGTH * i + 1;

        DataCopy(xPrev, srcValueAddr + REP_LENGTH * i);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(xNext, uregIn, curtSrcAddr);

        MicroAPI::Compare<T1, CMPMODE::NE>(cmpRet, (MicroAPI::RegTensor<T1>&)xPrev, (MicroAPI::RegTensor<T1>&)xNext, pregLoop);

        CollectAndCopy2Ub<T>(dstValueAddr, xPrev, out, cmpRet, uregOut);
    }
    MicroAPI::DataCopyUnAlignPost(dstValueAddr, uregOut);
}

template <int REP_LENGTH>
static __aicore__ inline void VFCollectPostUniqueValueB64(__ubuf__ int32_t* dstValueAddr,
                                                          __ubuf__ int32_t* srcValueAddr, uint32_t repeatTimes,
                                                          uint32_t totalNums)
{
    MicroAPI::RegTensor<int32_t> xPrev;
    MicroAPI::RegTensor<int32_t> xNext;
    MicroAPI::RegTensor<int32_t> out;

    MicroAPI::UnalignReg uregIn;
    MicroAPI::UnalignReg uregOut;

    MicroAPI::MaskReg cmpRet;
    MicroAPI::MaskReg pregLoop;

    MicroAPI::MaskReg maskEven;
    MicroAPI::MaskReg maskOdd;

    uint32_t sreg0 = totalNums;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        pregLoop = MicroAPI::UpdateMask<int32_t>(sreg0);
        auto curtSrcAddr = srcValueAddr + REP_LENGTH * i + 2;
        DataCopy(xPrev, srcValueAddr + REP_LENGTH * i);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(xNext, uregIn, curtSrcAddr);
        MicroAPI::Compare<int32_t, CMPMODE::NE>(cmpRet, xPrev, xNext, pregLoop);

        MicroAPI::MaskDeInterleave<int32_t>(maskEven, maskOdd, cmpRet, cmpRet);
        MicroAPI::MaskOr(cmpRet, maskEven, maskOdd, pregLoop);
        MicroAPI::MaskInterleave<int32_t>(maskEven, maskOdd, cmpRet, cmpRet);
        CollectAndCopy2Ub<int32_t>(dstValueAddr, xPrev, out, maskEven, uregOut);
    }
    MicroAPI::DataCopyUnAlignPost(dstValueAddr, uregOut);
}

/*
    x: tensor, length = num
    mask[i] = x[i] != x[i+1], i < num - 1
    ret = mask.nonzero()
*/
template <typename VALUE_TYPE, typename INPUT_TYPE, bool IS_TAIL>
__aicore__ inline void CollectPostUniqueIdx(LocalTensor<int32_t>& dstIdx, LocalTensor<VALUE_TYPE>& srcValue,
                                            int32_t startCount, int32_t endCount, uint32_t nums, uint64_t& rsvdCnt, uint64_t position)
{
    // Set phy addr
    __local_mem__ VALUE_TYPE* srcValueAddr = (__local_mem__ VALUE_TYPE*)srcValue[0].GetPhyAddr();
    __local_mem__ int32_t* dstIdxAddr = (__local_mem__ int32_t*)dstIdx[position].GetPhyAddr();

    // Compute VF params
    constexpr uint32_t repNums = GetVLEleNums<VALUE_TYPE>();
    uint32_t totalNums = nums - 1;
    uint32_t repTimes = CEIL_DIV(totalNums, repNums);

    AscendC::VF_CALL<VFCollectPostUniqueIdx<VALUE_TYPE, INPUT_TYPE, repNums>>(dstIdxAddr, srcValueAddr, startCount, repTimes,
                                                                  totalNums);
    rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(int32_t);
    if constexpr (IS_TAIL) {
        dstIdx.SetValue(position + rsvdCnt, endCount);
        rsvdCnt += 1;
    }
}

/*
    x: tensor, length = num
    mask[i] = x[i] != x[i+1], i < num - 1
    ret = x[mask.nonzero()]
*/
template <typename VALUE_TYPE, typename INPUT_TYPE, bool IS_TAIL>
__aicore__ inline void CollectPostUniqueValue(LocalTensor<VALUE_TYPE>& dstValue, LocalTensor<VALUE_TYPE>& srcValue,
                                              uint32_t nums, uint64_t& rsvdCnt)
{
    if constexpr (is_same<VALUE_TYPE, int64_t>::value) {
        // Set phy addr
        __local_mem__ int32_t* srcValueAddr =
            (__local_mem__ int32_t*)srcValue[0].template ReinterpretCast<int32_t>().GetPhyAddr();
        __local_mem__ int32_t* dstValueAddr =
            (__local_mem__ int32_t*)dstValue[0].template ReinterpretCast<int32_t>().GetPhyAddr();

        // Compute VF params
        constexpr uint32_t repNums = GetVLEleNums<int32_t>();
        uint32_t totalNums = (nums - 1) * 2;
        uint32_t repTimes = CEIL_DIV(totalNums, repNums);
        AscendC::VF_CALL<VFCollectPostUniqueValueB64<repNums>>(dstValueAddr, srcValueAddr, repTimes, totalNums);
    } else {
        // Set phy addr
        __local_mem__ VALUE_TYPE* srcValueAddr = (__local_mem__ VALUE_TYPE*)srcValue[0].GetPhyAddr();
        __local_mem__ VALUE_TYPE* dstValueAddr = (__local_mem__ VALUE_TYPE*)dstValue[0].GetPhyAddr();

        // Compute VF params
        constexpr uint32_t repNums = GetVLEleNums<VALUE_TYPE>();
        uint32_t totalNums = nums - 1;
        uint32_t repTimes = CEIL_DIV(totalNums, repNums);
        AscendC::VF_CALL<VFCollectPostUniqueValue<VALUE_TYPE, INPUT_TYPE, repNums>>(dstValueAddr, srcValueAddr, repTimes,
                                                                        totalNums);
    }
    rsvdCnt = GetSpr<SpecialPurposeReg::AR>() / sizeof(VALUE_TYPE);
    if constexpr (IS_TAIL) {
        dstValue.SetValue(rsvdCnt, srcValue.GetValue(nums - 1));
        rsvdCnt += 1;
    }
}

template <int REP_LENGTH, typename T>
static __aicore__ inline void VFPostAdjDiff(__ubuf__ T* dstIdxAddr, __ubuf__ T* srcIdxAddr,
                                            uint32_t repeatTimes, uint32_t totalNums, uint16_t hasTail)
{
    MicroAPI::RegTensor<T> idxPrev;
    MicroAPI::RegTensor<T> idxNext;
    MicroAPI::RegTensor<T> out;

    MicroAPI::UnalignReg uregIn;
    MicroAPI::UnalignReg uregOut;

    MicroAPI::MaskReg pregLoop;

    uint32_t sreg0 = totalNums;
    uint32_t tailLen = totalNums % REP_LENGTH;

    auto curtDstAddr = dstIdxAddr + 1;

    // main block
    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        pregLoop = MicroAPI::UpdateMask<T>(sreg0);
        auto curtSrcAddr = srcIdxAddr + REP_LENGTH * i + 1;
        DataCopy(idxPrev, srcIdxAddr + REP_LENGTH * i);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(idxNext, uregIn, curtSrcAddr);
        MicroAPI::Sub(out, idxNext, idxPrev, pregLoop);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(curtDstAddr, out, uregOut,
                                                                                    REP_LENGTH);
    }

    // tail block
    for (uint16_t i = 0; i < (uint16_t)hasTail; ++i) {
        pregLoop = MicroAPI::UpdateMask<T>(sreg0);
        auto curtSrcAddr = srcIdxAddr + REP_LENGTH * repeatTimes + 1;
        DataCopy(idxPrev, srcIdxAddr + REP_LENGTH * repeatTimes);
        MicroAPI::DataCopyUnAlignPre(uregIn, curtSrcAddr);
        MicroAPI::DataCopyUnAlign(idxNext, uregIn, curtSrcAddr);
        MicroAPI::Sub(out, idxNext, idxPrev, pregLoop);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(curtDstAddr, out, uregOut, tailLen);
    }
    MicroAPI::DataCopyUnAlignPost(curtDstAddr, uregOut, 0);
}

/*
    srcIdx: idx tensor which start with 1, length = num
    dstIdx[0] = firstValue
    dstIdx[i] = dstIdx[i] - dstIdx[i - 1], for i > 0
    consider compute dstIdx[0] in VF......
*/
template <typename T>
__aicore__ inline void PostAdjDiff(LocalTensor<T>& dstIdx, LocalTensor<T>& srcIdx, T firstValue,
                                   uint32_t nums, uint64_t position)
{
    // Set phy addr
    __local_mem__ T* srcIdxAddr = (__local_mem__ T*)srcIdx[0].GetPhyAddr();
    __local_mem__ T* dstIdxAddr = (__local_mem__ T*)dstIdx[position].GetPhyAddr();

    // Compute VF params
    constexpr uint32_t repNums = GetVLEleNums<T>();
    uint32_t totalNums = nums - 1;
    uint32_t repTimes = totalNums / repNums;
    uint16_t hasTail = (totalNums % repNums == 0) ? 0 : 1;

    AscendC::VF_CALL<VFPostAdjDiff<repNums, T>>(dstIdxAddr, srcIdxAddr, repTimes, totalNums, hasTail);
    dstIdx.SetValue(position, firstValue);
}

template <int REP_LENGTH>
static __aicore__ inline void VFCastAndAddsOffsets(__ubuf__ int64_t* dstIdxAddr, __ubuf__ int32_t* srcIdxAddr,
                                            uint32_t repeatTimes, uint32_t totalNums, int64_t offset)
{
    MicroAPI::RegTensor<int64_t> srcReg;
    MicroAPI::RegTensor<int64_t> dstReg;

    MicroAPI::MaskReg pregLoop;

    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        pregLoop = MicroAPI::UpdateMask<int64_t>(totalNums);
        MicroAPI::AddrReg srcOffset = MicroAPI::CreateAddrReg<int32_t>(i,REP_LENGTH);
        MicroAPI::AddrReg dstOffset = MicroAPI::CreateAddrReg<int64_t>(i,REP_LENGTH);
        MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_UNPACK_B32>((MicroAPI::RegTensor<int32_t>&)srcReg, srcIdxAddr, srcOffset);
        MicroAPI::Adds(dstReg, srcReg, offset, pregLoop);
        MicroAPI::DataCopy(dstIdxAddr, dstReg, dstOffset, pregLoop);
    }
}

__aicore__ inline void CastAndAddsOffsets(LocalTensor<int64_t>& dstIdx, LocalTensor<int32_t>& srcIdx,  uint32_t nums, uint64_t position, int64_t offset)
{
    // Set phy addr
    __local_mem__ int32_t* srcIdxAddr = (__local_mem__ int32_t*)srcIdx[position].GetPhyAddr();
    __local_mem__ int64_t* dstIdxAddr = (__local_mem__ int64_t*)dstIdx[0].GetPhyAddr();

    // Compute VF params
    constexpr uint32_t repNums = GetVLEleNums<int64_t>();
    uint32_t repTimes = CEIL_DIV(nums, repNums);

    AscendC::VF_CALL<VFCastAndAddsOffsets<repNums>>(dstIdxAddr, srcIdxAddr, repTimes, nums, offset);
}
#endif  // UNIQUE_CONSECUTIVE_HELPER_H