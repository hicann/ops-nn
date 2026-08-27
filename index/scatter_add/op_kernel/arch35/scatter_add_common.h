/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_add_common.h
 * \brief common fun of scatter_add
 */

#ifndef SCATTER_ADD_COMMON_IMPL_H
#define SCATTER_ADD_COMMON_IMPL_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace ScatterAddCommon {
using namespace AscendC;
constexpr uint32_t VECTOR_LENGTH = platform::GetVRegSize();
constexpr uint32_t VL_B32 = VECTOR_LENGTH / sizeof(uint32_t);
constexpr uint32_t VF_B32 = VECTOR_LENGTH / sizeof(int32_t);
constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint64_t SORT_PAD_NUM = 2;
constexpr uint64_t HASH_BUCKER_BUFFER_SIZE = 128 * sizeof(float);
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint32_t TWO = 2;
constexpr uint32_t THREE = 3;
constexpr uint32_t FOUR = 4;
constexpr uint32_t CAST_0 = 0;
constexpr uint32_t CAST_1 = 1;
constexpr uint32_t CAST_2 = 2;
constexpr uint32_t CAST_3 = 3;
constexpr uint32_t CAST_4 = 4;
constexpr uint32_t CAST_5 = 5;
constexpr int64_t VFLEN_INT64 = platform::GetVRegSize() / sizeof(int64_t);
constexpr int64_t VFLEN_INT32 = platform::GetVRegSize() / sizeof(int32_t);
constexpr int64_t VFLEN_INT16 = platform::GetVRegSize() / sizeof(int16_t);
constexpr int64_t VFLEN_INT16HALF = platform::GetVRegSize() / sizeof(int16_t) / TWO;
constexpr int64_t VFLEN_UINT8 = platform::GetVRegSize() / sizeof(uint8_t);
constexpr int64_t VFLEN_UINT8HALFHALF = platform::GetVRegSize() / sizeof(uint8_t) / FOUR;
constexpr uint32_t ADD = 0;
constexpr uint32_t SUB = 1;

constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};
static constexpr Reg::CastTrait castTraitU82Int32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

typedef struct {
    uint16_t segCount; // 记录每次拿到的局部排序后索引的重复次数
    uint32_t outGmIndex;
    uint32_t xPerRowNum;
    __ubuf__ uint32_t* sortedIdxAddr;
} updateAddParams;

template <typename T>
__simd_vf__ inline void CastToInt32Vf(__ubuf__ T* srcAddr, __ubuf__ int32_t* dstAddr, uint16_t loopTimes,
                                      uint32_t dataLen)
{
    Reg::RegTensor<T> srcValue;
    Reg::MaskReg preg;
    uint32_t sregMask = dataLen;
    for (uint16_t i = 0; i < loopTimes; i++) {
        auto dstReg = Reg::CreateAddrReg<int32_t>(i, static_cast<uint16_t>(VL_B32));
        auto srcReg = Reg::CreateAddrReg<T>(i, static_cast<uint16_t>(VL_B32));
        preg = Reg::UpdateMask<int32_t>(sregMask);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK4_B8>(srcValue, srcAddr, srcReg);
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_NORM>(dstAddr, (Reg::RegTensor<int32_t>&)srcValue, dstReg, preg);
    }
}

template <typename T>
__aicore__ inline void CastToInt32(LocalTensor<int32_t>& dstLocal, LocalTensor<T>& srcLocal, uint32_t dataLen)
{
    __ubuf__ T* srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();
    __ubuf__ int32_t* dstAddr = (__ubuf__ int32_t*)dstLocal.GetPhyAddr();

    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);

    CastToInt32Vf(srcAddr, dstAddr, loopTimes, dataLen);
}

template <typename T>
__simd_vf__ inline void NegateUpdateVf(__ubuf__ T* updatesAddr, uint32_t loopSize, uint16_t loopTimes, uint32_t dataLen)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> updatesValue;
        Reg::RegTensor<bfloat16_t> scalarReg;
        Reg::RegTensor<bfloat16_t> dstReg;
        Reg::MaskReg maskReg;
        uint32_t count = dataLen;
        bfloat16_t scalarValue = -1;
        Reg::Duplicate(scalarReg, scalarValue);
        for (uint16_t j = 0; j < loopTimes; j++) {
            maskReg = Reg::UpdateMask<T>(count);
            Reg::LoadAlign(updatesValue, updatesAddr + loopSize * j);
            Reg::Mul(dstReg, updatesValue, scalarReg, maskReg);
            Reg::StoreAlign(updatesAddr + loopSize * j, dstReg, maskReg);
        }
    } else {
        Reg::RegTensor<T> updatesValue;
        Reg::RegTensor<T> negValue;
        Reg::MaskReg maskReg;
        uint32_t count = dataLen;
        for (uint16_t j = 0; j < loopTimes; j++) {
            maskReg = Reg::UpdateMask<T>(count);
            Reg::LoadAlign(updatesValue, updatesAddr + loopSize * j);
            Reg::Neg(negValue, updatesValue, maskReg);
            Reg::StoreAlign(updatesAddr + loopSize * j, negValue, maskReg);
        }
    }
}

template <typename T>
__aicore__ inline void NegateUpdate(LocalTensor<T>& updatesLocal, uint32_t dataLen)
{
    if constexpr (IsSameType<T, uint8_t>::value) {
        return;
    }

    __ubuf__ T* updatesAddr = (__ubuf__ T*)updatesLocal.GetPhyAddr();
    uint32_t loopSize = platform::GetVRegSize() / sizeof(T);
    uint16_t loopTimes = ops::CeilDiv(dataLen, loopSize);

    NegateUpdateVf(updatesAddr, loopSize, loopTimes, dataLen);
}

template <typename T>
__simd_vf__ inline void CastToOriginVf(__ubuf__ int32_t* srcAddr, __ubuf__ T* dstAddr, uint32_t dataLen,
                                       uint16_t loopTimes, uint16_t stride)
{
    Reg::RegTensor<int32_t> srcValue;
    Reg::MaskReg preg;
    uint32_t sregMask = dataLen;
    for (uint16_t i = 0; i < loopTimes; i++) {
        auto dstReg = Reg::CreateAddrReg<T>(i, static_cast<uint16_t>(VL_B32));
        auto srcReg = Reg::CreateAddrReg<int32_t>(i, static_cast<uint16_t>(VL_B32));
        preg = Reg::UpdateMask<int32_t>(sregMask);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>(srcValue, srcAddr, srcReg);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK4_B32>(dstAddr, (Reg::RegTensor<T>&)srcValue, dstReg, preg);
    }
}

template <typename T>
__aicore__ inline void CastToOrigin(LocalTensor<T>& dstLocal, LocalTensor<int32_t>& srcLocal, uint32_t dataLen)
{
    __ubuf__ int32_t* srcAddr = (__ubuf__ int32_t*)srcLocal.GetPhyAddr();
    __ubuf__ T* dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();

    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);
    uint16_t stride = static_cast<uint16_t>(VL_B32);

    CastToOriginVf(srcAddr, dstAddr, dataLen, loopTimes, stride);
}

template <typename IDX_T, typename CAST_T, uint32_t castType>
__aicore__ inline void IndicesSortCast(LocalTensor<IDX_T> indicesLocal, LocalTensor<CAST_T> indicesCastLocal,
                                       LocalTensor<int32_t> indicesCastTmpLocal, uint32_t indicesCount)
{
    if constexpr (castType == CAST_4) { // int32 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(255), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               indicesCount);
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_3) { // int64 Cast int16
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_5) { // int64 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(255), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               indicesCount);
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else { // CAST_1 + CAST_2, int32 Cast int16 + int64 Cast int32
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
    }
}

template <typename T, uint32_t scatterOp>
__aicore__ inline void BroadcastUpdatesScalar(LocalTensor<T> updatesLocal, GlobalTensor<T> updatesGm, int32_t count)
{
    T updatesValue = updatesGm.GetValue(0);
    if constexpr (scatterOp == SUB) {
        updatesValue = -updatesValue;
    }
    auto vWaitSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(vWaitSEventID);
    WaitFlag<HardEvent::S_V>(vWaitSEventID);
    Duplicate(updatesLocal, updatesValue, count);
}

template <typename IDX_T>
__simd_callee__ inline void ComputeUniqueIdNumInt64(__ubuf__ IDX_T* indicesAddr, __ubuf__ int32_t* uniqueIdCountsAddr,
                                                    uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::Reg::RegTensor<int32_t> orderReg, selReg;
    AscendC::Reg::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::Reg::MaskReg cmpMask, maskReg, maskHalf;
    AscendC::Reg::UnalignRegForLoad u0;
    AscendC::Reg::UnalignRegForStore uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::Reg::Arange(orderReg, i * VFLEN_INT64);
        maskReg = AscendC::Reg::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_INT64;
        AscendC::Reg::LoadAlign(sortedIdxReg, startAddr);
        AscendC::Reg::LoadUnAlignPre(u0, startAddr - 1);
        AscendC::Reg::LoadUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::Reg::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::Reg::Pack<AscendC::Reg::HighLowPart::LOWEST>(maskHalf, cmpMask);
        AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg, orderReg, maskHalf);
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg,
                                                                                         uOut);
    }
    AscendC::Reg::StoreUnAlignPost(uniqueIdCountsAddr, uOut);
}

template <typename IDX_T>
__simd_callee__ inline void ComputeUniqueIdNumInt32(__ubuf__ IDX_T* indicesAddr, __ubuf__ int32_t* uniqueIdCountsAddr,
                                                    uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::Reg::RegTensor<int32_t> orderReg, selReg;
    AscendC::Reg::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::Reg::MaskReg cmpMask, maskReg;
    AscendC::Reg::UnalignRegForLoad u0;
    AscendC::Reg::UnalignRegForStore uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::Reg::Arange(orderReg, i * VFLEN_INT32);
        maskReg = AscendC::Reg::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_INT32;
        AscendC::Reg::LoadAlign(sortedIdxReg, startAddr);
        AscendC::Reg::LoadUnAlignPre(u0, startAddr - 1);
        AscendC::Reg::LoadUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::Reg::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg, orderReg, cmpMask);
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg,
                                                                                         uOut);
    }
    AscendC::Reg::StoreUnAlignPost(uniqueIdCountsAddr, uOut);
}

template <typename IDX_T>
__simd_callee__ inline void ComputeUniqueIdNumInt16(__ubuf__ IDX_T* indicesAddr, __ubuf__ int32_t* uniqueIdCountsAddr,
                                                    uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::Reg::RegTensor<int32_t> orderReg, orderReg2, selReg, selReg2;
    AscendC::Reg::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::Reg::MaskReg cmpMask, maskReg, maskDouble1, maskDouble2;
    AscendC::Reg::UnalignRegForLoad u0;
    AscendC::Reg::UnalignRegForStore uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::Reg::Arange(orderReg, i * VFLEN_INT16);
        AscendC::Reg::Arange(orderReg2, i * VFLEN_INT16 + VFLEN_INT16HALF);
        maskReg = AscendC::Reg::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_INT16;
        AscendC::Reg::LoadAlign(sortedIdxReg, startAddr);
        AscendC::Reg::LoadUnAlignPre(u0, startAddr - 1);
        AscendC::Reg::LoadUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::Reg::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(maskDouble1, cmpMask);
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskDouble2, cmpMask);
        AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg, orderReg, maskDouble1);
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg,
                                                                                         uOut);
        AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskDouble2);
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2,
                                                                                         uOut);
    }
    AscendC::Reg::StoreUnAlignPost(uniqueIdCountsAddr, uOut);
}

template <typename IDX_T>
__simd_callee__ inline void ComputeUniqueIdNumUint8(__ubuf__ IDX_T* indicesAddr, __ubuf__ int32_t* uniqueIdCountsAddr,
                                                    uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::Reg::RegTensor<int32_t> orderReg, orderReg2, orderReg3, orderReg4;
    AscendC::Reg::RegTensor<int32_t> selReg, selReg2, selReg3, selReg4;
    AscendC::Reg::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::Reg::MaskReg cmpMask, maskReg, maskFour1, maskFour2, maskFour3, maskFour4;
    AscendC::Reg::UnalignRegForLoad u0;
    AscendC::Reg::UnalignRegForStore uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::Reg::Arange(orderReg, i * VFLEN_UINT8);
        AscendC::Reg::Arange(orderReg2, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF);
        AscendC::Reg::Arange(orderReg3, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * TWO);
        AscendC::Reg::Arange(orderReg4, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * THREE);
        maskReg = AscendC::Reg::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_UINT8;
        AscendC::Reg::LoadAlign(sortedIdxReg, startAddr);
        AscendC::Reg::LoadUnAlignPre(u0, startAddr - 1);
        AscendC::Reg::LoadUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::Reg::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(maskFour3, cmpMask);
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskFour4, cmpMask);
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(maskFour1, maskFour3);
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskFour2, maskFour3);
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(maskFour3, maskFour4);
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskFour4, maskFour4);
        AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg, orderReg, maskFour1);
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg,
                                                                                         uOut);
        AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskFour2);
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2,
                                                                                         uOut);
        AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg3, orderReg3, maskFour3);
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg3,
                                                                                         uOut);
        AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg4, orderReg4, maskFour4);
        AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg4,
                                                                                         uOut);
    }
    AscendC::Reg::StoreUnAlignPost(uniqueIdCountsAddr, uOut);
}

template <typename IDX_T>
__simd_vf__ inline void ComputeUniqueIdNumVf(__ubuf__ IDX_T* indicesAddr, __ubuf__ int32_t* uniqueIdCountsAddr,
                                             int64_t dataLen, uint16_t loopCnt)
{
    AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    if constexpr (std::is_same<int64_t, IDX_T>::value) {
        ComputeUniqueIdNumInt64<IDX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
    } else if constexpr (std::is_same<int32_t, IDX_T>::value) {
        ComputeUniqueIdNumInt32<IDX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
    } else if constexpr (std::is_same<int16_t, IDX_T>::value) {
        ComputeUniqueIdNumInt16<IDX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
    } else { // uint8
        ComputeUniqueIdNumUint8<IDX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
    }
}

template <typename IDX_T>
__aicore__ inline uint32_t ComputeUniqueIdNum(LocalTensor<IDX_T> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal,
                                              int64_t dataLen)
{
    __ubuf__ IDX_T* indicesAddr = (__ubuf__ IDX_T*)indicesLocal[(UB_AGLIN_VALUE / sizeof(IDX_T))].GetPhyAddr();
    __ubuf__ int32_t* uniqueIdCountsAddr = (__ubuf__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

    constexpr int64_t vfLen = platform::GetVRegSize() / sizeof(IDX_T);
    uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);

    ComputeUniqueIdNumVf(indicesAddr, uniqueIdCountsAddr, dataLen, loopCnt);
    uint32_t uniqueIdNum = ((AscendC::Reg::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
    return uniqueIdNum;
}

template <typename IDX_T>
__aicore__ inline uint32_t SortAndComputeUniqueIdx(int64_t rowLen, LocalTensor<IDX_T> indicesSrcLocal,
                                                   LocalTensor<IDX_T> sortIndicesLocal,
                                                   LocalTensor<int32_t> uniqueIdxLocal,
                                                   LocalTensor<uint32_t> updatesOriginIdexLocal)
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    int64_t shiftOffset = UB_AGLIN_VALUE / sizeof(IDX_T);
    LocalTensor<IDX_T> shiftSortLocal = sortIndicesLocal[shiftOffset];
    AscendC::Sort<IDX_T, true, sortConfig>(shiftSortLocal, updatesOriginIdexLocal, indicesSrcLocal,
                                           static_cast<uint32_t>(rowLen));
    Duplicate(sortIndicesLocal, (IDX_T)-1, shiftOffset);
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    shiftSortLocal(rowLen) = -1;

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    return ComputeUniqueIdNum(sortIndicesLocal, uniqueIdxLocal, rowLen);
}

__simd_vf__ inline void ComputeUniqueIdTimesVf(__ubuf__ int32_t* noDupResAddr, uint32_t counterStatFre,
                                               uint16_t loopCntStatFre)
{
    AscendC::Reg::RegTensor<int32_t> beginReg;
    AscendC::Reg::RegTensor<int32_t> endReg;
    AscendC::Reg::RegTensor<int32_t> subReg;
    AscendC::Reg::MaskReg maskRegUpdate;
    AscendC::Reg::UnalignRegForLoad u0;
    for (uint16_t i = 0; i < loopCntStatFre; i++) {
        auto noDupResAddrUpdate = noDupResAddr + i * VF_B32 + 1;
        maskRegUpdate = AscendC::Reg::UpdateMask<int32_t>(counterStatFre);
        AscendC::Reg::LoadAlign(beginReg, noDupResAddr + i * VF_B32);
        AscendC::Reg::LoadUnAlignPre(u0, noDupResAddrUpdate);
        AscendC::Reg::LoadUnAlign<int32_t>(endReg, u0, noDupResAddrUpdate);
        AscendC::Reg::Sub(subReg, endReg, beginReg, maskRegUpdate);
        AscendC::Reg::StoreAlign(noDupResAddr + i * VF_B32, subReg, maskRegUpdate);
    }
}

__aicore__ inline void ComputeUniqueIdTimes(LocalTensor<int32_t>& noDupRes, uint32_t& arNum)
{
    __ubuf__ int32_t* noDupResAddr = (__ubuf__ int32_t*)noDupRes.GetPhyAddr();
    uint16_t loopCntStatFre = (arNum + VF_B32 - 1) / VF_B32;
    uint32_t counterStatFre = arNum;

    ComputeUniqueIdTimesVf(noDupResAddr, counterStatFre, loopCntStatFre);

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst,
                              uint32_t copyLen, uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;               // 连续传输块个数
    dataCoptExtParams.blockLen = copyLen * sizeof(T);    // 每块大小
    dataCoptExtParams.srcStride = srcStride * sizeof(T); // 源地址相邻块间隔
    dataCoptExtParams.dstStride = dstStride * sizeof(T); // 目的地址相邻块间隔
    DataCopyPad(dstLocal, srcGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T>
__aicore__ inline void CopyOut(GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst,
                               uint32_t copyLen, uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(dstGm[offset], srcLocal, dataCoptExtParams);
}

} // namespace ScatterAddCommon
#endif // SCATTER_ADD_COMMON_IMPL_H
