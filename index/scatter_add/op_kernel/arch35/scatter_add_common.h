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
static constexpr MicroAPI::CastTrait castTraitU82Int32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

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

template <typename T>
__aicore__ inline void CastToInt32(LocalTensor<int32_t>& dstLocal, LocalTensor<T>& srcLocal, uint32_t dataLen)
{
    __local_mem__ T* srcAddr = (__local_mem__ T*)srcLocal.GetPhyAddr();
    __local_mem__ int32_t* dstAddr = (__local_mem__ int32_t*)dstLocal.GetPhyAddr();

    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> srcValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = dataLen;
        for (uint16_t i = 0; i < loopTimes; i++) {
            auto dstReg = MicroAPI::CreateAddrReg<int32_t>(i, static_cast<uint16_t>(VL_B32));
            auto srcReg = MicroAPI::CreateAddrReg<T>(i, static_cast<uint16_t>(VL_B32));
            preg = MicroAPI::UpdateMask<int32_t>(sregMask);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcValue, srcAddr, srcReg);
            MicroAPI::DataCopy<int32_t, MicroAPI::StoreDist::DIST_NORM>(
                dstAddr, (MicroAPI::RegTensor<int32_t>&)srcValue, dstReg, preg);
        }
    }
}

template<typename T>
__aicore__ inline void NegateUpdate(LocalTensor<T>& updatesLocal, uint32_t dataLen)
{
    if constexpr (IsSameType<T, uint8_t>::value) {
        return ;
    }

    __local_mem__ T* updatesAddr = (__local_mem__ T*)updatesLocal.GetPhyAddr();
    uint32_t loopSize = platform::GetVRegSize() / sizeof(T);
    uint16_t loopTimes = ops::CeilDiv(dataLen, loopSize);  
             
    if constexpr (IsSameType<T, bfloat16_t>::value) { 
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<bfloat16_t> updatesValue;
            MicroAPI::RegTensor<bfloat16_t> scalarReg;
            MicroAPI::RegTensor<bfloat16_t> dstReg;
            MicroAPI::MaskReg maskReg;
            uint32_t count = dataLen;
            bfloat16_t scalarValue = -1;
            MicroAPI::Duplicate(scalarReg, scalarValue);
            for (uint16_t j = 0; j < loopTimes; j++) {
                maskReg = MicroAPI::UpdateMask<T>(count);
                MicroAPI::DataCopy(updatesValue, updatesAddr + loopSize * j);
                MicroAPI::Mul(dstReg, updatesValue, scalarReg, maskReg);
                MicroAPI::DataCopy(updatesAddr + loopSize * j, dstReg, maskReg);
            }
        }
    } else {  
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> updatesValue;
            MicroAPI::RegTensor<T> negValue;
            MicroAPI::MaskReg maskReg;
            uint32_t count = dataLen;
            for (uint16_t j = 0; j < loopTimes; j++) {
                maskReg = MicroAPI::UpdateMask<T>(count);
                MicroAPI::DataCopy(updatesValue, updatesAddr + loopSize * j);
                MicroAPI::Neg(negValue, updatesValue, maskReg);
                MicroAPI::DataCopy(updatesAddr + loopSize * j, negValue, maskReg);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CastToOrigin(LocalTensor<T>& dstLocal, LocalTensor<int32_t>& srcLocal, uint32_t dataLen)
{
    __local_mem__ int32_t* srcAddr = (__local_mem__ int32_t*)srcLocal.GetPhyAddr();
    __local_mem__ T* dstAddr = (__local_mem__ T*)dstLocal.GetPhyAddr();

    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);
    uint16_t stride = static_cast<uint16_t>(VL_B32);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> srcValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = dataLen;
        for (uint16_t i = 0; i < loopTimes; i++) {
            auto dstReg = MicroAPI::CreateAddrReg<T>(i, static_cast<uint16_t>(VL_B32));
            auto srcReg = MicroAPI::CreateAddrReg<int32_t>(i, static_cast<uint16_t>(VL_B32));
            preg = MicroAPI::UpdateMask<int32_t>(sregMask);
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(srcValue, srcAddr, srcReg);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK4_B32>(
                dstAddr, (MicroAPI::RegTensor<T>&)srcValue, dstReg, preg);
        }
    }
}

template <typename IDX_T, typename CAST_T, uint32_t castType>
__aicore__ inline void IndicesSortCast(LocalTensor<IDX_T> indicesLocal, LocalTensor<CAST_T> indicesCastLocal,
                                                LocalTensor<int32_t> indicesCastTmpLocal, uint32_t indicesCount)
{
    if constexpr (castType == CAST_4) {  // int32 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(255), SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_3) {  // int64 Cast int16
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_5) {  // int64 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(255), SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else {    // CAST_1 + CAST_2, int32 Cast int16 + int64 Cast int32
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

template<typename IDX_T>
__aicore__ void ComputeUniqueIdNumInt64(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
    AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskHalf;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT64);
        maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_INT64;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskHalf);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            uniqueIdCountsAddr, selReg, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template<typename IDX_T>
__aicore__ void ComputeUniqueIdNumInt32(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
    AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT32);
        maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_INT32;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, cmpMask);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            uniqueIdCountsAddr, selReg, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template<typename IDX_T>
__aicore__ void ComputeUniqueIdNumInt16(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, orderReg2, selReg, selReg2;
    AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskDouble1, maskDouble2;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT16);
        AscendC::MicroAPI::Arange(orderReg2, i * VFLEN_INT16 + VFLEN_INT16HALF);
        maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_INT16;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskDouble1, cmpMask);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskDouble2, cmpMask);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskDouble1);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskDouble2);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg, uOut);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template<typename IDX_T>
__aicore__ void ComputeUniqueIdNumUint8(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, orderReg2, orderReg3, orderReg4;
    AscendC::MicroAPI::RegTensor<int32_t> selReg, selReg2, selReg3, selReg4;
    AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskFour1, maskFour2, maskFour3, maskFour4;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_UINT8);
        AscendC::MicroAPI::Arange(orderReg2, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF);
        AscendC::MicroAPI::Arange(orderReg3, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * TWO);
        AscendC::MicroAPI::Arange(orderReg4, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * THREE);
        maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_UINT8;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour3, cmpMask);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour4, cmpMask);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour1, maskFour3);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour2, maskFour3);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour3, maskFour4);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour4, maskFour4);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskFour1);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskFour2);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg3, orderReg3, maskFour3);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg4, orderReg4, maskFour4);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg, uOut);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2, uOut);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg3, uOut);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg4, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template<typename IDX_T>
__aicore__ uint32_t ComputeUniqueIdNum(LocalTensor<IDX_T> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
{
    __local_mem__ IDX_T* indicesAddr = (__local_mem__ IDX_T*)indicesLocal[(UB_AGLIN_VALUE / sizeof(IDX_T))].GetPhyAddr();
    __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

    constexpr int64_t vfLen = platform::GetVRegSize() / sizeof(IDX_T);
    uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

        if constexpr (std::is_same<int64_t, IDX_T>::value) {
            ComputeUniqueIdNumInt64<IDX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
        } else if constexpr (std::is_same<int32_t, IDX_T>::value) {
            ComputeUniqueIdNumInt32<IDX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
        } else if constexpr (std::is_same<int16_t, IDX_T>::value) {
            ComputeUniqueIdNumInt16<IDX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
        } else {  // uint8
            ComputeUniqueIdNumUint8<IDX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
        }
    }
    uint32_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
    return uniqueIdNum;
}

template<typename IDX_T>
__aicore__ uint32_t SortAndComputeUniqueIdx(int64_t rowLen, LocalTensor<IDX_T> indicesSrcLocal, LocalTensor<IDX_T> sortIndicesLocal, 
    LocalTensor<int32_t> uniqueIdxLocal, LocalTensor<uint32_t> updatesOriginIdexLocal)
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    int64_t shiftOffset = UB_AGLIN_VALUE / sizeof(IDX_T);
    LocalTensor<IDX_T> shiftSortLocal = sortIndicesLocal[shiftOffset];
    AscendC::Sort<IDX_T, true, sortConfig>(
        shiftSortLocal, updatesOriginIdexLocal, indicesSrcLocal, static_cast<uint32_t>(rowLen));
    Duplicate(sortIndicesLocal, (IDX_T)-1, shiftOffset);
    shiftSortLocal(rowLen) = -1;
    
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    return ComputeUniqueIdNum(sortIndicesLocal, uniqueIdxLocal, rowLen);
}

__aicore__ inline void ComputeUniqueIdTimes(LocalTensor<int32_t>& noDupRes, uint32_t& arNum)
{
    __local_mem__ int32_t* noDupResAddr = (__ubuf__ int32_t*)noDupRes.GetPhyAddr();
    uint16_t loopCntStatFre = (arNum + VF_B32 - 1) / VF_B32;
    uint32_t counterStatFre = arNum;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> beginReg;
        AscendC::MicroAPI::RegTensor<int32_t> endReg;
        AscendC::MicroAPI::RegTensor<int32_t> subReg;
        AscendC::MicroAPI::MaskReg maskRegUpdate;
        AscendC::MicroAPI::UnalignReg u0;

        for (uint16_t i = 0; i < loopCntStatFre; i++) {
            auto noDupResAddrUpdate = noDupResAddr + i * VF_B32 + 1;
            maskRegUpdate = AscendC::MicroAPI::UpdateMask<int32_t>(counterStatFre);
            AscendC::MicroAPI::DataCopy(beginReg, noDupResAddr + i * VF_B32);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, noDupResAddrUpdate);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t>(endReg, u0, noDupResAddrUpdate);
            AscendC::MicroAPI::Sub(subReg, endReg, beginReg, maskRegUpdate);
            AscendC::MicroAPI::DataCopy(noDupResAddr + i * VF_B32, subReg, maskRegUpdate);
        }
    }

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

template <typename T>
__aicore__ inline void CopyIn(
    LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;                // 连续传输块个数
    dataCoptExtParams.blockLen = copyLen * sizeof(T);     // 每块大小
    dataCoptExtParams.srcStride = srcStride * sizeof(T);  // 源地址相邻块间隔
    dataCoptExtParams.dstStride = dstStride * sizeof(T);  // 目的地址相邻块间隔
    DataCopyPad(dstLocal, srcGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T>
__aicore__ inline void CopyOut(
    GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(dstGm[offset], srcLocal, dataCoptExtParams);
}

}  // namespace ScatterAddCommon
#endif  // SCATTER_ADD_COMMON_IMPL_H