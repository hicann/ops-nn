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
constexpr uint32_t CAST_0 = 0;
constexpr uint32_t CAST_1 = 1;
constexpr uint32_t CAST_2 = 2;
constexpr uint32_t CAST_3 = 3;

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

template <typename T, uint64_t bufferNum, bool updatesIsScalar>
__aicore__ inline void BroadcastUpdatesScalar(
    TQue<QuePosition::VECIN, bufferNum> updatesQueue, GlobalTensor<T> updatesGm, int32_t count)
{
    if constexpr (updatesIsScalar) {
        T updatesValue = updatesGm.GetValue(0);
        auto vWaitSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(vWaitSEventID);
        WaitFlag<HardEvent::S_V>(vWaitSEventID);
        LocalTensor<T> updatesLocal = updatesQueue.template AllocTensor<T>();
        Duplicate(updatesLocal, updatesValue, count);
        updatesQueue.EnQue(updatesLocal);
    }
}

template<typename IDX_T>
__aicore__ uint32_t ComputeUniqueIdNum(LocalTensor<IDX_T> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
{
    __local_mem__ IDX_T* indicesAddr = (__local_mem__ IDX_T*)indicesLocal[(UB_AGLIN_VALUE / sizeof(IDX_T))].GetPhyAddr();
    __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

    constexpr int64_t vfLen = platform::GetVRegSize() / sizeof(IDX_T);
    uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
    uint32_t counter = dataLen + 1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> orderReg;
        AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg;
        AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxShiftOneReg;
        AscendC::MicroAPI::RegTensor<int32_t> selReg;
        AscendC::MicroAPI::RegTensor<int32_t> orderReg2;
        AscendC::MicroAPI::RegTensor<int32_t> selReg2;
        AscendC::MicroAPI::MaskReg cmpMask;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * vfLen);
            maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
            auto startAddr = indicesAddr + i * vfLen;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            if constexpr (std::is_same<int64_t, IDX_T>::value) {
                AscendC::MicroAPI::MaskReg maskHalf;
                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskHalf);
            } else if constexpr (std::is_same<int32_t, IDX_T>::value) {
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, cmpMask);
            } else {  // int16
                AscendC::MicroAPI::Arange(orderReg2, i * vfLen + vfLen / 2);
                AscendC::MicroAPI::MaskReg maskDouble1;
                AscendC::MicroAPI::MaskReg maskDouble2;
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskDouble1, cmpMask);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskDouble2, cmpMask);
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskDouble1);
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskDouble2);
            }
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg, uOut);
            if constexpr (std::is_same<int16_t, IDX_T>::value) {
                AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    uniqueIdCountsAddr, selReg2, uOut);
            }
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
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