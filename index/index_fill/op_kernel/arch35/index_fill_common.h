/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_fill_common.h
 * \brief common kernel function of index_fill
 */

#ifndef INDEX_FILL_COMMON_H
#define INDEX_FILL_COMMON_H

#include <limits>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "index_fill_struct.h"

namespace IndexFill {
using namespace AscendC;

constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr int64_t VFLEN_INT64 = platform::GetVRegSize() / sizeof(int64_t);
constexpr int64_t VFLEN_INT32 = platform::GetVRegSize() / sizeof(int32_t);

constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

template <typename INDEX_TYPE>
constexpr INDEX_TYPE INDEX_TYPE_MAX = std::numeric_limits<INDEX_TYPE>::max();

template <typename INDEX_TYPE>
constexpr INDEX_TYPE INDEX_TYPE_MIN = std::numeric_limits<INDEX_TYPE>::min();

__aicore__ inline void PIPE_V_S() {
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

__aicore__ inline void PIPE_MTE2_S() {
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

__aicore__ inline void PIPE_V_MTE3() {
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

__aicore__ inline void PIPE_S_V() {
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

__aicore__ inline void ZeroMemory(GlobalTensor<int8_t>& tensor, uint64_t length)
{
    __gm__ int8_t* baseAddr = (__gm__ int8_t*)tensor.GetPhyAddr();
    int8_t zero = 0;
    if (length <= MASK_THRESHPLD) {
        AscendC::Fill(tensor, length, zero);
        return;
    }

    uint64_t usedCoreNum = GetBlockNum();
    if (usedCoreNum == 0) {
        return;
    }
    uint64_t numelPerCore = length / usedCoreNum;
    uint64_t tailBlockNum = length - numelPerCore * usedCoreNum;
    if (numelPerCore <= 0) {
        usedCoreNum = tailBlockNum;
    }

    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= usedCoreNum) {
        return;
    }

    uint64_t offset = 0;
    uint64_t len = 0;
    GlobalTensor<int8_t> subMask;
    if (blockIdx < tailBlockNum) {
        len = numelPerCore + 1;
        offset = blockIdx * (numelPerCore + 1);
    } else {
        len = numelPerCore;
        offset = tailBlockNum * (numelPerCore + 1) + (blockIdx - tailBlockNum) * numelPerCore;
    }

    subMask.SetGlobalBuffer(baseAddr + offset, len);
    AscendC::Fill(subMask, len, zero);
}

template <typename INDEX_TYPE>
__aicore__ inline INDEX_TYPE GetFrontSentinelValue(INDEX_TYPE first)
{
    // 获取前向区域的哨兵值: 只要哨兵值与首元素值不同，则是合适的哨兵值
    INDEX_TYPE minValue = INDEX_TYPE_MIN<INDEX_TYPE>;
    return (minValue == first) ? static_cast<INDEX_TYPE>(-1) : minValue;
}

template <typename INDEX_TYPE>
__aicore__ inline INDEX_TYPE GetTailSentinelValue(INDEX_TYPE last)
{
    // 获取后向区域的哨兵值: 只要哨兵值与尾元素不同，则是合适的哨兵值
    INDEX_TYPE maxValue = INDEX_TYPE_MAX<INDEX_TYPE>;
    return (maxValue == last) ? static_cast<INDEX_TYPE>(1) : maxValue;
}

template<typename INDEX_TYPE>
__simd_callee__ inline void ComputeUniqueIdNumInt64(__ubuf__ INDEX_TYPE* sortedInputAddr, __ubuf__ int32_t* uniqueIndicesAddr, uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
    AscendC::MicroAPI::RegTensor<INDEX_TYPE> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskHalf;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT64);
        maskReg = AscendC::MicroAPI::UpdateMask<INDEX_TYPE>(counter);
        auto startAddr = sortedInputAddr + i * VFLEN_INT64;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<INDEX_TYPE>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<INDEX_TYPE, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskHalf);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            uniqueIndicesAddr, selReg, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIndicesAddr, uOut);
}

template<typename INDEX_TYPE>
__simd_callee__ inline void ComputeUniqueIdNumInt32(__ubuf__ INDEX_TYPE* sortedInputAddr, __ubuf__ int32_t* uniqueIndicesAddr, uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
    AscendC::MicroAPI::RegTensor<INDEX_TYPE> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT32);
        maskReg = AscendC::MicroAPI::UpdateMask<INDEX_TYPE>(counter);
        auto startAddr = sortedInputAddr + i * VFLEN_INT32;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<INDEX_TYPE>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<INDEX_TYPE, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, cmpMask);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            uniqueIndicesAddr, selReg, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIndicesAddr, uOut);
}

template<typename INDEX_TYPE>
__simd_vf__ inline void ComputeUniqueIdNumVf(__ubuf__ INDEX_TYPE* sortedInputAddr, __ubuf__ int32_t* uniqueIndicesAddr, uint16_t loopCnt, int64_t dataLen)
{
    AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    if constexpr (std::is_same<int64_t, INDEX_TYPE>::value) {
        ComputeUniqueIdNumInt64<INDEX_TYPE>(sortedInputAddr, uniqueIndicesAddr, loopCnt, dataLen);
    } else if constexpr (std::is_same<int32_t, INDEX_TYPE>::value) {
        ComputeUniqueIdNumInt32<INDEX_TYPE>(sortedInputAddr, uniqueIndicesAddr, loopCnt, dataLen);
    }
}

template<typename INDEX_TYPE>
__aicore__ inline uint32_t ComputeUniqueIdNum(LocalTensor<INDEX_TYPE> sortedInput, LocalTensor<int32_t> uniqueIndicesOut, int64_t dataLen)
{
    constexpr int64_t vfLen = platform::GetVRegSize() / sizeof(INDEX_TYPE);
    uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
    ComputeUniqueIdNumVf<INDEX_TYPE>(
        (__ubuf__ INDEX_TYPE*)sortedInput[(UB_AGLIN_VALUE / sizeof(INDEX_TYPE))].GetPhyAddr(),
        (__ubuf__ int32_t*)uniqueIndicesOut.GetPhyAddr(),
        loopCnt, dataLen);
    uint32_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
    return uniqueIdNum;
}

 /**
  * 将大小为size的输入tensor: src， 排序并去重
  *
  * @param sortedOut 输出: 排序后的结果。
  *     1).sortedOut的前32字节为前哨兵区域，值都为actualSortOut[0] - 1,区域为: [0, frontSentinelAreaSize - 1], 哨兵区域大小为: frontSentinelAreaSize
  *     2).sortedOut[frontSentinelAreaSize, size - 1]为实际的排序结果
  *     3).后哨兵区域只有一个元素: sortedOut[size]，值为actualSortOut[size - 1] + 1.
  * @param uniqueIndicesOut  输出: 去重的位置索引
  * @param src               输入: 待排序去重的输入张量
  * @param size              输入: 输入张量的大小
  * @return 返回有不重复的元素的总数
  */
template<typename INDEX_TYPE>
__aicore__ inline uint32_t SortAndUnique(LocalTensor<INDEX_TYPE> sortedOut, LocalTensor<int32_t> uniqueIndicesOut, LocalTensor<INDEX_TYPE> src, int64_t size)
{
    PIPE_V_S();

    int64_t frontSentinelAreaSize = UB_AGLIN_VALUE / sizeof(INDEX_TYPE);
    LocalTensor<INDEX_TYPE> actualSortOut = sortedOut[frontSentinelAreaSize];
    AscendC::Sort<INDEX_TYPE, true, sortConfig>(actualSortOut, src, static_cast<uint32_t>(size));
    PIPE_V_S();

    INDEX_TYPE frontSentinel = GetFrontSentinelValue(actualSortOut(0));
    INDEX_TYPE tailSentinel = GetTailSentinelValue(actualSortOut(size - 1));
    Duplicate(sortedOut, frontSentinel, frontSentinelAreaSize);
    PIPE_V_S();
    actualSortOut(size) = tailSentinel;

    PIPE_S_V();
    return ComputeUniqueIdNum(sortedOut, uniqueIndicesOut, size);
}

}
#endif  // INDEX_FILL_COMMON_H
