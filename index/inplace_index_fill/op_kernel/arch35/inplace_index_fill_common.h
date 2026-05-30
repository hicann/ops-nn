/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_common.h
 * \brief common kernel functions: SortAndUnique for DenseIndices template
 */

#ifndef INPLACE_INDEX_FILL_COMMON_H
#define INPLACE_INDEX_FILL_COMMON_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "inplace_index_fill_struct.h"

namespace InplaceIndexFillDenseIndices {
using namespace AscendC;

constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr int64_t VFLEN_INT64 = platform::GetVRegSize() / sizeof(int64_t);
constexpr int64_t VFLEN_INT32 = platform::GetVRegSize() / sizeof(int32_t);

constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

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
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

__aicore__ inline void PIPE_S_V() {
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

template<typename INDEX_TYPE>
__aicore__ inline void ComputeUniqueIdNumInt64(__local_mem__ INDEX_TYPE* sortedInputAddr, __local_mem__ int32_t* uniqueIndicesAddr, uint16_t loopCnt, int64_t dataLen)
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
__aicore__ inline void ComputeUniqueIdNumInt32(__local_mem__ INDEX_TYPE* sortedInputAddr, __local_mem__ int32_t* uniqueIndicesAddr, uint16_t loopCnt, int64_t dataLen)
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
__aicore__ inline uint32_t ComputeUniqueIdNum(LocalTensor<INDEX_TYPE> sortedInput, LocalTensor<int32_t> uniqueIndicesOut, int64_t dataLen)
{
    __local_mem__ INDEX_TYPE* sortedInputAddr = (__local_mem__ INDEX_TYPE*)sortedInput[(UB_AGLIN_VALUE / sizeof(INDEX_TYPE))].GetPhyAddr();
    __local_mem__ int32_t* uniqueIndicesAddr = (__local_mem__ int32_t*)uniqueIndicesOut.GetPhyAddr();

    constexpr int64_t vfLen = platform::GetVRegSize() / sizeof(INDEX_TYPE);
    uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
        if constexpr (std::is_same<int64_t, INDEX_TYPE>::value) {
            ComputeUniqueIdNumInt64<INDEX_TYPE>(sortedInputAddr, uniqueIndicesAddr, loopCnt, dataLen);
        } else if constexpr (std::is_same<int32_t, INDEX_TYPE>::value) {
            ComputeUniqueIdNumInt32<INDEX_TYPE>(sortedInputAddr, uniqueIndicesAddr, loopCnt, dataLen);
        }
    }
    uint32_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
    return uniqueIdNum;
}

/**
 * Sort and deduplicate indices in UB
 *
 * @param sortedOut    output: sorted result with sentinel areas
 *     1) first 32 bytes are front sentinel area (value = -1)
 *     2) sortedOut[frontSentinelAreaSize, size-1] is the actual sorted result
 *     3) one element after: sortedOut[size] = -1 (back sentinel)
 * @param uniqueIndicesOut  output: unique position indices
 * @param src               input: tensor to sort and deduplicate
 * @param size              input: number of elements
 * @return number of unique elements
 */
template<typename INDEX_TYPE>
__aicore__ inline uint32_t SortAndUnique(LocalTensor<INDEX_TYPE> sortedOut, LocalTensor<int32_t> uniqueIndicesOut, LocalTensor<INDEX_TYPE> src, int64_t size)
{
    PIPE_V_S();

    int64_t frontSentinelAreaSize = UB_AGLIN_VALUE / sizeof(INDEX_TYPE);
    LocalTensor<INDEX_TYPE> actualSortOut = sortedOut[frontSentinelAreaSize];
    AscendC::Sort<INDEX_TYPE, true, sortConfig>(actualSortOut, src, static_cast<uint32_t>(size));
    Duplicate(sortedOut, (INDEX_TYPE)-1, frontSentinelAreaSize);
    PIPE_V_S();
    actualSortOut(size) = -1;

    PIPE_S_V();
    return ComputeUniqueIdNum(sortedOut, uniqueIndicesOut, size);
}

} // namespace InplaceIndexFillDenseIndices

// ============================================================================
// 公共函数：InitMask 和 MaskBasedFillSimt
// 被 SIMT 模板和 DenseIndices 模板共用
// ============================================================================
namespace InplaceIndexFillCommon {
using namespace AscendC;
using InplaceIndexFill::SIMT_THREAD_NUM;

// 多核并行将 mask 位图初始化为全 0，并执行核间同步
__aicore__ inline void InitMaskZero(
    GlobalTensor<int8_t>& maskGm, GM_ADDR workspace,
    int64_t n, uint32_t blockIdx, uint32_t blockNum)
{
    maskGm.SetGlobalBuffer((__gm__ int8_t*)(workspace), n * sizeof(int8_t));

    constexpr int64_t ALIGN_SIZE = 32;
    int64_t perCore = ((n + blockNum - 1) / blockNum + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
    int64_t start = blockIdx * perCore;
    if (start < n) {
        int64_t len = (start + perCore <= n) ? perCore : ((n - start + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE);
        GlobalTensor<int8_t> maskSlice;
        maskSlice.SetGlobalBuffer((__gm__ int8_t*)(workspace) + start, len);
        InitOutput<int8_t>(maskSlice, len, int8_t(0));
    }

    SyncAll();
}

// 基于 mask 位图的顺序填充 SIMT 核函数
// 遍历 P*N*Q，对 mask[nIdx]==1 的位置写入 value
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void MaskBasedFillSimt(
    __gm__ T_X* x, __gm__ int8_t* mask, T_X value,
    COM_T total_num, COM_T n,
    COM_T shift_n, COM_T magic_n,
    COM_T shift_q, COM_T magic_q)
{
    COM_T threadIdx = static_cast<COM_T>(Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
    COM_T threadNum = static_cast<COM_T>(Simt::GetBlockNum() * Simt::GetThreadNum());

    for (COM_T i = threadIdx; i < total_num; i += threadNum) {
        COM_T pnIdx = Simt::UintDiv(i, magic_q, shift_q);
        COM_T pIdx = Simt::UintDiv(pnIdx, magic_n, shift_n);
        COM_T nIdx = pnIdx - (pIdx * n);
        if (mask[nIdx] == 0) {
            continue;
        }
        x[i] = value;
    }
}

// 封装 magic number 计算 + MaskBasedFillSimt 调用
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void CallMaskBasedFill(
    __gm__ T_X* x, __gm__ int8_t* maskAddr, T_X fillValue,
    COM_T n, COM_T p, COM_T q)
{
    COM_T process_num = n * p * q;

    COM_T shift_n = 0, magic_n = 0;
    GetUintDivMagicAndShift(magic_n, shift_n, n);

    COM_T shift_q = 0, magic_q = 0;
    GetUintDivMagicAndShift(magic_q, shift_q, q);

    Simt::VF_CALL<MaskBasedFillSimt<T_X, INDEX_TYPE, COM_T>>(
        Simt::Dim3{SIMT_THREAD_NUM},
        x, maskAddr, fillValue, process_num, n,
        shift_n, magic_n, shift_q, magic_q);
}

} // namespace InplaceIndexFillCommon

#endif  // INPLACE_INDEX_FILL_COMMON_H