/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_BASE_H
#define UNSORTED_SEGMENT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "op_kernel/math_util.h"
#include "unsorted_segment_struct.h"

namespace UnsortedSegment {
using namespace AscendC;
using namespace platform;

#ifdef __DAV_FPGA__
constexpr int64_t SIMT_THREAD_DIM = 128;
constexpr int64_t SIMT_THREAD_DIM_LAUNCH_BOUND = 512;
constexpr int32_t SORT_THREAD_DIM = 128;
constexpr int32_t SORT_THREAD_DIM_LAUNCH_BOUND = 512;
#else
constexpr int64_t SIMT_THREAD_DIM = 2048;
constexpr int64_t SIMT_THREAD_DIM_LAUNCH_BOUND = 2048;
constexpr int32_t SORT_THREAD_DIM = 1024;
constexpr int32_t SORT_THREAD_DIM_LAUNCH_BOUND = 1024;
#endif
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BUFFER_ADD_NUM = 2;
constexpr uint64_t ONE_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t SEGMENT_ID_FACTOR = 8;
constexpr uint32_t ROW_NUM = 16;
constexpr uint32_t COUNT = 64;
constexpr uint32_t HALFTIME = 4;
constexpr uint32_t TWO = 2;
constexpr uint32_t VF_SIZE = platform::GetVRegSize();
constexpr uint32_t VF_B32 = VF_SIZE / sizeof(int32_t);
constexpr uint64_t MIN_FACTOR = 2 * 1024;
constexpr uint64_t GM_ALIGN = 512;

constexpr float FLOAT32_MAX = 3.4028235e+38f;
constexpr half FLOAT16_MAX = 65504.0f;
constexpr bfloat16_t BFLOAT16_MAX = 3.3895314e+38f;
// constexpr int32_t INT32_MAX = 2147483647;
// constexpr int64_t UINT32_MAX = 4294967295;
// constexpr int64_t INT64_MAX = 9223372036854775807LL;
// constexpr uint64_t UINT64_MAX = 18446744073709551615ULL;

typedef struct {
    uint16_t segCount;
    uint32_t outGmIndex;
    uint32_t xPerRowNum;
    __local_mem__ uint32_t* sortedIdxAddr;
} xAddParams;

template <typename T>
__aicore__ inline T Aligned(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T>
__aicore__ inline constexpr T GetDtypeMax()
{
    T dtypeMax = 0;
    if constexpr (IsSameType<T, int32_t>::value) {
        dtypeMax = INT32_MAX;
    } else if constexpr (IsSameType<T, int64_t>::value) {
        dtypeMax = INT64_MAX;
    } else if constexpr (IsSameType<T, uint32_t>::value) {
        dtypeMax = UINT32_MAX;
    } else if constexpr (IsSameType<T, uint64_t>::value) {
        dtypeMax = UINT64_MAX;
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        dtypeMax = BFLOAT16_MAX;
    } else if constexpr (IsSameType<T, half>::value) {
        dtypeMax = FLOAT16_MAX;
    } else if constexpr (IsSameType<T, float>::value) {
        dtypeMax = FLOAT32_MAX;
    }
    return dtypeMax;
}

__aicore__ inline uint32_t RoundUpOneBlock(uint32_t x)
{
    return (x + ONE_BLOCK_SIZE - 1) / ONE_BLOCK_SIZE * ONE_BLOCK_SIZE;
}

template <typename TX>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_DIM_LAUNCH_BOUND) inline void ComputeSetValue(
    __gm__ TX* outputGm, const uint32_t blockNums, const uint32_t outputLength)
{
    for (uint32_t outputIndex = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); outputIndex < outputLength;
         outputIndex = outputIndex + blockNums * Simt::GetThreadNum()) {
        outputGm[outputIndex] = static_cast<TX>(0);
    }
}

template <typename TX, typename Index, uint8_t Mode>
__simt_vf__ __aicore__ LAUNCH_BOUND(SORT_THREAD_DIM_LAUNCH_BOUND) inline void SimtGatherValue(
    __ubuf__ TX* midResPtr, __ubuf__ TX* xUbLocalPtr, __ubuf__ Index* indexUb, const uint32_t outputOuterDimSize,
    const uint32_t innerDimSize, const uint32_t needIndexOneUb, const uint32_t outputOffset)
{
    Index midBaseOffset = Simt::GetThreadIdx<1>() * outputOffset;
    Index offset = Simt::GetThreadIdx<1>();
    for (; offset < needIndexOneUb; offset += ROW_NUM) {
        Index indexVal = indexUb[offset];
        if (indexVal >= 0 && indexVal < outputOuterDimSize) {
            Index midResOffSet = indexVal * innerDimSize;
            Index xUbLocalOffSet = offset * innerDimSize;
            if constexpr (Mode == 0) {
                TX midResP = midResPtr[midBaseOffset + midResOffSet + Simt::GetThreadIdx<0>()];
                TX xUbLocalRes = xUbLocalPtr[xUbLocalOffSet + Simt::GetThreadIdx<0>()];
                midResPtr[midBaseOffset + midResOffSet + Simt::GetThreadIdx<0>()] = midResP < xUbLocalRes ? midResP : xUbLocalRes;
            }
        }
    }
}

template <typename TX, typename Index, typename COM_T, uint8_t Mode>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeSegment(
    __gm__ TX* xGm, __gm__ Index* segmentIdsGm, __gm__ TX* outputGm, const uint32_t blockNums, const COM_T inputLength,
    const COM_T innerDimSize, const uint64_t outputOuterDimSize, const COM_T magic, const COM_T shift)
{
    for (COM_T inputIndex = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); inputIndex < inputLength;
         inputIndex = inputIndex + blockNums * Simt::GetThreadNum()) {
        COM_T inputSegmentIndex = Simt::UintDiv(inputIndex, magic, shift);
        COM_T segmentOffset = inputIndex - inputSegmentIndex * innerDimSize;
        const Index outputSegmentIndex = segmentIdsGm[inputSegmentIndex];
        if (outputSegmentIndex < 0 || outputSegmentIndex >= outputOuterDimSize) {
            continue;
        }
        const uint64_t outputIndex = outputSegmentIndex * innerDimSize + segmentOffset;
        if constexpr (Mode == 0) {
            Simt::AtomicMin(outputGm + outputIndex, xGm[inputIndex]);
        }
    }
}

template <typename TX, typename Index, uint8_t Mode>
__simt_vf__ __aicore__ LAUNCH_BOUND(SORT_THREAD_DIM_LAUNCH_BOUND) inline void SegmentReduceSortSimt(
    __ubuf__ TX* inputAddr, __ubuf__ uint32_t* sortedOriginIndexAddr, __ubuf__ Index* sortedAddr,
    __ubuf__ uint32_t* cumSumAddr, __gm__ TX* outputAddr, int32_t uniqueIndexNum, uint32_t lastDim,
    uint32_t outputOuterDimSize)
{
    int32_t blockIdx = Simt::GetThreadIdx<1>();
    int32_t blockNum = Simt::GetThreadNum<1>();
    int32_t innerOffset = Simt::GetThreadIdx<0>();
    for (int32_t i = blockIdx; i < uniqueIndexNum; i += blockNum) {
        if (sortedAddr[cumSumAddr[i]] < 0 || sortedAddr[cumSumAddr[i]] >= outputOuterDimSize) {
            continue;
        }
        if constexpr (Mode == 0) {
            TX result = GetDtypeMax<TX>();
            for (int32_t tid = 0; tid < cumSumAddr[i + 1] - cumSumAddr[i]; tid++) {
                int32_t srcOffset = sortedOriginIndexAddr[cumSumAddr[i] + tid] * lastDim + innerOffset;
                TX inputRes = inputAddr[srcOffset];
                result = result < inputRes ? result : inputRes;
            }
            int64_t gmDstOffset = sortedAddr[cumSumAddr[i]] * lastDim + innerOffset;
            Simt::AtomicMin(outputAddr + gmDstOffset, result);
        }
    }
    return;
}

template <typename T, uint8_t Mode>
__aicore__ inline void InitGm(GM_ADDR output, uint64_t totalNum)
{
    uint64_t initPerCore = (totalNum + GetBlockNum() - 1) / GetBlockNum();
    initPerCore = Aligned(initPerCore, GM_ALIGN / sizeof(T));
    uint64_t minFactorNum = MIN_FACTOR / sizeof(T);
    initPerCore = minFactorNum > initPerCore ? minFactorNum : initPerCore;
    uint64_t coreNum = Ops::Base::CeilDiv(totalNum, initPerCore);
    uint64_t initLastCore = totalNum - (coreNum - 1) * initPerCore;
    uint64_t initCoreReal = GetBlockIdx() == (coreNum - 1) ? initLastCore : initPerCore;

    AscendC::GlobalTensor<T> yGmInit;
    yGmInit.SetGlobalBuffer((__gm__ T*)output + GetBlockIdx() * initPerCore);
    if (GetBlockIdx() < coreNum) {
        if constexpr (Mode == 0) {
            InitGlobalMemory(yGmInit, initCoreReal, GetDtypeMax<T>());
        }
    }
    SyncAll();
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
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = dstStride * sizeof(T);
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

template <typename Index>
__aicore__ inline void UniqueGetElm(
    const LocalTensor<Index>& sortedIndice, LocalTensor<int32_t>& noDupRes, uint32_t processIdx, uint32_t shiftOffset,
    uint32_t vfIndicesNum, int64_t& arNum)
{
    __local_mem__ Index* sortedIndicesAddr = (__ubuf__ Index*)sortedIndice.GetPhyAddr();
    __local_mem__ int32_t* noDupResAddr = (__ubuf__ int32_t*)noDupRes.GetPhyAddr();

    uint16_t loopCnt = (uint16_t)((processIdx + vfIndicesNum) / vfIndicesNum);

    int32_t scalar = 0;
    uint32_t counter = processIdx + 1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> orderReg;
        AscendC::MicroAPI::RegTensor<int32_t> selReg;
        AscendC::MicroAPI::RegTensor<Index> indicesReg;
        AscendC::MicroAPI::RegTensor<Index> indicesShiftOneReg;

        AscendC::MicroAPI::MaskReg cmpMask;
        AscendC::MicroAPI::MaskReg maskRegUpdate;
        AscendC::MicroAPI::UnalignReg u0;
        MicroAPI::UnalignReg ureg;
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

        for (uint16_t i = 0; i < loopCnt; ++i) {
            scalar = i * vfIndicesNum;
            auto sortedIndicesAddrUpdate = sortedIndicesAddr + shiftOffset + i * vfIndicesNum;
            AscendC::MicroAPI::Arange(orderReg, scalar);
            maskRegUpdate = AscendC::MicroAPI::UpdateMask<Index>(counter);
            AscendC::MicroAPI::DataCopy(indicesReg, sortedIndicesAddrUpdate);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, sortedIndicesAddrUpdate - 1);
            AscendC::MicroAPI::DataCopyUnAlign<Index>(indicesShiftOneReg, u0, sortedIndicesAddrUpdate - 1);
            AscendC::MicroAPI::Compare<Index, CMPMODE::NE>(cmpMask, indicesReg, indicesShiftOneReg, maskRegUpdate);
            if constexpr (IsSameType<Index, int64_t>::value) {
                AscendC::MicroAPI::MaskReg maskHalf;
                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
                // vSQZ
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                    selReg, orderReg, maskHalf);
            } else {
                // vSQZ
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                    selReg, orderReg, cmpMask);
            }
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                noDupResAddr, selReg, ureg);
            AscendC::MicroAPI::DataCopyUnAlignPost(noDupResAddr, ureg);
        }
    }
}

__aicore__ inline void UniqueStat(LocalTensor<int32_t>& noDupRes, int64_t& arNum)
{
    __local_mem__ int32_t* noDupResAddr = (__ubuf__ int32_t*)noDupRes.GetPhyAddr();

    uint16_t loopCntStatFre = (arNum + VF_B32 - 1) / VF_B32;
    uint32_t counterStatFre = static_cast<uint32_t>(arNum);
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
} // namespace UnsortedSegment
#endif
