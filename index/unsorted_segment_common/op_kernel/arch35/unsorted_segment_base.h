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
constexpr uint32_t THREE = 3;
constexpr uint32_t VF_SIZE = platform::GetVRegSize();
constexpr uint32_t VF_B32 = VF_SIZE / sizeof(int32_t);
constexpr uint64_t MIN_FACTOR = 2 * 1024;
constexpr uint64_t GM_ALIGN = 512;

constexpr float FLOAT32_MAX = 3.4028235e+38f;
constexpr half FLOAT16_MAX = 65504.0f;
constexpr bfloat16_t BFLOAT16_MAX = 3.3895314e+38f;

constexpr uint32_t CAST_0 = 0;
constexpr uint32_t CAST_1 = 1;
constexpr uint32_t CAST_2 = 2;
constexpr uint32_t CAST_3 = 3;
constexpr uint32_t CAST_4 = 4;
constexpr uint32_t CAST_5 = 5;
constexpr uint32_t MASK_UINT8 = 255;
constexpr int64_t VFLEN_INT64 = platform::GetVRegSize() / sizeof(int64_t);
constexpr int64_t VFLEN_INT32 = platform::GetVRegSize() / sizeof(int32_t);
constexpr int64_t VFLEN_INT16 = platform::GetVRegSize() / sizeof(int16_t);
constexpr int64_t VFLEN_INT16HALF = platform::GetVRegSize() / sizeof(int16_t) / TWO;
constexpr int64_t VFLEN_UINT8 = platform::GetVRegSize() / sizeof(uint8_t);
constexpr int64_t VFLEN_UINT8HALFHALF = platform::GetVRegSize() / sizeof(uint8_t) / HALFTIME;

template <typename T, uint32_t CAST_MODE>  
struct CastType {                                                         
    using type = typename std::conditional<
        CAST_MODE == CAST_1, int16_t, typename std::conditional<CAST_MODE == CAST_2, int32_t, 
            typename std::conditional<CAST_MODE == CAST_3, int16_t, typename std::conditional<CAST_MODE == CAST_4, uint8_t,
                typename std::conditional<CAST_MODE == CAST_5, uint8_t, T>::type>::type>::type>::type>::type;
};

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

template <typename IDX_T, typename CAST_T, uint32_t castType>
__aicore__ inline void IndicesSortCast(LocalTensor<IDX_T> indicesLocal, LocalTensor<CAST_T> indicesCastLocal,
                                                LocalTensor<int32_t> indicesCastTmpLocal, uint32_t indicesCount)
{
    if constexpr (castType == CAST_4) {  // int32 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(MASK_UINT8), SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_3) {  // int64 Cast int16
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_5) {  // int64 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(MASK_UINT8), SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else {    // CAST_1 + CAST_2, int32 Cast int16 + int64 Cast int32
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
    }
}

template<typename IDX_T>
__aicore__ inline void ComputeUniqueIdNumInt64(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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
__aicore__ inline void ComputeUniqueIdNumInt32(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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
__aicore__ inline void ComputeUniqueIdNumInt16(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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
__aicore__ inline void ComputeUniqueIdNumUint8(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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

template <typename IDX_T>
__aicore__ inline int64_t UniqueGetElm(
    const LocalTensor<IDX_T>& sortedIndice, LocalTensor<int32_t>& noDupRes, int64_t dataLen)
{
    __local_mem__ IDX_T* indicesAddr = (__local_mem__ IDX_T*)sortedIndice[(ONE_BLOCK_SIZE / sizeof(IDX_T))].GetPhyAddr();
    __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)noDupRes.GetPhyAddr();

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
    int64_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
    return uniqueIdNum;
}

__aicore__ inline void UniqueStat(LocalTensor<int32_t>& noDupRes, int64_t& arNum)
{
    __local_mem__ int32_t* noDupResAddr = (__local_mem__ int32_t*)noDupRes.GetPhyAddr();

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
