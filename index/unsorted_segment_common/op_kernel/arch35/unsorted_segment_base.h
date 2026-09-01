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
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

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

constexpr uint32_t CAST_NONE = 0;
constexpr uint32_t CAST_INT32_TO_INT16 = 1;
constexpr uint32_t CAST_INT64_TO_INT32 = 2;
constexpr uint32_t CAST_INT64_TO_INT16 = 3;
constexpr uint32_t CAST_INT32_TO_UINT8 = 4;
constexpr uint32_t CAST_INT64_TO_UINT8 = 5;
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
        CAST_MODE == CAST_INT32_TO_INT16, int16_t,
        typename std::conditional<
            CAST_MODE == CAST_INT64_TO_INT32, int32_t,
            typename std::conditional<
                CAST_MODE == CAST_INT64_TO_INT16, int16_t,
                typename std::conditional<CAST_MODE == CAST_INT32_TO_UINT8, uint8_t,
                                          typename std::conditional<CAST_MODE == CAST_INT64_TO_UINT8, uint8_t,
                                                                    T>::type>::type>::type>::type>::type;
};

typedef struct {
    uint16_t segCount;
    uint32_t outGmIndex;
    uint32_t xPerRowNum;
    __ubuf__ uint32_t* sortedIdxAddr;
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
    for (uint32_t outputIndex = blockIdx.x * blockDim.x + threadIdx.x; outputIndex < outputLength;
         outputIndex = outputIndex + blockNums * blockDim.x) {
        outputGm[outputIndex] = static_cast<TX>(0);
    }
}

template <typename TX, typename Index, typename SimtGatherFunc>
__simt_vf__ __aicore__ LAUNCH_BOUND(SORT_THREAD_DIM_LAUNCH_BOUND) inline void SimtGatherValue(
    __ubuf__ TX* midResPtr, __ubuf__ TX* xUbLocalPtr, __ubuf__ Index* indexUb, const uint32_t outputOuterDimSize,
    const uint32_t innerDimSize, const uint32_t needIndexOneUb, const uint32_t outputOffset, const uint32_t parallelNum)
{
    if (innerDimSize == 1U) {
        uint32_t offset32 = static_cast<uint32_t>(threadIdx.y);
        uint32_t midBase32 = static_cast<uint32_t>(threadIdx.y) * outputOffset;
        for (; offset32 < needIndexOneUb; offset32 += parallelNum) {
            Index indexVal = indexUb[offset32];
            if (indexVal >= 0 && indexVal < outputOuterDimSize) {
                uint32_t dstOffset = midBase32 + static_cast<uint32_t>(indexVal);
                midResPtr[dstOffset] = SimtGatherFunc()(midResPtr[dstOffset], xUbLocalPtr[offset32]);
            }
        }
        return;
    }
    Index midBaseOffset = threadIdx.y * outputOffset;
    Index offset = threadIdx.y;
    for (; offset < needIndexOneUb; offset += parallelNum) {
        Index indexVal = indexUb[offset];
        if (indexVal >= 0 && indexVal < outputOuterDimSize) {
            Index midResOffSet = indexVal * innerDimSize;
            Index xUbLocalOffSet = offset * innerDimSize;
            TX midResP = midResPtr[midBaseOffset + midResOffSet + threadIdx.x];
            TX xUbLocalRes = xUbLocalPtr[xUbLocalOffSet + threadIdx.x];
            midResPtr[midBaseOffset + midResOffSet + threadIdx.x] = SimtGatherFunc()(midResP, xUbLocalRes);
        }
    }
}

template <typename TX, typename Index, typename COM_T, typename SimtAtomicFunc>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeSegment(
    __gm__ TX* xGm, __gm__ Index* segmentIdsGm, __gm__ TX* outputGm, const uint32_t blockNums, const COM_T inputLength,
    const COM_T innerDimSize, const uint64_t outputOuterDimSize, const COM_T magic, const COM_T shift)
{
    for (COM_T inputIndex = blockIdx.x * blockDim.x + threadIdx.x; inputIndex < inputLength;
         inputIndex = inputIndex + blockNums * blockDim.x) {
        COM_T inputSegmentIndex = Simt::UintDiv(inputIndex, magic, shift);
        COM_T segmentOffset = inputIndex - inputSegmentIndex * innerDimSize;
        const Index outputSegmentIndex = segmentIdsGm[inputSegmentIndex];
        if (outputSegmentIndex < 0 || outputSegmentIndex >= outputOuterDimSize) {
            continue;
        }
        const uint64_t outputIndex = outputSegmentIndex * innerDimSize + segmentOffset;
        SimtAtomicFunc()(outputGm, outputIndex, xGm[inputIndex]);
    }
}

template <typename TX, typename Index, typename SimtGatherFunc, typename SimtAtomicFunc, typename InitValueType>
__simt_vf__ __aicore__ LAUNCH_BOUND(SORT_THREAD_DIM_LAUNCH_BOUND) inline void SegmentReduceSortSimt(
    __ubuf__ TX* inputAddr, __ubuf__ uint32_t* sortedOriginIndexAddr, __ubuf__ Index* sortedAddr,
    __ubuf__ uint32_t* cumSumAddr, __gm__ TX* outputAddr, int32_t uniqueIndexNum, uint32_t lastDim,
    uint32_t outputOuterDimSize)
{
    int32_t blockIdxLocal = threadIdx.y;
    int32_t blockNum = blockDim.y;
    int32_t innerOffset = threadIdx.x;
    for (int32_t i = blockIdxLocal; i < uniqueIndexNum; i += blockNum) {
        if (sortedAddr[cumSumAddr[i]] < 0 || sortedAddr[cumSumAddr[i]] >= outputOuterDimSize) {
            continue;
        }
        TX result = InitValueType::Get();
        for (int32_t tid = 0; tid < cumSumAddr[i + 1] - cumSumAddr[i]; tid++) {
            int32_t srcOffset = sortedOriginIndexAddr[cumSumAddr[i] + tid] * lastDim + innerOffset;
            TX inputRes = inputAddr[srcOffset];
            result = SimtGatherFunc()(result, inputRes);
        }
        int64_t gmDstOffset = sortedAddr[cumSumAddr[i]] * lastDim + innerOffset;
        SimtAtomicFunc()(outputAddr, gmDstOffset, result);
    }
    return;
}

template <typename T, typename GmInitFunc>
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
        GmInitFunc()(yGmInit, initCoreReal);
    }
    SyncAll();
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
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = dstStride * sizeof(T);
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

template <typename IDX_T, typename CAST_T, uint32_t castType>
__aicore__ inline void IndicesSortCast(LocalTensor<IDX_T> indicesLocal, LocalTensor<CAST_T> indicesCastLocal,
                                       LocalTensor<int32_t> indicesCastTmpLocal, uint32_t indicesCount)
{
    if constexpr (castType == CAST_INT32_TO_UINT8) { // int32 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(MASK_UINT8),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_INT64_TO_INT16) { // int64 Cast int16
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_INT64_TO_UINT8) { // int64 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(MASK_UINT8),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else { // CAST_INT32_TO_INT16 + CAST_INT64_TO_INT32, int32 Cast int16 + int64 Cast int32
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
    }
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
__simd_vf__ inline void UniqueGetElmVf(__ubuf__ IDX_T* indicesAddr, __ubuf__ int32_t* uniqueIdCountsAddr,
                                       uint16_t loopCnt, int64_t dataLen)
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
__aicore__ inline int64_t UniqueGetElm(const LocalTensor<IDX_T>& sortedIndice, LocalTensor<int32_t>& noDupRes,
                                       int64_t dataLen)
{
    __ubuf__ IDX_T* indicesAddr = (__ubuf__ IDX_T*)sortedIndice[(ONE_BLOCK_SIZE / sizeof(IDX_T))].GetPhyAddr();
    __ubuf__ int32_t* uniqueIdCountsAddr = (__ubuf__ int32_t*)noDupRes.GetPhyAddr();

    constexpr int64_t vfLen = platform::GetVRegSize() / sizeof(IDX_T);
    uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
    UniqueGetElmVf<IDX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
    int64_t uniqueIdNum = ((AscendC::Reg::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t));
    uniqueIdNum = uniqueIdNum > 0 ? uniqueIdNum - 1 : 0;
    return uniqueIdNum;
}

__simd_vf__ inline void UniqueStatVf(__ubuf__ int32_t* noDupResAddr, uint16_t loopCntStatFre, uint32_t counterStatFre)
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

__aicore__ inline void UniqueStat(LocalTensor<int32_t>& noDupRes, int64_t& arNum)
{
    __ubuf__ int32_t* noDupResAddr = (__ubuf__ int32_t*)noDupRes.GetPhyAddr();

    uint16_t loopCntStatFre = (arNum + VF_B32 - 1) / VF_B32;
    uint32_t counterStatFre = static_cast<uint32_t>(arNum);
    UniqueStatVf(noDupResAddr, loopCntStatFre, counterStatFre);

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}
} // namespace UnsortedSegment
#endif
