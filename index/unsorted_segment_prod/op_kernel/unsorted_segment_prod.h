/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_PROD_H
#define UNSORTED_SEGMENT_PROD_H

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace UnsortedSegmentProd {
using namespace AscendC;
using namespace platform;

constexpr uint64_t MAX_INT32_NUM = 2147483647;
constexpr uint64_t GM_ALIGN = 512;
constexpr uint64_t MIN_FACTOR = 2 * 1024;

#ifdef __DAV_FPGA__
constexpr int64_t SIMT_THREAD_DIM_LAUNCH_BOUND = 512;
#else
constexpr int64_t SIMT_THREAD_DIM_LAUNCH_BOUND = 2048;
#endif

// ======================== Identity value: 1 for prod ========================
template <typename T>
__aicore__ inline constexpr T GetIdentityOne()
{
    if constexpr (IsSameType<T, int32_t>::value) {
        return static_cast<T>(1);
    } else if constexpr (IsSameType<T, int64_t>::value) {
        return static_cast<T>(1);
    } else if constexpr (IsSameType<T, uint32_t>::value) {
        return static_cast<T>(1);
    } else if constexpr (IsSameType<T, uint64_t>::value) {
        return static_cast<T>(1);
    } else if constexpr (IsSameType<T, half>::value) {
        return static_cast<half>(1.0f);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        return static_cast<bfloat16_t>(1.0f);
    } else if constexpr (IsSameType<T, float>::value) {
        return 1.0f;
    }
    return static_cast<T>(1);
}

// ======================== GM Init: fill output with 1 ========================
template <typename T>
struct InitGmOneValue {
    __aicore__ inline InitGmOneValue()
    {}
    __aicore__ inline void operator()(GlobalTensor<T> yGmInit, uint64_t initCoreReal)
    {
        InitGlobalMemory(yGmInit, initCoreReal, GetIdentityOne<T>());
    }
};

template <typename T>
__aicore__ inline T AlignedVal(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T, typename GmInitFunc>
__aicore__ inline void InitGmProd(GM_ADDR output, uint64_t totalNum)
{
    uint64_t initPerCore = (totalNum + GetBlockNum() - 1) / GetBlockNum();
    initPerCore = AlignedVal(initPerCore, GM_ALIGN / sizeof(T));
    uint64_t minFactorNum = MIN_FACTOR / sizeof(T);
    initPerCore = minFactorNum > initPerCore ? minFactorNum : initPerCore;
    uint64_t coreNum = (totalNum + initPerCore - 1) / initPerCore;
    uint64_t initLastCore = totalNum - (coreNum - 1) * initPerCore;
    uint64_t initCoreReal = GetBlockIdx() == (coreNum - 1) ? initLastCore : initPerCore;

    AscendC::GlobalTensor<T> yGmInit;
    yGmInit.SetGlobalBuffer((__gm__ T*)output + GetBlockIdx() * initPerCore);
    if (GetBlockIdx() < coreNum) {
        GmInitFunc()(yGmInit, initCoreReal);
    }
    SyncAll();
}

// ======================== AtomCas Prod for int32/int64/uint32/uint64/float ========================
template <typename TX, typename Index, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeSegmentProdNormal(
    __gm__ TX* xGm, __gm__ Index* segmentIdsGm, __gm__ TX* outputGm, const uint32_t blockNums,
    const COM_T inputLength, const COM_T innerDimSize, const uint64_t outputOuterDimSize,
    const COM_T magic, const COM_T shift)
{
    for (COM_T inputIndex = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
         inputIndex < inputLength; inputIndex = inputIndex + blockNums * Simt::GetThreadNum()) {
        COM_T inputSegmentIndex = Simt::UintDiv(inputIndex, magic, shift);
        COM_T segmentOffset = inputIndex - inputSegmentIndex * innerDimSize;
        const Index outputSegmentIndex = segmentIdsGm[inputSegmentIndex];
        if (outputSegmentIndex >= static_cast<Index>(outputOuterDimSize) || outputSegmentIndex < 0) {
            continue;
        }
        const uint64_t outputIndex = outputSegmentIndex * innerDimSize + segmentOffset;
        TX curGmValue = xGm[inputIndex];
        // atomCas loop for multiply
        while (true) {
            TX oldVal = outputGm[outputIndex];
            TX newVal = oldVal * curGmValue;
            TX cas = Simt::AtomicCas(outputGm + outputIndex, oldVal, newVal);
            if (cas == oldVal) {
                break;
            }
        }
    }
}

// ======================== AtomCas Prod for half/bfloat16 using uint32_t CAS ========================
template <typename TX, typename Index, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeSegmentProdFp16(
    __gm__ TX* xGm, __gm__ Index* segmentIdsGm, __gm__ TX* outputGm, const uint32_t blockNums,
    const COM_T inputLength, const COM_T innerDimSize, const uint64_t outputOuterDimSize,
    const COM_T magic, const COM_T shift, const uint64_t totalOutputSize)
{
    for (COM_T inputIndex = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
         inputIndex < inputLength;
         inputIndex = inputIndex + blockNums * Simt::GetThreadNum()) {
        COM_T inputSegmentIndex = Simt::UintDiv(inputIndex, magic, shift);
        COM_T segmentOffset = inputIndex - inputSegmentIndex * innerDimSize;
        const Index outputSegmentIndex = segmentIdsGm[inputSegmentIndex];
        if (outputSegmentIndex < 0 || outputSegmentIndex >= static_cast<Index>(outputOuterDimSize)) {
            continue;
        }
        const uint64_t outputIndex = outputSegmentIndex * innerDimSize + segmentOffset;
        TX curGmValue = xGm[inputIndex];
 
        uint64_t pairIndex = outputIndex & ~1ULL;
        bool isLow = (outputIndex & 1) == 0;
 
        if (pairIndex + 1 >= totalOutputSize) {
            continue;
        }
        while (true) {
            TX oldVal1 = outputGm[pairIndex];
            TX oldVal2 = outputGm[pairIndex + 1];
            uint32_t oldU32;
            reinterpret_cast<TX*>(&oldU32)[0] = oldVal1;
            reinterpret_cast<TX*>(&oldU32)[1] = oldVal2;
            TX oldVal = isLow ? oldVal1 : oldVal2;
            TX neighbor = isLow ? oldVal2 : oldVal1;
            TX newVal = oldVal * curGmValue;
            uint32_t newU32;
            if (isLow) {
                reinterpret_cast<TX*>(&newU32)[0] = newVal;
                reinterpret_cast<TX*>(&newU32)[1] = neighbor;
            } else {
                reinterpret_cast<TX*>(&newU32)[0] = neighbor;
                reinterpret_cast<TX*>(&newU32)[1] = newVal;
            }
            uint32_t casU32 = Simt::AtomicCas(reinterpret_cast<__gm__ uint32_t*>(outputGm + pairIndex),
                                              oldU32, newU32);
            if (casU32 == oldU32) {
                break;
            }
        }
    }
}
 
// ======================== AtomCas Prod for half/bfloat16 using half2 ========================
template <typename TX, typename Index, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeSegmentProdHalf2Even(
    __gm__ TX* xGm, __gm__ Index* segmentIdsGm, __gm__ TX* outputGm, const uint32_t blockNums,
    const COM_T usedThread, const COM_T innerDimSize, const uint64_t outputOuterDimSize)
{
    for (COM_T inputIndex = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
         inputIndex < usedThread;
         inputIndex = inputIndex + blockNums * Simt::GetThreadNum()) {
        COM_T inputSegmentIndex = inputIndex * 2 / innerDimSize;
        COM_T segmentOffset = inputIndex * 2 - inputSegmentIndex * innerDimSize;
        const Index outputSegmentIndex = segmentIdsGm[inputSegmentIndex];
        if (outputSegmentIndex < 0 || outputSegmentIndex >= static_cast<Index>(outputOuterDimSize)) {
            continue;
        }
        const uint64_t outputIndex = outputSegmentIndex * innerDimSize + segmentOffset;
        TX curGmValue1 = xGm[inputIndex * 2];
        TX curGmValue2 = xGm[inputIndex * 2 + 1];
 
        while (true) {
            TX oldVal2 = outputGm[outputIndex + 1];
            TX oldVal1 = outputGm[outputIndex];
            uint32_t oldU32;
            uint32_t newU32;
            reinterpret_cast<TX*>(&oldU32)[1] = oldVal2;
            reinterpret_cast<TX*>(&oldU32)[0] = oldVal1;
            reinterpret_cast<TX*>(&newU32)[1] = oldVal2 * curGmValue2;
            reinterpret_cast<TX*>(&newU32)[0] = oldVal1 * curGmValue1;
            uint32_t casU32 = Simt::AtomicCas(reinterpret_cast<__gm__ uint32_t*>(outputGm + outputIndex),
                                              oldU32, newU32);
            if (casU32 == oldU32) {
                break;
            }
        }
    }
}
 
template <typename TX, typename COM_T>
__simt_callee__ inline void ComputeHalf2OddValues(
    __gm__ TX* xGm, const uint64_t outputRowIndex, const COM_T inRowIdx,
    const COM_T threadRowOffset, const COM_T eachRowThread, const COM_T innerDimSize,
    const bool outputRowOdd, TX& curGmValue1, TX& curGmValue2, uint64_t& outputIndex)
{
    if (outputRowOdd) {
        COM_T segmentOffset = (threadRowOffset == 0) ? static_cast<COM_T>(-1) : (threadRowOffset - 1) * 2 + 1;
        outputIndex = outputRowIndex + segmentOffset;
        if (threadRowOffset == 0) {
            curGmValue1 = GetIdentityOne<TX>();
            curGmValue2 = xGm[inRowIdx];
        } else {
            curGmValue1 = xGm[inRowIdx + segmentOffset];
            curGmValue2 = xGm[inRowIdx + segmentOffset + 1];
        }
    } else {
        COM_T segmentOffset = threadRowOffset * 2;
        outputIndex = outputRowIndex + segmentOffset;
        COM_T inIdx = inRowIdx + segmentOffset;
        if (threadRowOffset == eachRowThread - 1) {
            curGmValue1 = xGm[inIdx];
            curGmValue2 = GetIdentityOne<TX>();
        } else {
            curGmValue1 = xGm[inIdx];
            curGmValue2 = xGm[inIdx + 1];
        }
    }
}
 
template <typename TX, typename COM_T>
__simt_callee__ inline void CasHalf2OddBoundary(
    __gm__ TX* outputGm, __gm__ TX* xGm, const uint64_t outputRowIndex,
    const COM_T inRowIdx, const COM_T threadRowOffset, const COM_T eachRowThread,
    const COM_T innerDimSize, const bool outputRowOdd)
{
    uint64_t actualOutputIndex;
    TX actualCurGmValue;
    if (outputRowOdd && threadRowOffset == 0) {
        actualOutputIndex = outputRowIndex;
        actualCurGmValue = xGm[inRowIdx];
    } else if (!outputRowOdd && threadRowOffset == eachRowThread - 1) {
        actualOutputIndex = outputRowIndex + innerDimSize - 1;
        actualCurGmValue = xGm[inRowIdx + innerDimSize - 1];
    } else {
        return;
    }
    uint64_t pairIndex = (actualOutputIndex & 1) ? (actualOutputIndex - 1) : actualOutputIndex;
    bool isLow = !(actualOutputIndex & 1);
    while (true) {
        TX oldVal1 = outputGm[pairIndex];
        TX oldVal2 = outputGm[pairIndex + 1];
        uint32_t oldU32;
        reinterpret_cast<TX*>(&oldU32)[0] = oldVal1;
        reinterpret_cast<TX*>(&oldU32)[1] = oldVal2;
        TX oldVal = isLow ? oldVal1 : oldVal2;
        TX neighbor = isLow ? oldVal2 : oldVal1;
        TX newVal = oldVal * actualCurGmValue;
        uint32_t newU32;
        if (isLow) {
            reinterpret_cast<TX*>(&newU32)[0] = newVal;
            reinterpret_cast<TX*>(&newU32)[1] = neighbor;
        } else {
            reinterpret_cast<TX*>(&newU32)[0] = neighbor;
            reinterpret_cast<TX*>(&newU32)[1] = newVal;
        }
        uint32_t casU32 = Simt::AtomicCas(
            reinterpret_cast<__gm__ uint32_t*>(outputGm + pairIndex), oldU32, newU32);
        if (casU32 == oldU32) {
            break;
        }
    }
}
 
// Case 2: output row start address is odd OR innerDim (b) is odd
template <typename TX, typename Index, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_DIM_LAUNCH_BOUND) inline void SimtComputeSegmentProdHalf2Odd(
    __gm__ TX* xGm, __gm__ Index* segmentIdsGm, __gm__ TX* outputGm, const uint32_t blockNums,
    const COM_T usedThread, const COM_T innerDimSize, const uint64_t outputOuterDimSize,
    const COM_T eachRowThread, const uint64_t totalOutputSize)
{
    for (COM_T inputIndex = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
         inputIndex < usedThread;
         inputIndex = inputIndex + blockNums * Simt::GetThreadNum()) {
        COM_T inputSegmentIndex = inputIndex / eachRowThread;
        COM_T threadRowOffset = inputIndex - inputSegmentIndex * eachRowThread;
        const Index outputSegmentIndex = segmentIdsGm[inputSegmentIndex];
        if (outputSegmentIndex < 0 || outputSegmentIndex >= static_cast<Index>(outputOuterDimSize)) {
            continue;
        }
        const uint64_t outputRowIndex = outputSegmentIndex * innerDimSize;
        COM_T inRowIdx = inputSegmentIndex * innerDimSize;
        bool outputRowOdd = (reinterpret_cast<uint64_t>(outputGm + outputRowIndex) / sizeof(TX)) & 1;
        TX curGmValue1;
        TX curGmValue2;
        uint64_t outputIndex;
        ComputeHalf2OddValues(xGm, outputRowIndex, inRowIdx, threadRowOffset,
                              eachRowThread, innerDimSize, outputRowOdd,
                              curGmValue1, curGmValue2, outputIndex);
        if (outputIndex >= totalOutputSize || outputIndex + 1 >= totalOutputSize) {
            CasHalf2OddBoundary(outputGm, xGm, outputRowIndex, inRowIdx, threadRowOffset,
                                eachRowThread, innerDimSize, outputRowOdd);
            continue;
        }
        while (true) {
            TX oldVal1 = outputGm[outputIndex];
            TX oldVal2 = outputGm[outputIndex + 1];
            uint32_t oldU32;
            uint32_t newU32;
            reinterpret_cast<TX*>(&oldU32)[0] = oldVal1;
            reinterpret_cast<TX*>(&oldU32)[1] = oldVal2;
            reinterpret_cast<TX*>(&newU32)[0] = oldVal1 * curGmValue1;
            reinterpret_cast<TX*>(&newU32)[1] = oldVal2 * curGmValue2;
            uint32_t casU32 = Simt::AtomicCas(reinterpret_cast<__gm__ uint32_t*>(outputGm + outputIndex),
                                              oldU32, newU32);
            if (casU32 == oldU32) {
                break;
            }
        }
    }
}

// ======================== Kernel class ========================
template <typename T, typename Index>
class KernelUnsortedSegmentProd
{
public:
    __aicore__ inline KernelUnsortedSegmentProd(
        const UnsortedSegment::UnsortedSegmentSimtTilingData* tiling, TPipe* pipe)
        : td_(tiling), pipe_(pipe) {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
    {
        InitGmProd<T, InitGmOneValue<T>>(output, td_->outputOuterDim * td_->innerDim);

        xGm.SetGlobalBuffer((__gm__ T*)(x));
        segmentIdsGm.SetGlobalBuffer((__gm__ Index*)(segmentIds));
        outputGm.SetGlobalBuffer((__gm__ T*)(output));
    }

    template <typename COM_T>
    __aicore__ inline void SimtShellNormal(COM_T inputLength)
    {
        __gm__ T* input = (__gm__ T*)xGm.GetPhyAddr();
        __gm__ Index* segmentIds = (__gm__ Index*)segmentIdsGm.GetPhyAddr();
        __gm__ T* output = (__gm__ T*)outputGm.GetPhyAddr();

        uint32_t blockNums = GetBlockNum();
        COM_T innerDimSizeTmp = td_->innerDim;
        COM_T magic = 1;
        COM_T shift = 1;
        GetUintDivMagicAndShift(magic, shift, static_cast<COM_T>(innerDimSizeTmp));
        AscendC::Simt::VF_CALL<SimtComputeSegmentProdNormal<T, Index, COM_T>>(
            Simt::Dim3(static_cast<uint32_t>(td_->maxThread)), input, segmentIds, output,
            blockNums, inputLength, innerDimSizeTmp, td_->outputOuterDim, magic, shift);
    }

    template <typename COM_T>
    __aicore__ inline void SimtShellHalf2(COM_T inputLength)
    {
        __gm__ T* input = (__gm__ T*)xGm.GetPhyAddr();
        __gm__ Index* segmentIds = (__gm__ Index*)segmentIdsGm.GetPhyAddr();
        __gm__ T* output = (__gm__ T*)outputGm.GetPhyAddr();
 
        uint32_t blockNums = GetBlockNum();
        uint64_t innerDimSizeTmp = td_->innerDim;
        uint64_t inputOuterDim = td_->inputOuterDim;
        uint64_t totalOutputSize = td_->outputOuterDim * innerDimSizeTmp;

        // Check if output row start is even-aligned and innerDim is even
        bool innerDimEven = (innerDimSizeTmp % 2 == 0);
        bool outputStartEven = ((reinterpret_cast<uint64_t>(output) / sizeof(T)) % 2 == 0);

        if (innerDimEven && outputStartEven) {
            COM_T usedThread = static_cast<COM_T>(inputOuterDim * innerDimSizeTmp / 2);
            AscendC::Simt::VF_CALL<SimtComputeSegmentProdHalf2Even<T, Index, COM_T>>(
                Simt::Dim3(static_cast<uint32_t>(td_->maxThread)), input, segmentIds, output,
                blockNums, usedThread, static_cast<COM_T>(innerDimSizeTmp), td_->outputOuterDim);
        } else {
            COM_T eachRowThread = static_cast<COM_T>((innerDimSizeTmp + 2) / 2);
            COM_T usedThread = static_cast<COM_T>(inputOuterDim) * eachRowThread;
            AscendC::Simt::VF_CALL<SimtComputeSegmentProdHalf2Odd<T, Index, COM_T>>(
                Simt::Dim3(static_cast<uint32_t>(td_->maxThread)), input, segmentIds, output,
                blockNums, usedThread, static_cast<COM_T>(innerDimSizeTmp), td_->outputOuterDim,
                eachRowThread, totalOutputSize);
        }
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= GetBlockNum()) {
            return;
        }

        uint64_t inputLength = td_->inputOuterDim * td_->innerDim;

        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            // half/bfloat16: use half2/bfloat16x2_t atomCas
            if (inputLength > MAX_INT32_NUM) {
                SimtShellHalf2<uint64_t>(inputLength);
            } else {
                SimtShellHalf2<uint32_t>(inputLength);
            }
        } else {
            // int32/int64/uint32/uint64/float: use normal atomCas
            if (inputLength > MAX_INT32_NUM) {
                SimtShellNormal<uint64_t>(inputLength);
            } else {
                SimtShellNormal<uint32_t>(inputLength);
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    const UnsortedSegment::UnsortedSegmentSimtTilingData* td_;
    AscendC::GlobalTensor<T> xGm, outputGm;
    AscendC::GlobalTensor<Index> segmentIdsGm;
};

} // namespace UnsortedSegmentProd

#endif // UNSORTED_SEGMENT_PROD_H
