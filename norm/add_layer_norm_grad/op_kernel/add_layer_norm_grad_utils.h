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
 * \file add_layer_norm_grad_utils.h
 * \brief
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_UTILS_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_UTILS_H
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "kernel_tiling/kernel_tiling.h"

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

namespace AddLayerNormGrad {
using namespace AscendC;

constexpr uint32_t TAIL_BUFFER_SIZE = 32;
constexpr uint32_t REDUCE_SRC0_REPSTRIDE = 8;
constexpr uint32_t REDUCE_BUFFER_STAND_SIZE = 64;
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_AlIGN = 32;
constexpr float ZERO = 0;
constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr uint32_t MAX_COPY_LENTH = 2000;
constexpr int32_t CONSTANT_TWO = 2;
constexpr int32_t CONSTANT_EIGHT = 8;
constexpr uint32_t FULL_ALIGN_BLOCK = 64;
constexpr uint32_t MASK_BIT_COUNT = 64;
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t FP16_SIZE = 2;
constexpr uint64_t DUPLICATE_FP32_MASK = 0x8080808080808080;
constexpr uint64_t DUPLICATE_FP16_MASK = (static_cast<uint64_t>(0x80) << 8) | (static_cast<uint64_t>(0x80) << 24) | (static_cast<uint64_t>(0x80) << 40) | (static_cast<uint64_t>(0x80) << 56);

// define num_last_dim => reduce(gamma_axis)
// define num_first_dim => reduce(input_axis except gamma_axis)

__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
#if __CCE_AICORE__ == 220
    return true;
#else
    return false;
#endif
}

__aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
__aicore__ inline uint32_t MAX(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}
__aicore__ inline uint32_t ROUND_UP(uint32_t x, uint32_t block_number)
{
    if (block_number > 0) {
        return (x + block_number - 1) / block_number * block_number;
    }
    return 0;
}
__aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

uint32_t FLOOR_DIV(uint32_t x, uint32_t y)
{
    if (y > 0 && x + 1 - y > 0) {
        return (x + 1 - y) / y;
    }
    return 0;
}

template <bool forAtomicAdd = false, typename T>
__aicore__ inline void SafeDataCopy(
    const AscendC::GlobalTensor<T>& dstGlobal, const AscendC::LocalTensor<T>& srcLocal, const int64_t& calCount,
    bool recoverUbTailFormat = false)
{
    constexpr int typeSize = sizeof(T);                                // 元素字节数
    constexpr int numElemsPerBlock = AscendC::ONE_BLK_SIZE / typeSize; // 32byte元素数
    if constexpr (IsDataCopyPadSupport() && sizeof(T) < 8) { // 如果支持DataCopyPad则直接DataCopyPad拷贝
        AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(calCount * typeSize), 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    } else {
        if (likely(!(calCount % numElemsPerBlock))) { // 对齐则直接DataCopy拷贝
            struct AscendC::DataCopyParams copyParams;
            copyParams.blockLen = calCount / AscendC::AscendCUtils::GetC0Count(typeSize);
            DataCopy(dstGlobal, srcLocal, copyParams);
        } else { // 如果既不支持DataCopyPad也不对齐则地址回退拷贝
            const int numAlignedBlocks = calCount / numElemsPerBlock * numElemsPerBlock; // 对齐部分
            if (calCount * typeSize < AscendC::ONE_BLK_SIZE) {
                DataCopy(dstGlobal, srcLocal, numElemsPerBlock);
                return; // 此处依然有内存踩踏
            }
            DataCopy(dstGlobal, srcLocal, numAlignedBlocks);
            event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventID);
            const int rollbackEleCount = calCount - numAlignedBlocks;          // 计算需要回退处理byte数
            const size_t rollbackDstIdx = numAlignedBlocks - numElemsPerBlock; // 操作回退的block元素索引
            const size_t rollbackSrcIdx = rollbackDstIdx + rollbackEleCount;   // 回退来源元素索引
            if constexpr (!forAtomicAdd) {
                for (int i = 0; i < numElemsPerBlock; ++i) { // 将整个非对齐的尾块以一个block的size复制填充到前一个block中
                    srcLocal.SetValue((rollbackDstIdx + i), srcLocal.GetValue(rollbackSrcIdx + i)); // 重造local buf
                }
            } else {
                const size_t setZeroEleCount = numElemsPerBlock - rollbackEleCount; // 需要置0的元素量
                for (int i = 0; i < setZeroEleCount; ++i) {
                    srcLocal.SetValue((rollbackDstIdx + i), 0); // Atomic模式下，回退部分置0，使得回退部分不会重复加
                }
                for (int i = setZeroEleCount; i < numElemsPerBlock; ++i) { // 将整个非对齐的尾块复制填充到前一个block中
                    srcLocal.SetValue((rollbackDstIdx + i), srcLocal.GetValue(rollbackSrcIdx + i)); // 重造local buf
                }
                DataCopy(dstGlobal[calCount - numElemsPerBlock], srcLocal[rollbackDstIdx], numElemsPerBlock);
                return; // AtomicAdd模式下 暂不支持recoverUbTailFormat，待后续扩展支持
            }
            DataCopy(dstGlobal[calCount - numElemsPerBlock], srcLocal[rollbackDstIdx], numElemsPerBlock);
            if (recoverUbTailFormat) { // 还原回滚现场
                event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
                DataCopy(
                    srcLocal[rollbackDstIdx], dstGlobal[rollbackDstIdx], numElemsPerBlock); // 还原用于回退的block内容
            }
        }
    }
}

template <typename T>
__aicore__ inline uint64_t* DuplicateWithMask(uint32_t maskCount, uint32_t maskCalcCount, uint64_t* mask)
{
    if(sizeof(T) == FP16_SIZE) {
        mask[0] = DUPLICATE_FP16_MASK;
        mask[1] = DUPLICATE_FP16_MASK;
    } else {
        mask[0] = DUPLICATE_FP32_MASK;
        mask[1] = 0;
    }
    for (uint32_t idx = 0; idx < maskCount - 1; idx++) {
        mask[0] |= mask[0] >> 1;
    }
    if (sizeof(T) == FP16_SIZE) {
        mask[1] = mask[0];
    }
    if(sizeof(T) == FP16_SIZE) {
        if (maskCalcCount < MASK_BIT_COUNT) {
            mask[1] = 0;
            mask[0] &= (static_cast<uint64_t>(1) << maskCalcCount) - 1;
        } else if (maskCalcCount < FP16_SIZE * MASK_BIT_COUNT) {
            mask[1] &= (static_cast<uint64_t>(1) << (maskCalcCount - 64)) - 1;
        }
    } else if (sizeof(T) == FP32_SIZE) {
        if (maskCalcCount < MASK_BIT_COUNT) {
            mask[0] &= (static_cast<uint64_t>(1) << maskCalcCount) - 1;
        }
    }
    return mask;
}

__aicore__ inline float ReduceSumCustom(const LocalTensor<float>& src_local, int32_t count)
{
    ReduceSum(src_local, src_local, src_local, count);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    WaitFlag<HardEvent::V_S>(event_v_s);
    float rstd_value = src_local.GetValue(0);
    return rstd_value;
}

template <typename T>
__aicore__ inline void DataCopyCustomWithSmallShape(
    GlobalTensor<T>& dstTensor, LocalTensor<T>& srcTensor, const uint32_t processElem, const uint32_t offset,
    const uint16_t blockCount)
{
    int32_t blockNumel = BLOCK_AlIGN / sizeof(T);
    int32_t blockNum = processElem / blockNumel;
    int32_t tail = processElem % blockNumel;
    int32_t blkLength = blockNum * blockNumel;
    int32_t repeatTimes = blockCount / CONSTANT_EIGHT;
    int32_t repeatTimesTail = blockCount % CONSTANT_EIGHT;
    bool canCompute = tail * blockCount > blockNumel;   //如果cancompute为false理论上必越界
    if(tail == 0) {
        return;
    } else {
        if (repeatTimes != 0) {
            uint64_t mask[2];
            DuplicateWithMask<T>(blockNumel - tail, 256 / sizeof(T), mask);
            Duplicate(srcTensor, static_cast<T>(0.0), mask, repeatTimes, 1, CONSTANT_EIGHT);
        }
        if (repeatTimesTail != 0) {
            uint64_t maskTail[2];
            DuplicateWithMask<T>(blockNumel - tail, repeatTimesTail * 32 / sizeof(T), maskTail);
            if (maskTail[0] != 0 || maskTail[1] != 0) {
                Duplicate(srcTensor[repeatTimes * BLOCK_AlIGN / sizeof(T) * CONSTANT_EIGHT], static_cast<T>(0.0), maskTail, 1, 1, CONSTANT_EIGHT);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t idx = 0; idx < blockCount; idx++) {
        uint32_t curOffset = offset + idx * processElem;
        if (idx == blockCount - 1 && canCompute) {
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            for (uint32_t i = 0; i < blockNumel; i++) {   // 这里换别的搬运方法会卡死
                T tensorValue = srcTensor.GetValue(idx * ROUND_UP(processElem, blockNumel) + processElem - blockNumel + i);
                srcTensor.SetValue((idx - 1) * ROUND_UP(processElem, blockNumel) + i, tensorValue);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            SetAtomicAdd<T>();
            DataCopy(dstTensor[curOffset + processElem - blockNumel], srcTensor[(idx - 1) * ROUND_UP(processElem, blockNumel)],
            blockNumel);
            SetAtomicNone();
        } else {
            SetAtomicAdd<T>();
            DataCopy(dstTensor[curOffset], srcTensor[idx * ROUND_UP(processElem, blockNumel)], blockNumel);
            SetAtomicNone();
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyCustom(
    GlobalTensor<T>& dstTensor, LocalTensor<T>& srcTensor, const uint32_t processElem, const uint32_t offset, bool is_float,
    const uint16_t blockCount)
{
#if __CCE_AICORE__ == 220
    DataCopyParams dataCopyParamsND{blockCount, (uint16_t)(processElem * sizeof(T)), 0, 0};
    DataCopyPad(dstTensor[offset], srcTensor, dataCopyParamsND);
#else
    int32_t blockNumel = is_float ? BLOCK_AlIGN / sizeof(float) : BLOCK_AlIGN / sizeof(T);
    int32_t blockNum = processElem / blockNumel;
    int32_t tail = processElem % blockNumel;
    int32_t blkLength = blockNum * blockNumel;
    if (blockNum == 0) {
        DataCopyCustomWithSmallShape(dstTensor, srcTensor, processElem, offset, blockCount);
        return;
    }
    for (uint32_t idx = 0; idx < blockCount; idx++) {
        uint32_t curOffset = offset + idx * processElem;
        DataCopy(dstTensor[curOffset], srcTensor[idx * ROUND_UP(processElem, blockNumel)], blkLength);
        if (tail != 0) {
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            for (uint32_t i = 0; i < blockNumel; i++) {
                T tensorValue =
                    srcTensor.GetValue(idx * ROUND_UP(processElem, blockNumel) + processElem - blockNumel + i);
                srcTensor.SetValue(idx * ROUND_UP(processElem, blockNumel) + i, tensorValue);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(
                dstTensor[curOffset + processElem - blockNumel], srcTensor[idx * ROUND_UP(processElem, blockNumel)],
                blockNumel);
        }
    }
#endif
}

__aicore__ inline void DataCopyAutomicAdd(
    GlobalTensor<float> dstTensor, const LocalTensor<float> srcTensor, const uint32_t processElem,
    const uint32_t offset, const uint16_t blockCount)
{
#if __CCE_AICORE__ == 220
    DataCopyParams dataCopyParamsND{blockCount, (uint16_t)(processElem * sizeof(float)), 0, 0};
    DataCopyPad(dstTensor[offset], srcTensor, dataCopyParamsND);
#else
    SafeDataCopy<true>(dstTensor[offset], srcTensor, blockCount * processElem);
#endif
}
__aicore__ inline void InitGmData(
    GlobalTensor<float> outputPdGammaGm, GlobalTensor<float> outputPdBetaGm, const uint32_t dDimNum,
    LocalTensor<float> dbeta, uint32_t elemWithDInUBFp32)
{
#if __CCE_AICORE__ == 220
    InitOutput<float>(outputPdGammaGm, dDimNum, 0);
    InitOutput<float>(outputPdBetaGm, dDimNum, 0);
#else
    uint32_t maxCopyLenth = elemWithDInUBFp32 - FLOAT_BLOCK_ELEM;
    maxCopyLenth = MAX(maxCopyLenth, FLOAT_BLOCK_ELEM);
    uint32_t loopNum = dDimNum / maxCopyLenth;
    uint32_t tail = dDimNum % maxCopyLenth;
    if (loopNum == 0) {
        Duplicate(dbeta, 0.0f, ROUND_UP(dDimNum, FLOAT_BLOCK_ELEM));
    } else {
        Duplicate(dbeta, 0.0f, elemWithDInUBFp32);
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t idx = 0; idx < loopNum; idx++) {
        uint32_t curOffset = idx * maxCopyLenth;
        SafeDataCopy(outputPdGammaGm[curOffset], dbeta, maxCopyLenth);
        SafeDataCopy(outputPdBetaGm[curOffset], dbeta, maxCopyLenth);
    }
    if (tail != 0) {
        if (loopNum >= 1) {
            uint32_t rollbackTail = FLOAT_BLOCK_ELEM + tail;
            SafeDataCopy(outputPdGammaGm[maxCopyLenth * loopNum - FLOAT_BLOCK_ELEM], dbeta, rollbackTail);
            SafeDataCopy(outputPdBetaGm[maxCopyLenth * loopNum - FLOAT_BLOCK_ELEM], dbeta, rollbackTail);
        } else {
            SafeDataCopy(outputPdGammaGm[maxCopyLenth * loopNum], dbeta, tail);
            SafeDataCopy(outputPdBetaGm[maxCopyLenth * loopNum], dbeta, tail);
        }
    }
    PipeBarrier<PIPE_ALL>();
#endif
}

template <typename T>
__aicore__ inline void InitSingleData(GlobalTensor<T> dstGmTensor, LocalTensor<T> srcLocalTensor, uint32_t elemWithDInUBFp32)
{
#if __CCE_AICORE__ == 220
    InitOutput<T>(dstGmTensor, elemWithDInUBFp32, 0);
#else
    uint32_t alignLength = 32 / sizeof(T);
    uint32_t maxCopyLenth = ROUND_UP(elemWithDInUBFp32 + 1, alignLength) - alignLength;
    maxCopyLenth = MAX(maxCopyLenth, alignLength);
    uint32_t loopNum = elemWithDInUBFp32 / maxCopyLenth;
    uint32_t tail = elemWithDInUBFp32 % maxCopyLenth;

    Duplicate<T>(srcLocalTensor, 0.0, maxCopyLenth);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t idx = 0; idx < loopNum; idx++) {
        uint32_t curOffset = idx * maxCopyLenth;
        SafeDataCopy(dstGmTensor[curOffset], srcLocalTensor, maxCopyLenth);
    }
    if (tail != 0) {
        if (loopNum >= 1) {
            uint32_t rollbackTail = alignLength + tail;
            SafeDataCopy(dstGmTensor[maxCopyLenth * loopNum - alignLength], srcLocalTensor, rollbackTail);
        } else {
            for (uint32_t i = 0; i < tail; i++) {
                dstGmTensor.SetValue(i, static_cast<T>(0.0));
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
#endif
}
}

#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_UTILS_H