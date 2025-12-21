/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file tool.h
 * \brief
 */
#ifndef TOOL_H
#define TOOL_H

#include "kernel_log.h"
#include "kernel_operator.h"
#include "kernel_utils.h"

using AscendC::AIC;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GetBlockNum;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::InitOutput;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::ONE_REPEAT_BYTE_SIZE;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::ToFloat;
using AscendC::WaitFlag;

namespace QuantBatchMatmulV4 {
constexpr uint64_t SYNC_MODE0 = 0;
constexpr uint64_t SYNC_MODE2 = 2;
constexpr uint64_t SYNC_AIV_ONLY_ALL_FLAG = 6;
constexpr uint64_t SYNC_AIC_ONLY_ALL_FLAG = 7;
constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
constexpr uint64_t SYNC_AIV_ONLY_CONFIG = 1 | (SYNC_MODE0 << 4) | (SYNC_AIV_ONLY_ALL_FLAG << 8);
constexpr uint64_t SYNC_AIC_ONLY_CONFIG = 1 | (SYNC_MODE0 << 4) | (SYNC_AIC_ONLY_ALL_FLAG << 8);
constexpr uint64_t SYNC_AIV_AIC_CONFIG = 1 | (SYNC_MODE2 << 4) | (SYNC_AIV_AIC_FLAG << 8);
constexpr uint64_t SYNC_AIC_AIV_CONFIG = 1 | (SYNC_MODE2 << 4) | (SYNC_AIC_AIV_FLAG << 8);
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t SINGLE_BUFFER_NUM = 1;
constexpr int32_t FP32_MAX_MASK_SIZE = 64;
constexpr int32_t FP32_MASK_BLK_NUM = 8;
constexpr uint32_t FP16_MASK_SIZE = 128;
constexpr uint32_t FP32_BLOCK_SIZE = 8;
constexpr uint32_t FP16_BLOCK_SIZE = 16;
constexpr uint32_t FLOAT_DATA_BENCHMARK = 256;
constexpr uint32_t HALF_DATA_BENCHMARK = 512;
constexpr uint32_t INT8_DATA_BENCHMARK = 1024;
constexpr uint32_t INT4_DATA_BENCK_MARK = 2048;

// vector指令一个repeat最多处理256B，包含8个Block，repeat_stride最大为8
constexpr uint32_t VEC_REPEAT_MAX_STRIDE = 8;

template <typename T> __aicore__ inline T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

template <typename T> __aicore__ inline T CeilDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T> __aicore__ inline T FloorDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return a / b;
}

template <typename T> __aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t dim1,
    uint32_t dim0, uint32_t fullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (fullDim0 - dim0) * sizeof(T);
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    if (dim0 % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(dim0, static_cast<uint32_t>(32 / sizeof(T))) - dim0;
        padParams.paddingValue = 0;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t blockCount,
    uint32_t blockLen, uint32_t dstInnerLength, uint32_t srcInnerLength)
{
#if defined(__CCE_KT_TEST__)
    ASCENDC_ASSERT(dstInnerLength >= blockLen, {
        KERNEL_LOG(KERNEL_ERROR, "dstInnerLength[%d] should be larger than blockLen[%d].", dstInnerLength, blockLen);
    });
#endif
    DataCopyExtParams params;
    params.blockCount = blockCount;
    params.blockLen = blockLen * sizeof(T);
    params.srcStride = (srcInnerLength - blockLen) * sizeof(T);
    params.dstStride = (dstInnerLength - blockLen) * sizeof(T) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    if (blockLen % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(blockLen, static_cast<uint32_t>(32 / sizeof(T))) - blockLen;
        padParams.paddingValue = 0;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint32_t dim1,
    uint32_t dim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = 0;
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
    }
    DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1, uint32_t dim0,
    uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = CeilDiv((srcFullDim0 - dim0) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ inline void InitAtomicAddr(const GlobalTensor<T> dst, uint64_t initTotalSize, int32_t curBlockIdx)
{
    if ASCEND_IS_AIC {
        return;
    }
    static constexpr uint64_t addrAlignBlock = 512 / sizeof(T);
    uint64_t initBaseSize = CeilAlign(CeilDiv(initTotalSize, static_cast<uint64_t>(GetBlockNum() * 2)), addrAlignBlock);
    uint64_t initOffset = curBlockIdx * initBaseSize;
    if (initOffset < initTotalSize) {
        uint64_t initRealSize = initBaseSize;
        if (initOffset + initRealSize > initTotalSize) {
            initRealSize = initTotalSize - initOffset;
        }
        InitOutput<T>(dst[initOffset], initRealSize, (T)0.0);

        // 后续运算的mte2要等当前initOutput的v和mte3结束才行
        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        eventId = GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }
}

template <HardEvent event> class SyncProcessor {
public:
    __aicore__ inline SyncProcessor() {}
    TEventID eventIds_[DOUBLE_BUFFER_NUM];
    uint32_t doubleBufferNum_ = DOUBLE_BUFFER_NUM;
    uint64_t setTaskId_ = 0;
    uint64_t waitTaskId_ = 0;

    __aicore__ inline void Init(int32_t doubleBufferNum)
    {
        doubleBufferNum_ = doubleBufferNum;
        if (doubleBufferNum_ > DOUBLE_BUFFER_NUM) {
            doubleBufferNum_ = DOUBLE_BUFFER_NUM;
        }
        for (int32_t eventIdx = 0; eventIdx < doubleBufferNum_; eventIdx++) {
            eventIds_[eventIdx] = GetTPipePtr()->AllocEventID<event>();
        }
    };

    __aicore__ uint64_t GetBufferId()
    {
        return setTaskId_ % doubleBufferNum_;
    }

    __aicore__ inline void ReleaseTensor()
    {
        SetFlag<event>(eventIds_[setTaskId_ & (doubleBufferNum_ - 1)]);
        setTaskId_++;
    };

    __aicore__ inline void AllocTensor()
    {
        if (setTaskId_ < doubleBufferNum_) {
            return;
        }

        WaitFlag<event>(eventIds_[waitTaskId_ & (doubleBufferNum_ - 1)]);
        waitTaskId_++;
    };

    __aicore__ inline void Destory()
    {
        for (; waitTaskId_ < setTaskId_; waitTaskId_++) {
            WaitFlag<event>(eventIds_[waitTaskId_ & (doubleBufferNum_ - 1)]);
        }

        for (int32_t eventIdx = 0; eventIdx < doubleBufferNum_; eventIdx++) {
            GetTPipePtr()->ReleaseEventID<event>(eventIds_[eventIdx]);
        }
        setTaskId_ = 0;
        waitTaskId_ = 0;
    };
};
}
#endif