/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_experimental_tool.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_TOOL_H
#define WEIGHT_QUANT_BATCH_MATMUL_EXPERIMENT_TOOL_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"

using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;

namespace WeightQuantBatchMatmulExperimental {
static constexpr uint64_t FP32_MASK_BLK_NUM = 8;
static constexpr uint64_t FP32_MAX_MASK_SIZE = 64;
static constexpr uint64_t FP32_BLOCK_SIZE = 8;
static constexpr uint64_t FP16_BLOCK_SIZE = 16;
static constexpr uint64_t SYNC_MODE2 = 2;
static constexpr uint64_t SYNC_MODE0 = 0;

static constexpr uint64_t SYNC_AIV_ONLY_A_UNFOLD_FLAG = 3;
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
static constexpr uint64_t SYNC_AIV_ONLY_AMAX_FLAG = 6;
static constexpr uint64_t SYNC_AIC_ONLY_AUNFOLD_FLAG = 7;
static constexpr uint64_t UNFLOD_TIMES = 3;

// vector指令一个repeat最多处理256B，包含8个Block，repeat_stride最大为8
static constexpr uint32_t VEC_REPEAT_MAX_STRIDE = 8;

static constexpr uint64_t DOUBLE_BUFFER_NUM = 2;

struct A16W4MsdConstParam {
    uint64_t kaL1Size;
    uint64_t kbL1Size;
    uint64_t kUbSize;
    uint64_t kGmSize;
    uint64_t nGmSize;
    uint64_t mGmSize;
    uint64_t nL1Size;
    uint64_t mL1Size;
    uint64_t groupSize;
};

template <typename T>
__aicore__ constexpr uint32_t GetKBUnit()
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return 2048;  // 2048个int4是1kb
    }
    return 1024 / sizeof(T);  // 1024B除size获取1kb的element数量
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t blockCount,
                                     uint32_t blockLen, uint32_t dstInnerLength, uint32_t srcInnerLength)
{
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
    if constexpr (IsSameType<T, int4b_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
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
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint32_t dim1,
                                     uint32_t dim0, uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = CeilDiv((srcFullDim0 - dim0) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
    }
    DataCopyPad(dst, src, params);
}
}  // namespace WeightQuantBatchMatmulExperimental
#endif