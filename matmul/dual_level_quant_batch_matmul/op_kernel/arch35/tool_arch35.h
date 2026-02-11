/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tool_arch35.h
 * \brief
 */

#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_TOOL_ARCH35_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_TOOL_ARCH35_H

#include <limits>

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_utils.h"
#include "op_kernel/math_util.h"
#include "lib/matmul_intf.h"

using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::PaddingMode;

namespace DualLevelQuantBatchMatmul::Arch35 {
// buffer相关定义
static constexpr int32_t DOUBLE_BUFFER_NUM = 2;
static constexpr uint32_t L1_BUFFER_SIZE_BYTE = 512 * 1024;
static constexpr uint32_t L0C_BUFFER_SIZE_BYTE = 256 * 1024;
static constexpr uint32_t L0A_BUFFER_SIZE_BYTE = 64 * 1024;
static constexpr uint32_t L0B_BUFFER_SIZE_BYTE = 64 * 1024;

// 参数约束定义
static constexpr uint64_t MX_GROUPSIZE = 32;

template <typename T>
__aicore__ inline void DataCopyPad2D(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t blockCount, uint32_t blockLen,
    uint32_t dstInnerLength, uint32_t srcInnerLength)
{
    DataCopyExtParams params;
    params.blockCount = blockCount;
    params.blockLen = blockLen * sizeof(T);
    params.srcStride = (srcInnerLength - blockLen) * sizeof(T);
    params.dstStride = (dstInnerLength - blockLen) * sizeof(T) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    if (blockLen % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = Ops::Base::CeilAlign(blockLen, static_cast<uint32_t>(32 / sizeof(T))) - blockLen;
        padParams.paddingValue = 0;
    }

    if constexpr (
        IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, fp4x2_e1m2_t>::value) {
        // 4bit场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T, PaddingMode mode = PaddingMode::Normal>
__aicore__ inline void DataCopyPad2D(
    const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t dim1, uint32_t dim0, uint32_t srcFullDim0,
    uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    uint32_t dim0BlockNum = Ops::Base::CeilAlign(dim0, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(T)));
    params.srcStride =
        Ops::Base::CeilDiv((srcFullDim0 - dim0BlockNum) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
    }
    DataCopyPad<T, mode>(dst, src, params);
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

} // namespace DualLevelQuantBatchMatmul::Arch35
#endif