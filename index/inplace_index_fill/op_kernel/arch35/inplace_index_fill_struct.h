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
 * \file inplace_index_fill_struct.h
 * \brief
 */

#ifndef INPLACE_INDEX_FILL_STRUCT_H_
#define INPLACE_INDEX_FILL_STRUCT_H_

namespace InplaceIndexFill {

// SIMT 模板常量定义
constexpr uint32_t SIMT_THREAD_NUM = 2048; // SIMT 每核线程数
constexpr uint8_t SIMT_DB_BUFFER = 2;      // 双缓冲数量

class InplaceIndexFillSimtTilingData {
public:
    int64_t tilingKeySimt = 0;   // tiling key，用于 kernel 侧模板选择
    int64_t p = 0;               // x 在 dim 轴前面所有维度的乘积（preDimProduct）
    int64_t n = 0;               // dim 轴的维度大小（dimSize），indices 值域 [0, n)
    int64_t q = 0;               // x 在 dim 轴后面所有维度的乘积（postDimProduct）
    int64_t indicesNum = 0;      // indices 数组的长度
    int64_t coreNum = 0;         // 平台总核数
    int64_t usedCoreNum = 0;     // 实际使用的核数
    int64_t simtUsedCoreNum = 0; // SIMT 计算估算需要的核数
};

// DenseIndices 模板的 tiling 数据
class InplaceIndexFillSimtDenseIndicesTilingData {
public:
    int64_t tilingKeySimt = 0;   // tiling key，用于 kernel 侧模板选择
    int64_t p = 0;               // x 在 dim 轴前面所有维度的乘积（preDimProduct）
    int64_t n = 0;               // dim 轴的维度大小（dimSize），indices 值域 [0, n)
    int64_t q = 0;               // x 在 dim 轴后面所有维度的乘积（postDimProduct）
    int64_t indicesNum = 0;      // indices 数组的长度
    int64_t coreNum = 0;         // 平台总核数
    int64_t usedCoreNum = 0;     // 实际使用的核数
    int64_t simtUsedCoreNum = 0; // SIMT 计算估算需要的核数
    int64_t indicesUbFactor = 0; // UB 中每次加载的 indices 数量（派生新增，放最后）
};

class InplaceIndexFillSimdTilingData {
public:
    int64_t tilingKey = 0;
    int64_t preDimProduct = 0;  // x在dim轴前面的维度乘积
    int64_t dimSize = 0;        // dim轴的维度
    int64_t postDimProduct = 0; // x在dim轴后面的轴维度乘积
    int64_t indicesNum = 0;     // indicesNum

    int64_t perBlockData = 0;
    int64_t tailBlockData = 0;
    int64_t tailBlockNum = 0;
    int64_t qBlockFactor = 0;
    int64_t qUsedCoreNum = 0;
    int64_t usedCoreNum = 0;

    // UB参数
    int64_t qBufferSize = 0;
    int64_t indicesBufferSize = 0;
    int64_t indicesUbFactor = 0;
    int64_t qUbFactor = 0;
    int64_t qLoopSize = 0;
    int64_t qUbTailFactor = 0;
};

} // namespace InplaceIndexFill
#endif // INPLACE_INDEX_FILL_STRUCT_H_