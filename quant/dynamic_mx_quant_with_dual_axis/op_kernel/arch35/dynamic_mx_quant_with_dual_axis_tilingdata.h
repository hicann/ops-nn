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
 * \file dynamic_mx_quant_with_dual_axis_tilingdata.h
 * \brief
 */

#ifndef OPS_NN_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
#define OPS_NN_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H

#include <cstdint>

struct DynamicMxQuantWithDualAxisTilingData {
    int64_t totalCoreNum{0};           // 总核数
    int64_t usedCoreNum{0};            // 实际使用的核数
    int64_t roundMode{0};              // 数据类型转换的模式
    int64_t dstType{0};                // 输出y的数据类型
    int64_t scaleAlg{0};               // CuBlas实现或OCP实现，默认OCP实现
    int64_t blockSize{0};              //
    int64_t dim0{0};                   //
    int64_t dimNeg2{0};                //
    int64_t dimNeg1{0};                //
    int64_t blockW{0};                 // 所切基本块的宽
    int64_t splitBlockH{0};            // 所切基本块的高
    uint64_t tilingKey{0};             //
    int64_t dimNeg2Tail{0};            // -2轴方向尾块
    int64_t dimNeg1Tail{0};            // -1轴方向尾块
    int64_t dimNeg2SplitBlockNum{0};   // -2轴切分基本块的个数
    int64_t dimNeg1BlockNum{0};        // 尾轴切分基本块的个数
    int64_t blockPerHeadCore{0};       // 正常核计算的task数
    int64_t blockPerTailCore{0};       // 尾核计算的task数
    int64_t headCoreNum{0};            // 正常核个数
    int64_t dimNeg2IsOdd{0};           // 量化轴block数是否是奇数
    int64_t dimNeg1IsOdd{0};           // 尾轴block数是否为奇数
    int64_t dimNeg1IsPad{0};           // 尾轴是否需要32对齐
    int64_t blockCountPerBatch{0};     // 一个batch轴切分块数
    int64_t scale1ColCountPerBatch{0}; // 一个batch轴-1轴的scale列数
    int64_t scale2RowCountPerBatch{0}; // 一个batch轴-2轴的scale的行数
    float dstTypeMax{0.0f};    // 目标数据类型最大值 (scaleAlg=2时使用，0.0表示使用默认最大值)
    float invDstTypeMax{0.0f}; // 1 / dstTypeMax (预计算的倒数，用于kernel中避免浮点除法)
};
#endif // OPS_NN_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
