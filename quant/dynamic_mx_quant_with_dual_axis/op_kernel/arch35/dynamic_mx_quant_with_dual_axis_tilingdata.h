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
    int64_t totalCoreNum;           // 总核数
    int64_t usedCoreNum;            // 实际使用的核数
    int64_t roundMode;              // 数据类型转换的模式
    int64_t dstType;                // 输出y的数据类型
    int64_t scaleAlg;               // CuBlas实现或OCP实现，默认OCP实现
    int64_t blockSize;              //
    int64_t dim0;                   //
    int64_t dimNeg2;                //
    int64_t dimNeg1;                //
    int64_t blockW;                 // 所切基本块的宽
    int64_t splitBlockH;            // 所切基本块的高
    int64_t tilingKey;              //
    int64_t dimNeg2Tail;            // -2轴方向尾块
    int64_t dimNeg1Tail;            // -1轴方向尾块
    int64_t dimNeg2SplitBlockNum;   // -2轴切分基本块的个数
    int64_t dimNeg1BlockNum;        // 尾轴切分基本块的个数
    int64_t blockPerHeadCore;       // 正常核计算的task数
    int64_t blockPerTailCore;       // 尾核计算的task数
    int64_t headCoreNum;            // 正常核个数
    int64_t dimNeg2IsOdd;           // 量化轴block数是否是奇数
    int64_t dimNeg1IsOdd;           // 尾轴block数是否为奇数
    int64_t dimNeg1IsPad;           // 尾轴是否需要32对齐
    int64_t blockCountPerBatch;     // 一个batch轴切分块数
    int64_t scale1ColCountPerBatch; // 一个batch轴-1轴的scale列数
    int64_t scale2RowCountPerBatch; // 一个batch轴-2轴的scale的行数
};
#endif // OPS_NN_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
