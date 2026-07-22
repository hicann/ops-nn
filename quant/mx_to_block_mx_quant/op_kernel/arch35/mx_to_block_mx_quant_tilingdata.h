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
 * \file mx_to_block_mx_quant_tilingdata.h
 * \brief TilingData structure passed from host to kernel for MxToBlockMxQuant.
 */

#ifndef OPS_NN_MX_TO_BLOCK_MX_QUANT_TILINGDATA_H
#define OPS_NN_MX_TO_BLOCK_MX_QUANT_TILINGDATA_H

#include <cstdint>

struct MxToBlockMxQuantTilingData {
    int64_t ubSize{0};              // UB 大小（字节）
    int64_t dstType{0};             // 输出数据类型
    int64_t totalCoreNum{0};        // AIV 核总数
    int64_t usedCoreNum{0};         // 实际使用的核数
    int64_t batchNum{0};            // batch 数 B
    int64_t rowNum{0};              // 单 batch 行数 M
    int64_t colNum{0};              // 列数 K
    int64_t colScaleNum{0};         // mxscale 列方向合并最后两维的数量
    int64_t rowMode{0};             // 行切分模式：0=ALIGNED(M%64==0) 1=NOT_ALIGNED
    int64_t rowBlockNumPerBatch{0}; // 单 batch 行方向基本块数 = CeilDiv(M, 64)
    int64_t colBlockNumPerBatch{0}; // 单 batch 列方向基本块数 = CeilDiv(K, 512)
    int64_t rowTailLenPerBatch{0};  // 单 batch 行尾块行数（M%64==0 时为 64）
    int64_t colTailLenPerBatch{0};  // 单 batch 列尾块列数（K%512==0 时为 512）
    int64_t totalBlockNum{0};       // 总块数 = B * rowBlockNumPerBatch * colBlockNumPerBatch
    int64_t headCoreBlockNum{0};    // 头核处理块数 = CeilDiv(totalBlockNum, usedCoreNum)
    int64_t tailCoreBlockNum{0};    // 尾核处理块数 = headCoreBlockNum - 1
    int64_t headCoreNum{0};         // 头核数量 = usedCoreNum - tailCoreNum
    int64_t tailCoreNum{0};         // 尾核数量 = headCoreBlockNum * usedCoreNum - totalBlockNum
};

#endif // OPS_NN_MX_TO_BLOCK_MX_QUANT_TILINGDATA_H
