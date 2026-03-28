/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Our normal copyright notice. Below are our remarks.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef MAX_POOL_WITH_ARGMAX_V3_TILING_DATA_H
#define MAX_POOL_WITH_ARGMAX_V3_TILING_DATA_H

struct MaxPoolWithArgmaxV3TilingData {
    // ---- 输入输出维度 ----
    int64_t batchSize;          // N
    int64_t channels;           // C
    int64_t inputHeight;        // H_in
    int64_t inputWidth;         // W_in
    int64_t outputHeight;       // H_out
    int64_t outputWidth;        // W_out

    // ---- 池化参数 ----
    int32_t kernelH;            // kH
    int32_t kernelW;            // kW
    int32_t strideH;            // sH
    int32_t strideW;            // sW
    int32_t padH;               // padH
    int32_t padW;               // padW
    int32_t dilationH;          // dH
    int32_t dilationW;          // dW

    // ---- 多核切分参数 ----
    int64_t totalSlices;        // N * C（总独立切片数）
    int64_t slicesPerCore;      // 每核处理的切片数

    // ---- UB 切分参数 ----
    int64_t outputRowsPerTile;  // 每次 UB 迭代处理的输出行数
    int64_t inputWidthAligned;  // W_in 对齐到 32 字节后的元素数
    int64_t outputWidthAligned; // W_out 对齐到 32 字节后的元素数
};

#endif
