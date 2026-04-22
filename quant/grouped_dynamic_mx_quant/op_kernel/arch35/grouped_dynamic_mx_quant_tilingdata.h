/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_DYNAMIC_MX_QUANT_TILINGDATA_H
#define GROUPED_DYNAMIC_MX_QUANT_TILINGDATA_H

#include <cstdint>

struct GroupedDynamicMxQuantTilingData {
    int64_t totalCoreNum;
    int64_t usedCoreNum;           // 实际使用核数
    int64_t blockFactor;           // 头核处理数据量
    int64_t tailBlockFactor;       // 尾核处理数据量
    int64_t uo;                    // 切分轴上的循环次数
    int64_t maxUbCol;              // 单次循环要处理的数据大小
    int64_t ubFactor;              // 单次循环要处理的数据大小
    int64_t tailUbFactor;          // 尾循环要处理的数据大小
    int64_t blockSize;             // 量化数据块大小
    int64_t scaleAlg;              // OCP Microscaling Formats(Mx) Specification实现/cuBLAS实现，默认OCP实现
    int64_t preAxisSize;           // 输入row长度
    int64_t postAxisSize;          // 输入col长度
    float dstTypeMax;              // 目前仅支持0.0
};
#endif  // GROUPED_DYNAMIC_MX_QUANT_TILINGDATA_H
