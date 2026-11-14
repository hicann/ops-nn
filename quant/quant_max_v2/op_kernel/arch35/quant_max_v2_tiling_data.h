/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// QuantMaxV2 的 Host 与 Kernel 共用 tiling 数据结构。

#ifndef QUANT_MAX_V2_TILING_DATA_H
#define QUANT_MAX_V2_TILING_DATA_H
#include <cstdint>

struct QuantMaxV2TilingData {
    int64_t totalCoreNum;
    int64_t actualCoreNum;
    int64_t blockTailCoreNum;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t ubFactor;
    int64_t dim1;
    int64_t numGroups;
    // 分组模式已由模板参数确定，复用原有 8 字节存储两个整数定点倒数，不增加传输大小。
    uint32_t coreDivReciprocal;
    uint32_t tileDivReciprocal;
    float dstTypeMax;
    // 仅动态不分组使用的形状排程，分组模板不读取这些字段。
    int64_t tensorBlockLoopNum;
    int64_t tensorBlockLoopTail;
    int64_t tensorBlockTileCount;
    int64_t tensorTailLoopNum;
    int64_t tensorTailLoopTail;
    int64_t tensorTailTileCount;
    int64_t tensorTailCoreStart;
    // 动态 bf16 位级 tile 最大值 16 位暂存区的字节上限；仅 TENSOR 路消费。
    // Host 在 CalcUbFactor 内与 ubFactor 联立做不动点求解（tile 越小 tile 数越多、暂存越大），
    // 收敛后下发，kernel 侧不再现场推导（GROUP 路用同公式在 Init 内自算）。
    int64_t tensorBf16TileMaxBufferSize;
};

static_assert(sizeof(QuantMaxV2TilingData) == 144, "保持原 tiling 传输大小");
#endif // QUANT_MAX_V2_TILING_DATA_H
