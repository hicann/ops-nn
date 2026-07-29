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
 * \file in_infer_v2_tiling_data.h
 * \brief tiling data struct shared by host tiling and arch35 kernel（ND 单路径字段）
 *
 * y              = (x - mean) * (gamma / sqrt(variance + epsilon)) + beta   （gamma/beta 可选）
 *                  无 gamma/beta 时 y = (x - mean) / sqrt(variance + epsilon)
 * batch_mean     = mean / batch_variance = variance                       （透传拷贝）
 *
 * 单条计算路径（tilingKey=0）：ND/NCHW [N,C,R...] plane 连续，per-plane 标量广播。
 *
 * 统一切分模型：units 个独立单元按 unitCores 均分（former/latter），单元不足时
 * 单元内 inner 维再切 innerCores 份；blockIdx = unitIdx * innerCores + innerIdx。无 workspace。
 */

#ifndef IN_INFER_V2_TILING_DATA_H
#define IN_INFER_V2_TILING_DATA_H

#include <cstdint>

struct INInferV2TilingData {
    int64_t numN;          // N
    int64_t numC;          // C（dim1）
    int64_t innerSize;     // R = prod(d2:)
    int64_t units;         // plane 数 = N*C
    int64_t unitCores;     // 单元维切分核数
    int64_t formerCoreNum; // 前 formerCoreNum 核每核 formerUnits 个单元，其余 latterUnits 个
    int64_t formerUnits;
    int64_t latterUnits;
    int64_t innerCores;   // 单元内 inner 维切分份数（单元 < 核数且 inner 够大时 >1）
    int64_t innerPerCore; // 每份的 inner 长度，末尾截断
    int64_t ubTileSize;   // UB tile 元素数（VL=64 对齐）
    float epsilon;
    int64_t hasGammaBeta; // 0/1：gamma/beta 是否同时存在
    int64_t hasBatchMean; // 0/1：batch_mean 输出是否存在
    int64_t hasBatchVar;  // 0/1：batch_variance 输出是否存在
};

#endif // IN_INFER_V2_TILING_DATA_H
