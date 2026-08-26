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
 * \file bn_training_reduce_grad_tiling_data.h
 * \brief tiling data struct shared by host tiling and arch35 kernel（ND 单路径字段）
 *
 * numRecip    = 1 / (N * R)    （host fp64 计算后舍入 fp32）
 * negNumRecip = -numRecip      （fp32 取负，IEEE 取负精确，对齐 A2 tvm.const(-num_bw)）
 * sqrtVar[c]    = sqrt(batch_variance[c] + epsilon)
 * multiplier[c] = (diff_scale[c] * negNumRecip) / sqrtVar[c]
 * addend[c]     = (batch_mean[c] / sqrtVar[c]) * (diff_scale[c] * numRecip) + (diff_offset[c] * negNumRecip)
 * mulScale[c]   = scale[c] / sqrtVar[c]
 * y[n,c,r]      = ((grads[n,c,r] + multiplier[c] * x[n,c,r]) + addend[c]) * mulScale[c]
 *
 * 单条计算路径（tilingKey=0）：ND/NCHW [N,C,R...] plane 连续，plane p 属于 channel p%C，
 * per-channel 系数广播。
 *
 * 统一切分模型：units 个独立 plane 按 unitCores 均分（former/latter），plane 不足时
 * plane 内 inner 维再切 innerCores 份；blockIdx = unitIdx * innerCores + innerIdx。无 workspace。
 */

#ifndef BN_TRAINING_REDUCE_GRAD_TILING_DATA_H
#define BN_TRAINING_REDUCE_GRAD_TILING_DATA_H

#include <cstdint>

struct BNTrainingReduceGradTilingData {
    int64_t numN;      // N（kernel 不直接消费：units/numC 已含；保留供 host 日志与调试核对）
    int64_t numC;      // C（dim1）
    int64_t innerSize; // R = prod(d2:)
    int64_t units;     // plane 数 = N*C
    int64_t unitCores; // plane 维切分核数（kernel 由 blockIdx/formerCoreNum 反推，保留供 host 日志）
    int64_t formerCoreNum; // 前 formerCoreNum 核每核 formerUnits 个 plane，其余 latterUnits 个
    int64_t formerUnits;
    int64_t latterUnits;
    int64_t innerCores;   // plane 内 inner 维切分份数（plane < 核数且 inner 够大时 >1）
    int64_t innerPerCore; // 每份的 inner 长度，末尾截断
    int64_t ubTileSize;   // UB tile 元素数（VL=64 对齐）
    float epsilon;        // 加到 batch_variance 上再开方
    float numRecip;       // 1/(N*R)，fp64 计算后舍入 fp32（对齐 A2 TBE tvm.const 语义）
    float negNumRecip;    // -numRecip（fp32 取负精确，对齐 A2 TBE tvm.const(-num_bw) 语义）
    float reserved;       // 结构体 8B 对齐填充
};

#endif // BN_TRAINING_REDUCE_GRAD_TILING_DATA_H
