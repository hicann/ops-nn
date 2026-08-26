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
 * \file bn_training_update_grad_tiling_data.h
 * \brief tiling data struct shared by host tiling and arch35 kernel（ND 单路径字段）
 *
 * rstd[c]        = 1 / sqrt(batch_variance[c] + epsilon)
 * diff_scale[c]  = sum_{n,r} grads[n,c,r] * (x[n,c,r] - batch_mean[c]) * rstd[c]
 * diff_offset[c] = sum_{n,r} grads[n,c,r]
 *
 * 单条计算路径（tilingKey=0）：ND/NCHW [N,C,R...] plane 连续，channel 主切分
 * （每 channel 的完整归约由唯一归属核完成，零核间通信、无 workspace）。
 *
 * 切分模型：C 按 channelCores 均分（前 cFormerCoreNum 核每核 cFormerLen 个 channel，
 * 其余 cLatterLen 个）；blockIdx = cRangeIdx。核内 channel 范围再按 cLenCap 分 chunk、
 * R 按 sliceR 分片、N 按 rowsPerTile 分 tile。
 */

#ifndef BN_TRAINING_UPDATE_GRAD_TILING_DATA_H
#define BN_TRAINING_UPDATE_GRAD_TILING_DATA_H

#include <cstdint>

struct BNTrainingUpdateGradTilingData {
    int64_t numN;      // N（dim0；kernel 不直接消费，保留供 host 日志与调试核对）
    int64_t numC;      // C（dim1）
    int64_t innerSize; // R = prod(d2:)
    int64_t channelCores; // channel 维切分核数 = min(C, coreNum)（kernel 不直接消费：blockDim 已定，保留供核对）
    int64_t cFormerCoreNum; // 前 cFormerCoreNum 核每核 cFormerLen 个 channel，其余 cLatterLen 个
    int64_t cFormerLen;
    int64_t cLatterLen;
    int64_t cLenCap;     // channel chunk 长度上限（UB 反推）
    int64_t sliceR;      // R 维分片长度（R 极大时 < R，否则 == R）
    int64_t rowsPerTile; // 每 tile 的 n 行数上限（UB 反推）
    float epsilon;       // 加到 batch_variance 上再开方取倒数
    float reserved;      // 结构体 8B 对齐填充
};

#endif // BN_TRAINING_UPDATE_GRAD_TILING_DATA_H
