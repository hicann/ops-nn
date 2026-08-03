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
 * \file cosine_embedding_loss_tiling_data.h
 * \brief Plain tiling struct shared between arch35 host tiling and kernel.
 *
 * Logical layout: x1/x2 are broadcast to a common ND shape, dimension 1 is the
 * feature/reduction dimension, and target is broadcast with that reduced shape.
 * Output y is fp32: broadcast(reduced_x_shape, target_shape) for reduction "none",
 * and [1] for "sum"/"mean".
 */
#ifndef OPS_LOSS_COSINE_EMBEDDING_LOSS_TILING_DATA_H_
#define OPS_LOSS_COSINE_EMBEDDING_LOSS_TILING_DATA_H_

#include <cstdint>

constexpr uint32_t COSINE_EMBEDDING_LOSS_MAX_RANK = 8;
constexpr uint32_t COSINE_EMBEDDING_LOSS_GENERIC_PATH = 0;
constexpr uint32_t COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH = 1;
constexpr uint32_t COSINE_EMBEDDING_LOSS_REDUCTION_NONE = 0;
constexpr uint32_t COSINE_EMBEDDING_LOSS_REDUCTION_SUM = 1;
constexpr uint32_t COSINE_EMBEDDING_LOSS_REDUCTION_MEAN = 2;
constexpr int64_t COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE = 8; // 8 fp32 = 32B per core
constexpr int64_t COSINE_EMBEDDING_LOSS_MAX_CORE_NUM = 64;
constexpr int64_t COSINE_EMBEDDING_LOSS_PARTIAL_BUF_ELEMS = COSINE_EMBEDDING_LOSS_MAX_CORE_NUM *
                                                            COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE;

struct CosineEmbeddingLossTilingData {
    int64_t n = 0;      // number of output loss elements before reduction
    int64_t d = 0;      // feature length: broadcast(x1, x2).shape[1]
    int64_t dAlign = 0; // kept for diagnostics and host/kernel contract checks
    int64_t rowsPerCore = 0;
    int64_t tailRows = 0;
    int64_t usedCoreNum = 0;
    int64_t ubTileRows = 1;
    int64_t featureTile = 0;    // contiguous feature elements staged in UB per iteration
    int64_t reduceTmpBytes = 0; // UB scratch used by the fast-path feature reduction
    uint32_t reduction = COSINE_EMBEDDING_LOSS_REDUCTION_MEAN;
    uint32_t fastPath = COSINE_EMBEDDING_LOSS_GENERIC_PATH;
    uint32_t outputRank = 0;
    uint32_t xBroadcastRank = 0;
    float margin = 0.0f;
    float meanCoef = 1.0f; // 1/N for reduction "mean", else 1
    float eps = 1e-12f;    // epsilon added inside each sqrt(sum(x^2) + eps)

    int64_t outputShape[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
    int64_t x1OutStrides[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
    int64_t x2OutStrides[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
    int64_t targetOutStrides[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
    int64_t x1ReduceStride = 0;
    int64_t x2ReduceStride = 0;
};

#endif // OPS_LOSS_COSINE_EMBEDDING_LOSS_TILING_DATA_H_
