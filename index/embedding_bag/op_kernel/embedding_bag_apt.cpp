/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file embedding_bag_apt.cpp
 * \brief
 */

#if __CCE_AICORE__ == 310
#include "./arch35/embedding_bag_regbase_1d_sum.h"
#include "./arch35/embedding_bag_regbase_1d_mean.h"
#include "./arch35/embedding_bag_regbase_1d_max.h"
#include "./arch35/embedding_bag_regbase_2d.h"
#include "./arch35/embedding_bag_regbase_1d_simt.h"
#include "./arch35/embedding_bag_regbase_2d_simt.h"
#define TILING_KEY_INDICES_1D_SAME         100
#define TILING_KEY_INDICES_1D_PROMOTE      101
#define TILING_KEY_INDICES_2D_SAME         200
#define TILING_KEY_INDICES_2D_PROMOTE      201
#define TILING_KEY_SIMT_1D_SAME_INT32         300  
#define TILING_KEY_SIMT_1D_SAME_INT64         301  
#define TILING_KEY_SIMT_1D_PROMOTE_INT32         302  
#define TILING_KEY_SIMT_1D_PROMOTE_INT64         303  
#define TILING_KEY_SIMT_2D_SAME_INT32         400  
#define TILING_KEY_SIMT_2D_SAME_INT64         401  
#define TILING_KEY_SIMT_2D_PROMOTE_INT32         402  
#define TILING_KEY_SIMT_2D_PROMOTE_INT64         403 

#define MODE_SUM         0
#define MODE_MEAN        1
#define MODE_MAX         2
#else
#include "embedding_bag.h"
#include "embedding_bag_fp16.h"
#endif

using namespace AscendC;
// kernel function
extern "C" __global__ __aicore__ void embedding_bag(
    GM_ADDR weight, GM_ADDR indices, GM_ADDR offsets, GM_ADDR per_sample_weights, GM_ADDR y, GM_ADDR offset2bag,
    GM_ADDR bag_size, GM_ADDR max_indices, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

#if __CCE_AICORE__ == 310
    GM_ADDR gmParam[TENSOR_COUNT] = {weight, indices, offsets, per_sample_weights, y, offset2bag, bag_size, max_indices};
    if (TILING_KEY_IS(TILING_KEY_INDICES_1D_SAME)) {
        if (tilingData.mode == MODE_SUM) {
            EmbeddingBag::EmbeddingBagRegBaseIndx1dSum<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(gmParam);
            op.Process();
        } else if (tilingData.mode == MODE_MEAN) {
            EmbeddingBag::EmbeddingBagRegBaseIndx1dMean<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(gmParam);
            op.Process();
        } else if (tilingData.mode == MODE_MAX) {
            EmbeddingBag::EmbeddingBagRegBaseIndx1dMax<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(gmParam);
            op.Process();
        } else {
            return;
        }
    } else if (TILING_KEY_IS(TILING_KEY_INDICES_1D_PROMOTE)) {
        if (tilingData.mode == MODE_SUM) {
            EmbeddingBag::EmbeddingBagRegBaseIndx1dSum<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, int64_t> op(tilingData, pipe);
            op.Init(gmParam);
            op.Process();
        } else if (tilingData.mode == MODE_MEAN) {
            EmbeddingBag::EmbeddingBagRegBaseIndx1dMean<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, int64_t> op(tilingData, pipe);
            op.Init(gmParam);
            op.Process();
        } else if (tilingData.mode == MODE_MAX) {
            EmbeddingBag::EmbeddingBagRegBaseIndx1dMax<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, int64_t> op(tilingData, pipe);
            op.Init(gmParam);
            op.Process();
        } else {
            return;
        }
    } else if (TILING_KEY_IS(TILING_KEY_INDICES_2D_SAME)) {
        EmbeddingBag::EmbeddingBagRegBaseIndx2d<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_INDICES> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_INDICES_2D_PROMOTE)) {
        EmbeddingBag::EmbeddingBagRegBaseIndx2d<DTYPE_WEIGHT, DTYPE_INDICES, int64_t> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_1D_SAME_INT32)) {
        EmbeddingBag::EmbeddingBagRegBaseSimt1D<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_1D_SAME_INT64)) {
        EmbeddingBag::EmbeddingBagRegBaseSimt1D<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_1D_PROMOTE_INT32)) {
        EmbeddingBag::EmbeddingBagRegBaseSimt1D<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, int64_t, uint32_t> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_1D_PROMOTE_INT64)) {
        EmbeddingBag::EmbeddingBagRegBaseSimt1D<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, int64_t, uint64_t> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    }else if (TILING_KEY_IS(TILING_KEY_SIMT_2D_SAME_INT32)) {
        EmbeddingBag::EmbeddingBagRegBaseSimt2D<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, DTYPE_INDICES, uint32_t> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_2D_SAME_INT64)) {
        EmbeddingBag::EmbeddingBagRegBaseSimt2D<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, DTYPE_INDICES, uint64_t> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_2D_PROMOTE_INT32)) {
        EmbeddingBag::EmbeddingBagRegBaseSimt2D<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, int64_t, uint32_t> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_2D_PROMOTE_INT64)) {
        EmbeddingBag::EmbeddingBagRegBaseSimt2D<DTYPE_WEIGHT, DTYPE_INDICES, DTYPE_OFFSETS, int64_t, uint64_t> op(tilingData, pipe);
        op.Init(gmParam);
        op.Process();
    }
#else
    GM_ADDR gmTensor[8] = {weight, indices, offsets, per_sample_weights, y, offset2bag, bag_size, max_indices};
    if (TILING_KEY_IS(1)) {
        EmbeddingBag<float, int> op(gmTensor, tilingData, pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2)) {
        EmbeddingBagFP16<half, int> op(gmTensor, tilingData, pipe);
        op.Process();
    }
#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    if (TILING_KEY_IS(3)) {
        EmbeddingBagFP16<bfloat16_t, int> op(gmTensor, tilingData, pipe);
        op.Process();
    }
#endif
#endif
}
