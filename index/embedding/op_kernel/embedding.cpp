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
 * \file embedding.cpp
 * \brief
 */
#include "arch35/embedding_simt_two_dim.h"
#include "arch35/embedding_simt_no_contiguous.h"

using namespace AscendC;
using namespace Embedding;

#define SIMT_TWO_DIM_B8_INDEX_SIZE_32_TILING_KEY  2000000001UL
#define SIMT_TWO_DIM_B16_INDEX_SIZE_32_TILING_KEY  2000000002UL
#define SIMT_TWO_DIM_B32_INDEX_SIZE_32_TILING_KEY  2000000004UL
#define SIMT_TWO_DIM_B64_INDEX_SIZE_32_TILING_KEY  2000000008UL
#define EMBEDDING_NO_CONTIGUOUS_B8_INDEX_SIZE_INT32  101
#define EMBEDDING_NO_CONTIGUOUS_B16_INDEX_SIZE_INT32  102
#define EMBEDDING_NO_CONTIGUOUS_B32_INDEX_SIZE_INT32  104
#define EMBEDDING_NO_CONTIGUOUS_B64_INDEX_SIZE_INT32  108
#define EMBEDDING_NO_CONTIGUOUS_B8_INDEX_SIZE_INT64  111
#define EMBEDDING_NO_CONTIGUOUS_B16_INDEX_SIZE_INT64  112
#define EMBEDDING_NO_CONTIGUOUS_B32_INDEX_SIZE_INT64  114
#define EMBEDDING_NO_CONTIGUOUS_B64_INDEX_SIZE_INT64  118

extern "C" __global__ __aicore__ void embedding(GM_ADDR x, GM_ADDR indices, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  if (TILING_KEY_IS(EMBEDDING_NO_CONTIGUOUS_B8_INDEX_SIZE_INT32)) {
    GET_TILING_DATA_WITH_STRUCT(EmbeddingNoContiguousTilingData, tilingData, tiling);
    EmbeddingKernelNoContiguous<int8_t, DTYPE_INDICES, uint32_t> embeddingOp(&tilingData, &pipe);
    embeddingOp.Init(x, indices, y);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(EMBEDDING_NO_CONTIGUOUS_B16_INDEX_SIZE_INT32)) {
    GET_TILING_DATA_WITH_STRUCT(EmbeddingNoContiguousTilingData, tilingData, tiling);
    EmbeddingKernelNoContiguous<int16_t, DTYPE_INDICES, uint32_t> embeddingOp(&tilingData, &pipe);
    embeddingOp.Init(x, indices, y);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(EMBEDDING_NO_CONTIGUOUS_B32_INDEX_SIZE_INT32)) {
    GET_TILING_DATA_WITH_STRUCT(EmbeddingNoContiguousTilingData, tilingData, tiling);
    EmbeddingKernelNoContiguous<int32_t, DTYPE_INDICES, uint32_t> embeddingOp(&tilingData, &pipe);
    embeddingOp.Init(x, indices, y);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(EMBEDDING_NO_CONTIGUOUS_B64_INDEX_SIZE_INT32)) {
    GET_TILING_DATA_WITH_STRUCT(EmbeddingNoContiguousTilingData, tilingData, tiling);
    EmbeddingKernelNoContiguous<int64_t, DTYPE_INDICES, uint32_t> embeddingOp(&tilingData, &pipe);
    embeddingOp.Init(x, indices, y);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(EMBEDDING_NO_CONTIGUOUS_B8_INDEX_SIZE_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(EmbeddingNoContiguousTilingData, tilingData, tiling);
    EmbeddingKernelNoContiguous<int8_t, DTYPE_INDICES, uint64_t> embeddingOp(&tilingData, &pipe);
    embeddingOp.Init(x, indices, y);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(EMBEDDING_NO_CONTIGUOUS_B16_INDEX_SIZE_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(EmbeddingNoContiguousTilingData, tilingData, tiling);
    EmbeddingKernelNoContiguous<int16_t, DTYPE_INDICES, uint64_t> embeddingOp(&tilingData, &pipe);
    embeddingOp.Init(x, indices, y);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(EMBEDDING_NO_CONTIGUOUS_B32_INDEX_SIZE_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(EmbeddingNoContiguousTilingData, tilingData, tiling);
    EmbeddingKernelNoContiguous<int32_t, DTYPE_INDICES, uint64_t> embeddingOp(&tilingData, &pipe);
    embeddingOp.Init(x, indices, y);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(EMBEDDING_NO_CONTIGUOUS_B64_INDEX_SIZE_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(EmbeddingNoContiguousTilingData, tilingData, tiling);
    EmbeddingKernelNoContiguous<int64_t, DTYPE_INDICES, uint64_t> embeddingOp(&tilingData, &pipe);
    embeddingOp.Init(x, indices, y);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B8_INDEX_SIZE_32_TILING_KEY)) {
    GET_TILING_DATA_PTR_WITH_STRUCT(EmbeddingTilingDataSimtTwoDim, tilingData, tiling);
    EmbeddingSimtTwoDim<int8_t, DTYPE_INDICES, uint32_t> embeddingOp;
    embeddingOp.Init(x, indices, y, tilingData);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B16_INDEX_SIZE_32_TILING_KEY)) {
    GET_TILING_DATA_PTR_WITH_STRUCT(EmbeddingTilingDataSimtTwoDim, tilingData, tiling);
    EmbeddingSimtTwoDim<int16_t, DTYPE_INDICES, uint32_t> embeddingOp;
    embeddingOp.Init(x, indices, y, tilingData);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B32_INDEX_SIZE_32_TILING_KEY)) {
    GET_TILING_DATA_PTR_WITH_STRUCT(EmbeddingTilingDataSimtTwoDim, tilingData, tiling);
    EmbeddingSimtTwoDim<int32_t, DTYPE_INDICES, uint32_t> embeddingOp;
    embeddingOp.Init(x, indices, y, tilingData);
    embeddingOp.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B64_INDEX_SIZE_32_TILING_KEY)) {
    GET_TILING_DATA_PTR_WITH_STRUCT(EmbeddingTilingDataSimtTwoDim, tilingData, tiling);
    EmbeddingSimtTwoDim<int64_t, DTYPE_INDICES, uint32_t> embeddingOp;
    embeddingOp.Init(x, indices, y, tilingData);
    embeddingOp.Process();
  }
}