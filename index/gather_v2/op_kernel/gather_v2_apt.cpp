/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/gather_v2.h"
#include "arch35/gather_v2_simd.h"
#include "arch35/gather_v2_simd_two_dim.h"
#include "arch35/gather_v2_simd_last_gather.h"
#include "arch35/gather_v2_ga_all_load.h"
#include "arch35/gather_v2_after_gather_full_load.h"
#include "arch35/gather_v2_simt_two_dim.h"
#include "arch35/gather_v2_empty.h"

using namespace AscendC;
using namespace gatherv2;

#define TILING_KEY_X_B64_INDEX_SIZE_64  10000000000000000031UL
#define TILING_KEY_X_B32_INDEX_SIZE_64  10000000000000000021UL
#define TILING_KEY_X_B16_INDEX_SIZE_64  10000000000000000011UL
#define TILING_KEY_X_B8_INDEX_SIZE_64   10000000000000000001UL
#define TILING_KEY_X_B64_INDEX_SIZE_32  10000000000000000030UL
#define TILING_KEY_X_B32_INDEX_SIZE_32  10000000000000000020UL
#define TILING_KEY_X_B16_INDEX_SIZE_32  10000000000000000010UL
#define TILING_KEY_X_B8_INDEX_SIZE_32   10000000000000000000UL
#define SIMD_TILING_KEY                 1000000099
#define SIMD_TILING_KEY_TWO_DIM         1000000299
#define SIMD_LAST_GATHER_B8_TILING_KEY  1100000001UL
#define SIMD_LAST_GATHER_B16_TILING_KEY  1100000002UL
#define SIMD_LAST_GATHER_B32_TILING_KEY  1100000004UL
#define SIMD_LAST_GATHER_B64_TILING_KEY  1100000008UL
#define SIMD_LAST_GATHER_B8_SUPPORT_NEG_INDICE_TILING_KEY  1100000101UL
#define SIMD_LAST_GATHER_B16_SUPPORT_NEG_INDICE_TILING_KEY  1100000102UL
#define SIMD_LAST_GATHER_B32_SUPPORT_NEG_INDICE_TILING_KEY  1100000104UL
#define SIMD_LAST_GATHER_B64_SUPPORT_NEG_INDICE_TILING_KEY  1100000108UL
#define SIMD_GA_ALL_LOAD_TILING_KEY  3000UL
#define SIMD_GA_ALL_LOAD_SUPPORT_NEG_INDICE_TILING_KEY  3100UL

#define TILING_KEY_X_B64_INDEX_SIZE_2D  10000000000000000032UL
#define TILING_KEY_X_B32_INDEX_SIZE_2D  10000000000000000022UL
#define TILING_KEY_X_B16_INDEX_SIZE_2D  10000000000000000012UL
#define TILING_KEY_X_B8_INDEX_SIZE_2D   10000000000000000002UL

#define TILING_KEY_EMPTY  3000000000UL

#define SIMT_TWO_DIM_B8_INDEX_SIZE_32_TILING_KEY  2000000001UL
#define SIMT_TWO_DIM_B16_INDEX_SIZE_32_TILING_KEY  2000000002UL
#define SIMT_TWO_DIM_B32_INDEX_SIZE_32_TILING_KEY  2000000004UL
#define SIMT_TWO_DIM_B64_INDEX_SIZE_32_TILING_KEY  2000000008UL
#define SIMT_TWO_DIM_B8_INDEX_SIZE_64_TILING_KEY  2000000101UL
#define SIMT_TWO_DIM_B16_INDEX_SIZE_64_TILING_KEY  2000000102UL
#define SIMT_TWO_DIM_B32_INDEX_SIZE_64_TILING_KEY  2000000104UL
#define SIMT_TWO_DIM_B64_INDEX_SIZE_64_TILING_KEY  2000000108UL

extern "C" __global__ __aicore__ void gather_v2(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling) {
  TPipe pipe;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); 
  if (TILING_KEY_IS(TILING_KEY_X_B8_INDEX_SIZE_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2<int8_t, DTYPE_INDICES, uint64_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B16_INDEX_SIZE_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2<int16_t, DTYPE_INDICES, uint64_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B32_INDEX_SIZE_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2<int32_t, DTYPE_INDICES, uint64_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B64_INDEX_SIZE_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2<int64_t, DTYPE_INDICES, uint64_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B8_INDEX_SIZE_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2<int8_t, DTYPE_INDICES, uint32_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B16_INDEX_SIZE_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2<int16_t, DTYPE_INDICES, uint32_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B32_INDEX_SIZE_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2<int32_t, DTYPE_INDICES, uint32_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B64_INDEX_SIZE_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2<int64_t, DTYPE_INDICES, uint32_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2Simd<DTYPE_INDICES> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_TILING_KEY_TWO_DIM)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataSimdTwoDim, tilingDataIn, tiling);
    Gatherv2SimdTwoDim<DTYPE_INDICES> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_LAST_GATHER_B8_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2LastTilingData, tilingDataIn, tiling);
    Gatherv2SimdLastGather<int8_t, DTYPE_INDICES, false> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_LAST_GATHER_B16_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2LastTilingData, tilingDataIn, tiling);
    Gatherv2SimdLastGather<int16_t, DTYPE_INDICES, false> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_LAST_GATHER_B32_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2LastTilingData, tilingDataIn, tiling);
    Gatherv2SimdLastGather<int32_t, DTYPE_INDICES, false> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_LAST_GATHER_B64_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2LastTilingData, tilingDataIn, tiling);
    Gatherv2SimdLastGather<int64_t, DTYPE_INDICES, false> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_LAST_GATHER_B8_SUPPORT_NEG_INDICE_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2LastTilingData, tilingDataIn, tiling);
    Gatherv2SimdLastGather<int8_t, DTYPE_INDICES, true> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_LAST_GATHER_B16_SUPPORT_NEG_INDICE_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2LastTilingData, tilingDataIn, tiling);
    Gatherv2SimdLastGather<int16_t, DTYPE_INDICES, true> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_LAST_GATHER_B32_SUPPORT_NEG_INDICE_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2LastTilingData, tilingDataIn, tiling);
    Gatherv2SimdLastGather<int32_t, DTYPE_INDICES, true> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_LAST_GATHER_B64_SUPPORT_NEG_INDICE_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2LastTilingData, tilingDataIn, tiling);
    Gatherv2SimdLastGather<int64_t, DTYPE_INDICES, true> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_GA_ALL_LOAD_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2GaAllLoadTilingData, tilingDataIn, tiling);
    Gatherv2GaAllLoad<DTYPE_INDICES, false> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMD_GA_ALL_LOAD_SUPPORT_NEG_INDICE_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2GaAllLoadTilingData, tilingDataIn, tiling);
    Gatherv2GaAllLoad<DTYPE_INDICES, true> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B8_INDEX_SIZE_2D)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2FullLoad<int8_t, DTYPE_INDICES, uint32_t> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B16_INDEX_SIZE_2D)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2FullLoad<int16_t, DTYPE_INDICES, uint32_t> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B32_INDEX_SIZE_2D)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2FullLoad<int32_t, DTYPE_INDICES, uint32_t> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_X_B64_INDEX_SIZE_2D)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingData, tilingDataIn, tiling);
    Gatherv2FullLoad<int64_t, DTYPE_INDICES, uint32_t> gatherv2Op(&pipe);
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B8_INDEX_SIZE_64_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataSimtTwoDim, tilingDataIn, tiling);
    Gatherv2SimtTwoDim<int8_t, DTYPE_INDICES, uint64_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B16_INDEX_SIZE_64_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataSimtTwoDim, tilingDataIn, tiling);
    Gatherv2SimtTwoDim<int16_t, DTYPE_INDICES, uint64_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B32_INDEX_SIZE_64_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataSimtTwoDim, tilingDataIn, tiling);
    Gatherv2SimtTwoDim<int32_t, DTYPE_INDICES, uint64_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B64_INDEX_SIZE_64_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataSimtTwoDim, tilingDataIn, tiling);
    Gatherv2SimtTwoDim<int64_t, DTYPE_INDICES, uint64_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B8_INDEX_SIZE_32_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataSimtTwoDim, tilingDataIn, tiling);
    Gatherv2SimtTwoDim<int8_t, DTYPE_INDICES, uint32_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B16_INDEX_SIZE_32_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataSimtTwoDim, tilingDataIn, tiling);
    Gatherv2SimtTwoDim<int16_t, DTYPE_INDICES, uint32_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B32_INDEX_SIZE_32_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataSimtTwoDim, tilingDataIn, tiling);
    Gatherv2SimtTwoDim<int32_t, DTYPE_INDICES, uint32_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(SIMT_TWO_DIM_B64_INDEX_SIZE_32_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataSimtTwoDim, tilingDataIn, tiling);
    Gatherv2SimtTwoDim<int64_t, DTYPE_INDICES, uint32_t> gatherv2Op;
    gatherv2Op.Init(x, indices, axis, y, &tilingDataIn);
    gatherv2Op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_EMPTY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherV2TilingDataEmptyInput, tilingDataIn, tiling);
    Gatherv2Empty<int8_t> gatherv2Op;
    gatherv2Op.Init(y, &tilingDataIn);
    gatherv2Op.Process();
  }
}