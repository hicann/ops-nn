/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_nd.cpp
 * \brief
 */

#define MIX_KERNEL_X_B8_INDICES_32_INDEX_32 140UL
#define MIX_KERNEL_X_B8_INDICES_64_INDEX_32 180UL
#define MIX_KERNEL_X_B8_INDICES_64_INDEX_64 181UL

#define MIX_KERNEL_X_B16_INDICES_32_INDEX_32 240UL
#define MIX_KERNEL_X_B16_INDICES_64_INDEX_32 280UL
#define MIX_KERNEL_X_B16_INDICES_64_INDEX_64 281UL

#define MIX_KERNEL_X_B32_INDICES_32_INDEX_32 440UL
#define MIX_KERNEL_X_B32_INDICES_64_INDEX_32 480UL
#define MIX_KERNEL_X_B32_INDICES_64_INDEX_64 481UL

#define MIX_KERNEL_X_B64_INDICES_32_INDEX_32 840UL
#define MIX_KERNEL_X_B64_INDICES_64_INDEX_32 880UL
#define MIX_KERNEL_X_B64_INDICES_64_INDEX_64 881UL

#define SIMT_NOSUPPORT_NEGATIVE_X_B64_INDICES_64_INDEX_64 10000000001231101033UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B64_INDICES_64_INDEX_32 10000000001231100033UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B64_INDICES_32_INDEX_32 10000000001231100032UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B64_INDICES_32_INDEX_64 10000000001231101032UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B32_INDICES_64_INDEX_64 10000000001231101023UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B32_INDICES_64_INDEX_32 10000000001231100023UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B32_INDICES_32_INDEX_32 10000000001231100022UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B32_INDICES_32_INDEX_64 10000000001231101022UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B16_INDICES_64_INDEX_64 10000000001231101013UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B16_INDICES_64_INDEX_32 10000000001231100013UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B16_INDICES_32_INDEX_32 10000000001231100012UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B16_INDICES_32_INDEX_64 10000000001231101012UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B8_INDICES_64_INDEX_64 10000000001231101003UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B8_INDICES_64_INDEX_32 10000000001231100003UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B8_INDICES_32_INDEX_32 10000000001231100002UL
#define SIMT_NOSUPPORT_NEGATIVE_X_B8_INDICES_32_INDEX_64 10000000001231101002UL
#define SIMT_SUPPORT_NEGATIVE_X_B64_INDICES_64_INDEX_64 10000000001231101133UL
#define SIMT_SUPPORT_NEGATIVE_X_B64_INDICES_64_INDEX_32 10000000001231100133UL
#define SIMT_SUPPORT_NEGATIVE_X_B64_INDICES_32_INDEX_32 10000000001231100132UL
#define SIMT_SUPPORT_NEGATIVE_X_B64_INDICES_32_INDEX_64 10000000001231101132UL
#define SIMT_SUPPORT_NEGATIVE_X_B32_INDICES_64_INDEX_64 10000000001231101123UL
#define SIMT_SUPPORT_NEGATIVE_X_B32_INDICES_64_INDEX_32 10000000001231100123UL
#define SIMT_SUPPORT_NEGATIVE_X_B32_INDICES_32_INDEX_32 10000000001231100122UL
#define SIMT_SUPPORT_NEGATIVE_X_B32_INDICES_32_INDEX_64 10000000001231101122UL
#define SIMT_SUPPORT_NEGATIVE_X_B16_INDICES_64_INDEX_64 10000000001231101113UL
#define SIMT_SUPPORT_NEGATIVE_X_B16_INDICES_64_INDEX_32 10000000001231100113UL
#define SIMT_SUPPORT_NEGATIVE_X_B16_INDICES_32_INDEX_32 10000000001231100112UL
#define SIMT_SUPPORT_NEGATIVE_X_B16_INDICES_32_INDEX_64 10000000001231101112UL
#define SIMT_SUPPORT_NEGATIVE_X_B8_INDICES_64_INDEX_64 10000000001231101103UL
#define SIMT_SUPPORT_NEGATIVE_X_B8_INDICES_64_INDEX_32 10000000001231100103UL
#define SIMT_SUPPORT_NEGATIVE_X_B8_INDICES_32_INDEX_32 10000000001231100102UL
#define SIMT_SUPPORT_NEGATIVE_X_B8_INDICES_32_INDEX_64 10000000001231101102UL
#define ZERO_SHAPE_X_B64_INDICES_64 10000000000231110030UL
#define ZERO_SHAPE_X_B32_INDICES_64 10000000000231110020UL
#define ZERO_SHAPE_X_B16_INDICES_64 10000000000231110010UL
#define ZERO_SHAPE_X_B8_INDICES_64 10000000000231110000UL
#define SIMD_TILING_KEY 11111111
#define SIMD_GA_FULL_LOAD_VGATHER_TILING_KEY  4000UL
#define SIMD_GA_FULL_LOAD_VGATHER_SUPPORT_NEG_INDICE_TILING_KEY  4100UL
#define SIMD_GA_ALL_LOAD_TILING_KEY  3000UL
#define SIMD_GA_ALL_LOAD_SUPPORT_NEG_INDICE_TILING_KEY  3100UL

#include "./arch35/gather_nd_simd.h"
#include "./arch35/gather_nd_simt.h"
#include "./arch35/gather_nd_zero.h"
#include "./arch35/gather_nd_full_load_vgather.h"
#include "./arch35/gather_nd_all_load.h"
#include "./arch35/gather_nd_mix.h"

using namespace GatherNd;

extern "C" __global__ __aicore__ void gather_nd(GM_ADDR x, GM_ADDR indices, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  SetSysWorkspace(workspace);
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  TPipe pipeIn;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (TILING_KEY_IS(SIMD_GA_FULL_LOAD_VGATHER_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdGaAllLoadTilingData, tiling_data, tiling);
    GatherNdAllLoadV<DTYPE_INDICES, false> op;
    op.Init(x, indices, y, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMD_GA_FULL_LOAD_VGATHER_SUPPORT_NEG_INDICE_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdGaAllLoadTilingData, tiling_data, tiling);
    GatherNdAllLoadV<DTYPE_INDICES, true> op;
    op.Init(x, indices, y, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMD_GA_ALL_LOAD_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdGaAllLoadTilingData, tiling_data, tiling);
    GatherNdAllLoad<DTYPE_INDICES, false> op(&pipeIn);
    op.Init(x, indices, y, &tiling_data);
    op.Process();
  } else if (TILING_KEY_IS(SIMD_GA_ALL_LOAD_SUPPORT_NEG_INDICE_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdGaAllLoadTilingData, tiling_data, tiling);
    GatherNdAllLoad<DTYPE_INDICES, true> op(&pipeIn);
    op.Init(x, indices, y, &tiling_data);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B64_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int64_t, int64_t, uint64_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B64_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int64_t, int64_t, uint32_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B64_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int64_t, int32_t, uint32_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B64_INDICES_32_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int64_t, int32_t, uint64_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B32_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int32_t, int64_t, uint64_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B32_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int32_t, int64_t, uint32_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B32_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int32_t, int32_t, uint32_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B32_INDICES_32_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int32_t, int32_t, uint64_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B16_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int16_t, int64_t, uint64_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B16_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int16_t, int64_t, uint32_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B16_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int16_t, int32_t, uint32_t,false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B16_INDICES_32_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int16_t, int32_t, uint64_t,false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B8_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int8_t, int64_t, uint64_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B8_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int8_t, int64_t, uint32_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B8_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int8_t, int32_t, uint32_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_NOSUPPORT_NEGATIVE_X_B8_INDICES_32_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int8_t, int32_t, uint64_t, false> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B64_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int64_t, int64_t, uint64_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B64_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int64_t, int64_t, uint32_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B64_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int64_t, int32_t, uint32_t,true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B32_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int32_t, int64_t, uint64_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B32_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int32_t, int64_t, uint32_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B32_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int32_t, int32_t, uint32_t,true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B32_INDICES_32_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int32_t, int32_t, uint64_t,true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B16_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int16_t, int64_t, uint64_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B16_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int16_t, int64_t, uint32_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B16_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int16_t, int32_t, uint32_t,true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B16_INDICES_32_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int16_t, int32_t, uint64_t,true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B8_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int8_t, int64_t, uint64_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B8_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int8_t, int64_t, uint32_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B8_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int8_t, int32_t, uint32_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMT_SUPPORT_NEGATIVE_X_B8_INDICES_32_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimt<int8_t, int32_t, uint64_t, true> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(ZERO_SHAPE_X_B64_INDICES_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdZero<int64_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(ZERO_SHAPE_X_B32_INDICES_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdZero<int32_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(ZERO_SHAPE_X_B16_INDICES_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdZero<int16_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(ZERO_SHAPE_X_B8_INDICES_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdZero<int8_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(SIMD_TILING_KEY)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdSimd<DTYPE_INDICES> op;
    op.Init(x, indices, y, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B8_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int8_t, int32_t, int32_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B8_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int8_t, int64_t, int32_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B8_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int8_t, int64_t, int64_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B16_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int16_t, int32_t, int32_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B16_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int16_t, int64_t, int32_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B16_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int16_t, int64_t, int64_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B32_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int32_t, int32_t, int32_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B32_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int32_t, int64_t, int32_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B32_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int32_t, int64_t, int64_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B64_INDICES_32_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int64_t, int32_t, int32_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B64_INDICES_64_INDEX_32)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int64_t, int64_t, int32_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(MIX_KERNEL_X_B64_INDICES_64_INDEX_64)) {
    GET_TILING_DATA_WITH_STRUCT(GatherNdTilingData, tiling_data, tiling);
    GatherNdMixKernel<int64_t, int64_t, int64_t> op;
    op.Init(x, indices, y, userWS, &tiling_data, &pipeIn);
    op.Process();
  }
}