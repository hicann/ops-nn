/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
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
 * \file index.cpp
 * \brief index full load
 */

#include "arch35/index.h"
#include "arch35/index_perf.h"
#include "arch35/index_simd.h"
#include "arch35/index_full_load.h"
#include "arch35/index_no_continuous.h"

using namespace Index;
#define INDEX_PERF_B8 1001
#define INDEX_PERF_B16 1002
#define INDEX_PERF_B32 1004
#define INDEX_PERF_B64 1008
#define INDEX_SIMD 3001
#define INDEX_PERF_B8_INDEX_INT64 11001
#define INDEX_PERF_B16_INDEX_INT64 11002
#define INDEX_PERF_B32_INDEX_INT64 11004
#define INDEX_PERF_B64_INDEX_INT64 11008
#define INDEX_B8_INDEX_INT64 10001
#define INDEX_B16_INDEX_INT64 10002
#define INDEX_B32_INDEX_INT64 10004
#define INDEX_B64_INDEX_INT64 10008
#define INDEX_B128_INDEX_INT64 10016

#define NOCON_INDEX_B8_INDEX_INT32 20001
#define NOCON_INDEX_B16_INDEX_INT32 20002
#define NOCON_INDEX_B32_INDEX_INT32 20004
#define NOCON_INDEX_B64_INDEX_INT32 20008
#define NOCON_INDEX_B128_INDEX_INT32 20016
#define NOCON_INDEX_B8_INDEX_INT64 20101
#define NOCON_INDEX_B16_INDEX_INT64 20102
#define NOCON_INDEX_B32_INDEX_INT64 20104
#define NOCON_INDEX_B64_INDEX_INT64 20108
#define NOCON_INDEX_B128_INDEX_INT64 20116

#define NOCON_INDEX_B8_INDEX_INT32_OVERLENGTH 30001
#define NOCON_INDEX_B16_INDEX_INT32_OVERLENGTH 30002
#define NOCON_INDEX_B32_INDEX_INT32_OVERLENGTH 30004
#define NOCON_INDEX_B64_INDEX_INT32_OVERLENGTH 30008
#define NOCON_INDEX_B128_INDEX_INT32_OVERLENGTH 30016
#define NOCON_INDEX_B8_INDEX_INT64_OVERLENGTH 30101
#define NOCON_INDEX_B16_INDEX_INT64_OVERLENGTH 30102
#define NOCON_INDEX_B32_INDEX_INT64_OVERLENGTH 30104
#define NOCON_INDEX_B64_INDEX_INT64_OVERLENGTH 30108
#define NOCON_INDEX_B128_INDEX_INT64_OVERLENGTH 30116


#define TILING_KEY_FULL_LOAD_DIM1_MODE1 211
#define TILING_KEY_FULL_LOAD_DIM2_MODE0 220
#define TILING_KEY_FULL_LOAD_DIM2_MODE1 221
#define DIM2 2

extern "C" __global__ __aicore__ void index(GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides,
                                                GM_ADDR indices, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  SetSysWorkspace(workspace);
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }
  TPipe tPipe;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (TILING_KEY_IS(NOCON_INDEX_B8_INDEX_INT32)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int8_t, IndexAssign<int8_t, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B16_INDEX_INT32)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<half, IndexAssign<half, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B32_INDEX_INT32)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<float, IndexAssign<float, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B64_INDEX_INT32)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int64_t, IndexAssign<int64_t, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  }  else if (TILING_KEY_IS(NOCON_INDEX_B128_INDEX_INT32)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int4, IndexAssign<int4, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B8_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int8_t, IndexAssign<int8_t, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B16_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<half, IndexAssign<half, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B32_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<float, IndexAssign<float, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B64_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int64_t, IndexAssign<int64_t, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B128_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int4, IndexAssign<int4, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B8_INDEX_INT32_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int8_t, IndexAssign<int8_t, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B16_INDEX_INT32_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<half, IndexAssign<half, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B32_INDEX_INT32_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<float, IndexAssign<float, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B64_INDEX_INT32_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int64_t, IndexAssign<int64_t, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  }  else if (TILING_KEY_IS(NOCON_INDEX_B128_INDEX_INT32_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int4, IndexAssign<int4, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B8_INDEX_INT64_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int8_t, IndexAssign<int8_t, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B16_INDEX_INT64_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<half, IndexAssign<half, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B32_INDEX_INT64_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<float, IndexAssign<float, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B64_INDEX_INT64_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int64_t, IndexAssign<int64_t, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(NOCON_INDEX_B128_INDEX_INT64_OVERLENGTH)) {
    GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
    KernelIndexNoContiguous<int4, IndexAssign<int4, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(INDEX_SIMD)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimdTilingData, tilingData, tiling);
    IndexSimd<DTYPE_INDICES> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(1)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<int8_t, IndexAssign<int8_t, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<half, IndexAssign<half, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(4)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<float, IndexAssign<float, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(8)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<int64_t, IndexAssign<int64_t, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(16)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<int4, IndexAssign<int4, uint32_t>, DTYPE_INDICES, uint32_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(INDEX_B8_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<int8_t, IndexAssign<int8_t, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(INDEX_B16_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<half, IndexAssign<half, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(INDEX_B32_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<float, IndexAssign<float, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(INDEX_B64_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<int64_t, IndexAssign<int64_t, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(INDEX_B128_INDEX_INT64)) {
    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
    KernelIndex<int4, IndexAssign<int4, uint64_t>, DTYPE_INDICES, uint64_t> op;
    op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(INDEX_PERF_B8)) {
    Process<int8_t, DTYPE_INDICES, uint32_t>(
        (__gm__ int8_t*)output, (__gm__ int8_t*)inputX, indices,
        (__gm__ const IndexPerfSimtTilingData* __restrict)tiling);
  } else if (TILING_KEY_IS(INDEX_PERF_B16)) {
    Process<half, DTYPE_INDICES, uint32_t>(
        (__gm__ half*)output, (__gm__ half*)inputX, indices, (__gm__ const IndexPerfSimtTilingData* __restrict)tiling);
  } else if (TILING_KEY_IS(INDEX_PERF_B32)) {
    Process<float, DTYPE_INDICES, uint32_t>(
        (__gm__ float*)output, (__gm__ float*)inputX, indices,
        (__gm__ const IndexPerfSimtTilingData* __restrict)tiling);
  } else if (TILING_KEY_IS(INDEX_PERF_B64)) {
    Process<int64_t, DTYPE_INDICES, uint32_t>(
        (__gm__ int64_t*)output, (__gm__ int64_t*)inputX, indices,
        (__gm__ const IndexPerfSimtTilingData* __restrict)tiling);
  } else if (TILING_KEY_IS(INDEX_PERF_B8_INDEX_INT64)) {
    Process<int8_t, DTYPE_INDICES, uint64_t>(
        (__gm__ int8_t*)output, (__gm__ int8_t*)inputX, indices,
        (__gm__ const IndexPerfSimtTilingData* __restrict)tiling);
  } else if (TILING_KEY_IS(INDEX_PERF_B16_INDEX_INT64)) {
    Process<half, DTYPE_INDICES, uint64_t>(
        (__gm__ half*)output, (__gm__ half*)inputX, indices, (__gm__ const IndexPerfSimtTilingData* __restrict)tiling);
  } else if (TILING_KEY_IS(INDEX_PERF_B32_INDEX_INT64)) {
    Process<float, DTYPE_INDICES, uint64_t>(
        (__gm__ float*)output, (__gm__ float*)inputX, indices,
        (__gm__ const IndexPerfSimtTilingData* __restrict)tiling);
  } else if (TILING_KEY_IS(INDEX_PERF_B64_INDEX_INT64)) {
    Process<int64_t, DTYPE_INDICES, uint64_t>(
        (__gm__ int64_t*)output, (__gm__ int64_t*)inputX, indices,
        (__gm__ const IndexPerfSimtTilingData* __restrict)tiling);
  } else if constexpr (IsSameType<DTYPE_X, bool>::value) {
      if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_DIM1_MODE1)) {
          GET_TILING_DATA_WITH_STRUCT(IndexFullLoadTilingData, tilingData, tiling);
          KernelIndexFullLoad<int8_t, int32_t, DTYPE_INDICES, 1, 1> op(tPipe, tilingData);
          op.Init(inputX, indexedSizes, indices, output);
          op.Process();
      } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_DIM2_MODE0)) {
          GET_TILING_DATA_WITH_STRUCT(IndexFullLoadTilingData, tilingData, tiling);
          KernelIndexFullLoad<int8_t, int32_t, DTYPE_INDICES, 0, DIM2> op(tPipe, tilingData);
          op.Init(inputX, indexedSizes, indices, output);
          op.Process();
      } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_DIM2_MODE1)) {
          GET_TILING_DATA_WITH_STRUCT(IndexFullLoadTilingData, tilingData, tiling);
          KernelIndexFullLoad<int8_t, int32_t, DTYPE_INDICES, 1, DIM2> op(tPipe, tilingData);
          op.Init(inputX, indexedSizes, indices, output);
          op.Process();
      }
  } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_DIM1_MODE1)) {
      GET_TILING_DATA_WITH_STRUCT(IndexFullLoadTilingData, tilingData, tiling);
      KernelIndexFullLoad<DTYPE_X, int32_t, DTYPE_INDICES, 1, 1> op(tPipe, tilingData);
      op.Init(inputX, indexedSizes, indices, output);
      op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_DIM2_MODE0)) {
      GET_TILING_DATA_WITH_STRUCT(IndexFullLoadTilingData, tilingData, tiling);
      KernelIndexFullLoad<DTYPE_X, int32_t, DTYPE_INDICES, 0, DIM2> op(tPipe, tilingData);
      op.Init(inputX, indexedSizes, indices, output);
      op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_DIM2_MODE1)) {
      GET_TILING_DATA_WITH_STRUCT(IndexFullLoadTilingData, tilingData, tiling);
      KernelIndexFullLoad<DTYPE_X, int32_t, DTYPE_INDICES, 1, DIM2> op(tPipe, tilingData);
      op.Init(inputX, indexedSizes, indices, output);
      op.Process();
  }
}