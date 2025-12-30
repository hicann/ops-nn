/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file scatter_nd.cpp
 * \brief
 */

#include "arch35/scatter_nd.h"
#include "arch35/scatter_nd_deterministic.h"

using namespace ScatterNdSimt;
using namespace ScatterNdDeterministic;

namespace {
#define INT_INT_INT32_TILING_KEY     101
#define INT_HALF_INT32_TILING_KEY    102
#define INT_FLOAT_INT32_TILING_KEY   103
#define INT64_INT_INT32_TILING_KEY   201
#define INT64_HALF_INT32_TILING_KEY  202
#define INT64_FLOAT_INT32_TILING_KEY 203

#define INT_INT_INT64_TILING_KEY     121
#define INT_HALF_INT64_TILING_KEY    122
#define INT_FLOAT_INT64_TILING_KEY   123
#define INT64_INT_INT64_TILING_KEY   221
#define INT64_HALF_INT64_TILING_KEY  222
#define INT64_FLOAT_INT64_TILING_KEY 223
}

extern "C" __global__ __aicore__ void scatter_nd(GM_ADDR indices, GM_ADDR x, GM_ADDR shape, GM_ADDR y,
                                                  GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  SetSysWorkspace(workspace);
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }
  GET_TILING_DATA(tilingData, tiling);

  TPipe pipe;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (TILING_KEY_IS(INT_INT_INT32_TILING_KEY)) {
    ScatterNd<int32_t, int32_t, uint32_t> op;
    op.Init(indices, x, shape, y, &tilingData, &pipe);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(INT_HALF_INT32_TILING_KEY)) {
    if (tilingData.isDeterminTemplate) {
      ScatterNdDeterministicImpl<half, int32_t> op(tilingData, pipe);
      op.Init(indices, x, y, userWS);
      op.Process();
    } else {    
      ScatterNd<int32_t, half, uint32_t> op;
      op.Init(indices, x, shape, y, &tilingData, &pipe);
      op.Process(&tilingData);
    }
  } else if (TILING_KEY_IS(INT_FLOAT_INT32_TILING_KEY)) {
    if (tilingData.isDeterminTemplate) {
      ScatterNdDeterministicImpl<float, int32_t> op(tilingData, pipe);
      op.Init(indices, x, y, userWS);
      op.Process();
    } else {
      ScatterNd<int32_t, float, uint32_t> op;
      op.Init(indices, x, shape, y, &tilingData, &pipe);
      op.Process(&tilingData);
    }
  } else if (TILING_KEY_IS(INT64_INT_INT32_TILING_KEY)) {
    ScatterNd<int64_t, int32_t, uint32_t> op;
    op.Init(indices, x, shape, y, &tilingData, &pipe);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(INT64_HALF_INT32_TILING_KEY)) {
    if (tilingData.isDeterminTemplate) {
      ScatterNdDeterministicImpl<half, int64_t> op(tilingData, pipe);
      op.Init(indices, x, y, userWS);
      op.Process();
    } else {
      ScatterNd<int64_t, half, uint32_t> op;
      op.Init(indices, x, shape, y, &tilingData, &pipe);
      op.Process(&tilingData);
    }
  } else if (TILING_KEY_IS(INT64_FLOAT_INT32_TILING_KEY)) {
    if (tilingData.isDeterminTemplate) {
      ScatterNdDeterministicImpl<float, int64_t> op(tilingData, pipe);
      op.Init(indices, x, y, userWS);
      op.Process();
    } else {
      ScatterNd<int64_t, float, uint32_t> op;
      op.Init(indices, x, shape, y, &tilingData, &pipe);
      op.Process(&tilingData);
    }
  } else if (TILING_KEY_IS(INT_INT_INT64_TILING_KEY)) {
    ScatterNd<int32_t, int32_t, uint64_t> op;
    op.Init(indices, x, shape, y, &tilingData, &pipe);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(INT_HALF_INT64_TILING_KEY)) {
    if (tilingData.isDeterminTemplate) {
      ScatterNdDeterministicImpl<half, int32_t> op(tilingData, pipe);
      op.Init(indices, x, y, userWS);
      op.Process();
    } else {
      ScatterNd<int32_t, half, uint64_t> op;
      op.Init(indices, x, shape, y, &tilingData, &pipe);
      op.Process(&tilingData);
    }
  } else if (TILING_KEY_IS(INT_FLOAT_INT64_TILING_KEY)) {
    if (tilingData.isDeterminTemplate) {
      ScatterNdDeterministicImpl<float, int32_t> op(tilingData, pipe);
      op.Init(indices, x, y, userWS);
      op.Process();
    } else {
      ScatterNd<int32_t, float, uint64_t> op;
      op.Init(indices, x, shape, y, &tilingData, &pipe);
      op.Process(&tilingData);
    }
  } else if (TILING_KEY_IS(INT64_INT_INT64_TILING_KEY)) {
    ScatterNd<int64_t, int32_t, uint64_t> op;
    op.Init(indices, x, shape, y, &tilingData, &pipe);
    op.Process(&tilingData);
  } else if (TILING_KEY_IS(INT64_HALF_INT64_TILING_KEY)) {
    if (tilingData.isDeterminTemplate) {
      ScatterNdDeterministicImpl<half, int64_t> op(tilingData, pipe);
      op.Init(indices, x, y, userWS);
      op.Process();
    } else {
      ScatterNd<int64_t, half, uint64_t> op;
      op.Init(indices, x, shape, y, &tilingData, &pipe);
      op.Process(&tilingData);
    }
  } else if (TILING_KEY_IS(INT64_FLOAT_INT64_TILING_KEY)) {
    if (tilingData.isDeterminTemplate) {
      ScatterNdDeterministicImpl<float, int64_t> op(tilingData, pipe);
      op.Init(indices, x, y, userWS);
      op.Process();
    } else {
      ScatterNd<int64_t, float, uint64_t> op;
      op.Init(indices, x, shape, y, &tilingData, &pipe);
      op.Process(&tilingData);
    }
  }
}