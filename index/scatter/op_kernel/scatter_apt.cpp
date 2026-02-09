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
 * \file scatter_apt.cpp
 * \brief
 */

#include "arch35/scatter.h"
#include "arch35/scatter_simd.h"
#include "arch35/scatter_simt.h"

using namespace SCATTER;
using namespace AscendC;

#define TILING_KEY_INT32_INT8 0
#define TILING_KEY_INT32_UINT8 1
#define TILING_KEY_INT32_FLOAT16 2
#define TILING_KEY_INT32_BF16 3
#define TILING_KEY_INT32_FLOAT 4
#define TILING_KEY_INT32_INT32 5
#define TILING_KEY_INT64_INT8 6
#define TILING_KEY_INT64_UINT8 7
#define TILING_KEY_INT64_FLOAT16 8
#define TILING_KEY_INT64_BF16 9
#define TILING_KEY_INT64_FLOAT 10
#define TILING_KEY_INT64_INT32 11
#define TILING_KEY_INT32_INT2 12
#define TILING_KEY_INT64_INT2 13
#define TILING_KEY_UINT64_INT32_INT8 14
#define TILING_KEY_UINT64_INT32_UINT8 15
#define TILING_KEY_UINT64_INT32_FLOAT16 16
#define TILING_KEY_UINT64_INT32_BF16 17
#define TILING_KEY_UINT64_INT32_FLOAT 18
#define TILING_KEY_UINT64_INT32_INT32 19
#define TILING_KEY_UINT64_INT64_INT8 20
#define TILING_KEY_UINT64_INT64_UINT8 21
#define TILING_KEY_UINT64_INT64_FLOAT16 22
#define TILING_KEY_UINT64_INT64_BF16 23
#define TILING_KEY_UINT64_INT64_FLOAT 24
#define TILING_KEY_UINT64_INT64_INT32 25
#define TILING_KEY_UINT64_INT32_INT2 26
#define TILING_KEY_UINT64_INT64_INT2 27
#define TILING_KEY_SIMD 100
#define TILING_KEY_SIMD_PERF 200
#define TILING_KEY_ONECORE_INT32_INT8 300
#define TILING_KEY_ONECORE_INT32_UINT8 301
#define TILING_KEY_ONECORE_INT32_FLOAT16 302
#define TILING_KEY_ONECORE_INT32_BF16 303
#define TILING_KEY_ONECORE_INT32_FLOAT 304
#define TILING_KEY_ONECORE_INT32_INT32 305
#define TILING_KEY_ONECORE_INT64_INT8 306
#define TILING_KEY_ONECORE_INT64_UINT8 307
#define TILING_KEY_ONECORE_INT64_FLOAT16 308
#define TILING_KEY_ONECORE_INT64_BF16 309
#define TILING_KEY_ONECORE_INT64_FLOAT 310
#define TILING_KEY_ONECORE_INT64_INT32 311
#define TILING_KEY_ONECORE_INT32_INT2 312
#define TILING_KEY_ONECORE_INT64_INT2 313

extern "C" __global__ __aicore__ void scatter(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  SetSysWorkspace(workspace);
  GM_ADDR userWs = GetUserWorkspace(workspace);
  if (userWs == nullptr) {
    return;
  }
  GET_TILING_DATA(tilingData, tiling);
  TPipe pipeOp;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (TILING_KEY_IS(TILING_KEY_INT32_INT8)) {
    SCATTER::Scatter<int8_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT32_UINT8)) {
    SCATTER::Scatter<uint8_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT32_FLOAT16)) {
    SCATTER::Scatter<half, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT32_BF16)) {
    SCATTER::Scatter<bfloat16_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT32_FLOAT)) {
    SCATTER::Scatter<float, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT32_INT32)) {
    SCATTER::Scatter<int32_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT64_INT8)) {
    SCATTER::Scatter<int8_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT64_UINT8)) {
    SCATTER::Scatter<uint8_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT64_FLOAT16)) {
    SCATTER::Scatter<half, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT64_BF16)) {
    SCATTER::Scatter<bfloat16_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT64_FLOAT)) {
    SCATTER::Scatter<float, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT64_INT32)) {
    SCATTER::Scatter<int32_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT32_INT2)) {
    SCATTER::Scatter<int64_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_INT64_INT2)) {
    SCATTER::Scatter<int64_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT32_INT8)) {
    SCATTER::Scatter<int8_t, int32_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT32_UINT8)) {
    SCATTER::Scatter<uint8_t, int32_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT32_FLOAT16)) {
    SCATTER::Scatter<half, int32_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT32_BF16)) {
    SCATTER::Scatter<bfloat16_t, int32_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT32_FLOAT)) {
    SCATTER::Scatter<float, int32_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT32_INT32)) {
    SCATTER::Scatter<int32_t, int32_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT64_INT8)) {
    SCATTER::Scatter<int8_t, int64_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT64_UINT8)) {
    SCATTER::Scatter<uint8_t, int64_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT64_FLOAT16)) {
    SCATTER::Scatter<half, int64_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT64_BF16)) {
    SCATTER::Scatter<bfloat16_t, int64_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT64_FLOAT)) {
    SCATTER::Scatter<float, int64_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT64_INT32)) {
    SCATTER::Scatter<int32_t, int64_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT32_INT2)) {
    SCATTER::Scatter<int64_t, int32_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_UINT64_INT64_INT2)) {
    SCATTER::Scatter<int64_t, int64_t, uint64_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_SIMD)) {
    SCATTER::ScatterSimd<DTYPE_INDICES, 0> op(&tilingData, &pipeOp);
    op.Init(x, indices, updates, y);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_SIMD_PERF)) {
    SCATTER::ScatterSimd<DTYPE_INDICES, 1> op(&tilingData, &pipeOp);
    op.Init(x, indices, updates, y);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT32_INT8)) {
    SCATTER::ScatterSimt<int8_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT32_UINT8)) {
    SCATTER::ScatterSimt<uint8_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT32_FLOAT16)) {
    SCATTER::ScatterSimt<half, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT32_BF16)) {
    SCATTER::ScatterSimt<bfloat16_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT32_FLOAT)) {
    SCATTER::ScatterSimt<float, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT32_INT32)) {
    SCATTER::ScatterSimt<int32_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT64_INT8)) {
    SCATTER::ScatterSimt<int8_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT64_UINT8)) {
    SCATTER::ScatterSimt<uint8_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT64_FLOAT16)) {
    SCATTER::ScatterSimt<half, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT64_BF16)) {
    SCATTER::ScatterSimt<bfloat16_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT64_FLOAT)) {
    SCATTER::ScatterSimt<float, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT64_INT32)) {
    SCATTER::ScatterSimt<int32_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT32_INT2)) {
    SCATTER::ScatterSimt<int64_t, int32_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  } else if (TILING_KEY_IS(TILING_KEY_ONECORE_INT64_INT2)) {
    SCATTER::ScatterSimt<int64_t, int64_t, uint32_t> op;
    op.Init(x, indices, updates, y, userWs, &tilingData, &pipeOp);
    op.Process();
  }
  return;
}