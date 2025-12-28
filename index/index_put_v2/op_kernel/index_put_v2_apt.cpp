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
 * \file index_put_v2.cpp
 * \brief Ascendc IndexPutV2 kernel
 */

#include "../index/arch35/index.h"

using namespace Index;

extern "C" __global__ __aicore__ void index_put_v2(GM_ADDR inputX, GM_ADDR value, GM_ADDR indexedSizes,
                                                   GM_ADDR indexedStrides, GM_ADDR indices, GM_ADDR output,
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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (tilingData.accumulateMode) {
    if (TILING_KEY_IS(2)) {
        KernelIndex<half, IndexPutAdd<half>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
      KernelIndex<bfloat16_t, IndexPutAdd<bfloat16_t>, int32_t, uint32_t> op;
      op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
      op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelIndex<int32_t, IndexPutAdd<int32_t>, int32_t,  uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        KernelIndex<int32_t, IndexPutAdd<int32_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KernelIndex<float, IndexPutAdd<float>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        KernelIndex<int64_t, IndexPutAdd<int64_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        KernelIndex<int64_t, IndexPutAdd<int64_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(12)) {
        KernelIndex<int64_t, IndexPutAdd<int64_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        KernelIndex<half, IndexPutAdd<half>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(103)) {
      KernelIndex<bfloat16_t, IndexPutAdd<bfloat16_t>, int64_t, uint32_t> op;
      op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
      op.Process();
    } else if (TILING_KEY_IS(104)) {
        KernelIndex<int32_t, IndexPutAdd<int32_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(106)) {
        KernelIndex<int32_t, IndexPutAdd<int32_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(105)) {
        KernelIndex<float, IndexPutAdd<float>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(108)) {
        KernelIndex<int64_t, IndexPutAdd<int64_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(110)) {
        KernelIndex<int64_t, IndexPutAdd<int64_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(112)) {
        KernelIndex<int64_t, IndexPutAdd<int64_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    }
  } else {
    if (TILING_KEY_IS(0)) {
        KernelIndex<uint8_t, IndexPutAssign<uint8_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelIndex<int8_t, IndexPutAssign<int8_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelIndex<half, IndexPutAssign<half>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelIndex<bfloat16_t, IndexPutAssign<bfloat16_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelIndex<int32_t, IndexPutAssign<int32_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        KernelIndex<int32_t, IndexPutAssign<int32_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KernelIndex<float, IndexPutAssign<float>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        KernelIndex<int64_t, IndexPutAssign<int64_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        KernelIndex<int64_t, IndexPutAssign<int64_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        KernelIndex<bool, IndexPutAssign<bool>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(12)) {
        KernelIndex<int64_t, IndexPutAssign<int64_t>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(16)) {
        KernelIndex<int4, IndexPutAssign<int4>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(20)) {
        KernelIndex<int4, IndexPutAssign<int4>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(24)) {
        KernelIndex<int4, IndexPutAssign<int4>, int32_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    }else if (TILING_KEY_IS(100)) {
        KernelIndex<uint8_t, IndexPutAssign<uint8_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        KernelIndex<int8_t, IndexPutAssign<int8_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        KernelIndex<half, IndexPutAssign<half>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(103)) {
        KernelIndex<bfloat16_t, IndexPutAssign<bfloat16_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(104)) {
        KernelIndex<int32_t, IndexPutAssign<int32_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(106)) {
        KernelIndex<int32_t, IndexPutAssign<int32_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(105)) {
        KernelIndex<float, IndexPutAssign<float>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(108)) {
        KernelIndex<int64_t, IndexPutAssign<int64_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(110)) {
        KernelIndex<int64_t, IndexPutAssign<int64_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(111)) {
        KernelIndex<bool, IndexPutAssign<bool>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(112)) {
        KernelIndex<int64_t, IndexPutAssign<int64_t>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(116)) {
        KernelIndex<int4, IndexPutAssign<int4>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(120)) {
        KernelIndex<int4, IndexPutAssign<int4>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    } else if (TILING_KEY_IS(124)) {
        KernelIndex<int4, IndexPutAssign<int4>, int64_t, uint32_t> op;
        op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
        op.Process();
    }
  }
}