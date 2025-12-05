/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swi_glu_quant.cpp
 * \brief
 */

#include "swi_glu_quant.h"
#include "swi_glu_quant_static.h"
using namespace AscendC;
using namespace SwiGluQuantOpt;

extern "C" __global__ __aicore__ void swi_glu_quant(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR offsets,
                                                    GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace,
                                                    GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }
  GET_TILING_DATA(tilingDataIn, tiling);
  const SwiGluQuantTilingData* __restrict__ tiling_data = &tilingDataIn;
  if (GetBlockIdx() >= tiling_data->realCoreNum) {
    return;
  }
  TPipe pipe;

  #if ORIG_DTYPE_X == DT_BF16
     if (TILING_KEY_IS(206)) {
      SwiGluQuant<bfloat16_t, DTYPE_Y> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(204)) {
      SwiGluQuantStatic<bfloat16_t, DTYPE_Y, QuantType::STATIC_PER_TENSOR> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(205)) {
      SwiGluQuantStatic<bfloat16_t, DTYPE_Y, QuantType::STATIC_PER_CHANNEL> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, tiling_data);
      op.Process();
    }
  #elif ORIG_DTYPE_X == DT_FLOAT16
    if (TILING_KEY_IS(106)) {
      SwiGluQuant<half, DTYPE_Y> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, tiling_data);
      op.Process();
    }
    else if (TILING_KEY_IS(104)) {
      SwiGluQuantStatic<half, DTYPE_Y, QuantType::STATIC_PER_TENSOR> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(105)) {
      SwiGluQuantStatic<half, DTYPE_Y, QuantType::STATIC_PER_CHANNEL> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, tiling_data);
      op.Process();
    }
  #elif ORIG_DTYPE_X == DT_FLOAT
    if (TILING_KEY_IS(306)) {
      SwiGluQuant<float, DTYPE_Y> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(304)) {
      SwiGluQuantStatic<float, DTYPE_Y, QuantType::STATIC_PER_TENSOR> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, tiling_data);
      op.Process();
    } else if (TILING_KEY_IS(305)) {
      SwiGluQuantStatic<float, DTYPE_Y, QuantType::STATIC_PER_CHANNEL> op(&pipe);
      op.Init(x, smooth_scales, offsets, group_index, y, scale, workspace, tiling_data);
      op.Process();
    }
  #endif
}