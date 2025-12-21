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
 * \file swi_glu.cpp
 * \brief
 */
#include "arch35/swi_glu_impl.hpp"
#include "arch35/swi_glu_bf16.hpp"
#include "arch35/swi_glu_single.hpp"

using namespace AscendC;

#define TILING_KEY_EMPTY 100

extern "C" __global__ __aicore__ void swi_glu(GM_ADDR input_gm, GM_ADDR output_gm,
                                              GM_ADDR workspace, GM_ADDR tiling) {
  if TILING_KEY_IS(TILING_KEY_EMPTY) {
    return;
  }

  // DT_FLOAT = 0,            // float type
  // DT_FLOAT16 = 1,          // fp16 type
  // DT_BF16 = 27,            // bf16 type
  GET_TILING_DATA(tempTilingGm, tiling);
  if (tempTilingGm.isDoubleBuffer == 1) {
    if (TILING_KEY_IS(1)) {
      SwigluSingle<SwigluVectorBF16<half, float, half, 2>, half, half> op;
      op.Init(input_gm, nullptr, output_gm, tempTilingGm);
      op.Process();
    } else if (TILING_KEY_IS(0)) {
      SwigluSingle<SwigluVector<float, float, 2>, float, float> op;
      op.Init(input_gm, nullptr, output_gm, tempTilingGm);
      op.Process();
    } else if (TILING_KEY_IS(27)) {
      SwigluSingle<SwigluVectorBF16<bfloat16_t, float, bfloat16_t, 2>, bfloat16_t, bfloat16_t> op;
      op.Init(input_gm, nullptr, output_gm, tempTilingGm);
      op.Process();
    }
  } else {
    if (TILING_KEY_IS(1)) {
      SwigluSingle<SwigluVectorBF16<half, float, half, 1>, half, half> op;
      op.Init(input_gm, nullptr, output_gm, tempTilingGm);
      op.Process();
    } else if (TILING_KEY_IS(0)) {
      SwigluSingle<SwigluVector<float, float, 1>, float, float> op;
      op.Init(input_gm, nullptr, output_gm, tempTilingGm);
      op.Process();
    } else if (TILING_KEY_IS(27)) {
      SwigluSingle<SwigluVectorBF16<bfloat16_t, float, bfloat16_t, 1>, bfloat16_t, bfloat16_t> op;
      op.Init(input_gm, nullptr, output_gm, tempTilingGm);
      op.Process();
    }
  }
}