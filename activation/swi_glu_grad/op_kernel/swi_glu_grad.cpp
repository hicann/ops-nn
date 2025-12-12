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
 * \file swi_glu_grad.cpp
 * \brief
 */
#include "swi_glu_grad_float.hpp"
#include "swi_glu_grad_bf16.hpp"
#include "swi_glu_grad_single.hpp"

using namespace AscendC;
extern "C" __global__ __aicore__ void swi_glu_grad(GM_ADDR gradout_gm, GM_ADDR input_gm, GM_ADDR output_gm,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    // DT_FLOAT = 0,            // float type
    // DT_FLOAT16 = 1,          // fp16 type
    // DT_BF16 = 27,            // bf16 type
    GET_TILING_DATA(tempTilingGm, tiling);
    if (tempTilingGm.isDoubleBuffer == 1) {
      if (TILING_KEY_IS(1)) {
        SwiGluGradSingle <SwiGluGradBF16<half, float, half, 2>, half, half> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      } else if (TILING_KEY_IS(0)) {
        SwiGluGradSingle<SwiGluGradVector<float, float, float, float, float, 2>, float, float> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      }
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220
      else if (TILING_KEY_IS(27)) {
        SwiGluGradSingle <SwiGluGradBF16<bfloat16_t, float, bfloat16_t, 2>, bfloat16_t, bfloat16_t> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      }
#endif
    } else {
      if (TILING_KEY_IS(1)) {
        SwiGluGradSingle <SwiGluGradBF16<half, float, half, 1>, half, half> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      } else if (TILING_KEY_IS(0)) {
        SwiGluGradSingle<SwiGluGradVector<float, float, float, float, float, 1>, float, float> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      }
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220
      else if (TILING_KEY_IS(27)) {
        SwiGluGradSingle <SwiGluGradBF16<bfloat16_t, float, bfloat16_t, 1>, bfloat16_t, bfloat16_t> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      }
#endif
  }
}
