/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file squared_relu.cpp
 * \brief squared_relu kernel
 */

 #include "squared_relu.h"

 using namespace AscendC;

 using namespace SquaredRelu;

 extern "C" __global__ __aicore__ void squared_relu(GM_ADDR input, GM_ADDR output,
                                             GM_ADDR workspace, GM_ADDR tiling) {
     GET_TILING_DATA(tilingData, tiling);

     GM_ADDR userWs = nullptr;

 #if __CCE_AICORE__ == 220
     if (TILING_KEY_IS(1)) {
         SquaredReluND<half> op;
         op.Init(input, output, userWs, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(2)) {
         SquaredReluND<float> op;
         op.Init(input, output, userWs, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(3)) {
         SquaredReluND<bfloat16_t> op;
         op.Init(input, output, userWs, &tilingData);
         op.Process();
     }
 #else
 #endif
 }