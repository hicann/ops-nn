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
 * \file foreach_pow_scalar_and_tensor.cpp
 * \brief
 */

#include "foreach_pow_scalar_and_tensor.h"

using namespace ForeachPowScalarAndTensor;

extern "C" __global__ __aicore__ void foreach_pow_scalar_and_tensor(
    GM_ADDR scalar, GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachPowScalarAndTensorND<half> op;
        op.Init(scalar, inputs, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachPowScalarAndTensorND<float> op;
        op.Init(scalar, inputs, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachPowScalarAndTensorND<int> op;
        op.Init(scalar, inputs, outputs, userWS, &tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if (TILING_KEY_IS(4)) {
        ForeachPowScalarAndTensorND<bfloat16_t> op;
        op.Init(scalar, inputs, outputs, userWS, &tilingData);
        op.Process();
#endif
    }
}
