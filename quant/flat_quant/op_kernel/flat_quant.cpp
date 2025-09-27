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
 * \file flat_quant.cpp
 * \brief
 */

#include "flat_quant.h"

using namespace FlatQuantNS;

extern "C" __global__ __aicore__ void flat_quant(
                                                 GM_ADDR x,
                                                 GM_ADDR kronecker_p1,
                                                 GM_ADDR kronecker_p2,
                                                 GM_ADDR out,
                                                 GM_ADDR quant_scale,
                                                 GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const FlatQuantTilingData *__restrict tiling_data = &tilingData;

    GM_ADDR userWS = GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        if ASCEND_IS_AIV{
            if (tiling_data->dataType == 1) {
                TestVec<half> vec;
                vec.Init(x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, &tilingData);
                vec.Process();
            } else if (tiling_data->dataType == 2) {
                TestVec<bfloat16_t> vec;
                vec.Init(x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, &tilingData);
                vec.Process();
            }
        }
        if ASCEND_IS_AIC{
            if (tiling_data->dataType == 1) {
                TestCube<half> cube;
                cube.Init(workspace, &tilingData);
                cube.Process();
            } else if (tiling_data->dataType == 2) {
                TestCube<bfloat16_t> cube;
                cube.Init(workspace, &tilingData);
                cube.Process();
            }
        }
    }
}
