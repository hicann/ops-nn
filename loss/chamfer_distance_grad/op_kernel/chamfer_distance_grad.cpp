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
 * \file chamfer_distance_grad.cpp
 * \brief
 */
#include "chamfer_distance_grad.h"

// core func
extern "C" __global__ __aicore__ void chamfer_distance_grad(
    GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR idx1, GM_ADDR idx2, GM_ADDR grad_dist1, GM_ADDR grad_dist2, GM_ADDR grad_xyz1,
    GM_ADDR grad_xyz2, GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling_datas, tiling_data);
    if (TILING_KEY_IS(1)) {
        ChamferDistanceGrad<float> op32(
            xyz1, xyz2, grad_dist1, grad_dist2, idx1, idx2, grad_xyz1, grad_xyz2, &tiling_datas);
        op32.process();
    } else if (TILING_KEY_IS(2)) {
        ChamferDistanceGrad<half> op16(
            xyz1, xyz2, grad_dist1, grad_dist2, idx1, idx2, grad_xyz1, grad_xyz2, &tiling_datas);
        op16.process();
    }
}