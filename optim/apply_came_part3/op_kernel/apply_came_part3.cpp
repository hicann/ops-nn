/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "arch35/apply_came_part3_tiling_data.h"
#include "arch35/apply_came_part3_tiling_key.h"
#include "arch35/apply_came_part3_fp32.cpp"
#include "arch35/apply_came_part3_fp16.cpp"
#include "arch35/apply_came_part3_post.h"

template <typename D_T>
__global__ __aicore__ void apply_came_part3(GM_ADDR u, GM_ADDR m_in, GM_ADDR eps, GM_ADDR beta1, GM_ADDR clip_threshold,
                                            GM_ADDR sum_square_u, GM_ADDR global_shape, GM_ADDR m_out, GM_ADDR sum_u_r,
                                            GM_ADDR sum_u_c, GM_ADDR sum_u_rc, GM_ADDR workspace, GM_ADDR cameTiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    ENABLE_DETERMINISTIC();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ApplyCamePart3TilingData);

    GET_TILING_DATA(tiling_data, cameTiling);

    CamePart3InOut camePart3InOut;
    camePart3InOut.u = u;
    camePart3InOut.mIn = m_in;
    camePart3InOut.eps = eps;
    camePart3InOut.beta1 = beta1;
    camePart3InOut.clipThreshold = clip_threshold;
    camePart3InOut.sumSquareU = sum_square_u;
    camePart3InOut.globalShape = global_shape;
    camePart3InOut.mOut = m_out;
    camePart3InOut.sumUR = sum_u_r;
    camePart3InOut.sumUC = sum_u_c;
    camePart3InOut.sumURC = sum_u_rc;

    if constexpr (sizeof(D_T) == sizeof(float)) {
        ApplyCamePart3FP32 op;
        op.Init(camePart3InOut, userWS, &tiling_data);
        op.Process();
        PipeBarrier<PIPE_ALL>();
        ApplyCamePart3Post<float> op_post;
        op_post.Init(camePart3InOut, userWS, &tiling_data);
        op_post.Process();
    } else {
        ApplyCamePart3FP16<D_T> op;
        op.Init(camePart3InOut, userWS, &tiling_data);
        op.Process();
        PipeBarrier<PIPE_ALL>();
        ApplyCamePart3Post<float> op_post;
        op_post.Init(camePart3InOut, userWS, &tiling_data);
        op_post.Process();
    }
}
