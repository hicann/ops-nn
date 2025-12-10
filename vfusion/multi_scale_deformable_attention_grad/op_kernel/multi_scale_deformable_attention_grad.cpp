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
 * \file multi_scale_deformable_attention_grad.cpp
 * \brief
 */
#include "multi_scale_deformable_attention_grad.h"

using namespace AscendC;
// core func
extern "C" __global__ __aicore__ void multi_scale_deformable_attention_grad(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
                                                                            GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm,
                                                                            GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                                                                            GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm,
                                                                            GM_ADDR grad_attn_weight_gm, GM_ADDR workspace,
                                                                            GM_ADDR tiling_data)
{

    TPipe pipe;
    GET_TILING_DATA(tiling_datas, tiling_data);
    MultiScaleDeformableAttentionGrad op;
    op.Init(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_loc_gm, attn_weight_gm, grad_output_gm,
            grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm, &tiling_datas, &pipe);
    op.InitBuffer();
    op.GetLocalTensor();
    op.ClearOutput();
    op.Process();
    op.ReleaseEventID();
}
