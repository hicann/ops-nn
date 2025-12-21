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
 * \file apply_adam_w_quant.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "apply_adam_w_quant_fp32.h"
#include "apply_adam_w_quant_fp16.h"

using namespace ApplyAdamWQuantNS;

extern "C" __global__ __aicore__ void apply_adam_w_quant(
    GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR qmap_m, GM_ADDR qmap_v, GM_ADDR absmax_m, GM_ADDR absmax_v,
    GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref, GM_ADDR absmax_m_ref, GM_ADDR absmax_v_ref,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data_in, tiling);
        if (TILING_KEY_IS(100)) {
            ApplyAdamWQuant<float, int64_t> op;
            op.Init(
                var, grad, m, v, qmap_m, qmap_v, absmax_m, absmax_v, step, var_ref, m_ref, v_ref, absmax_m_ref,
                absmax_v_ref, &tiling_data_in);
            op.Process();
        }
        else if (TILING_KEY_IS(200)) {
            ApplyAdamWQuant16<float, int64_t, half> op;
            op.Init(
                var, grad, m, v, qmap_m, qmap_v, absmax_m, absmax_v, step, var_ref, m_ref, v_ref, absmax_m_ref,
                absmax_v_ref, &tiling_data_in);
            op.Process();
        }
        else if (TILING_KEY_IS(300)) {
            ApplyAdamWQuant16<float, int64_t, bfloat16_t> op;
            op.Init(
                var, grad, m, v, qmap_m, qmap_v, absmax_m, absmax_v, step, var_ref, m_ref, v_ref, absmax_m_ref,
                absmax_v_ref, &tiling_data_in);
            op.Process();
        }
}