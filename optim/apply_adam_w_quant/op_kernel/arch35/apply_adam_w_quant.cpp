/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_adam_w_quant.cpp
 * \brief ApplyAdamWQuant arch35 (Ascend950 / DAV_3510) kernel entry.
 *
 * arch35 移植:计算逻辑与 A2 完全一致(blockwise-256 量化 AdamW),复用同族 A2 的
 * 手写 TPipe/TBuf + 高层 Vec API 计算头(base/fp32/fp16),仅 tiling 数据从 framework
 * GET_TILING_DATA 换成 regbase 的 plain-POD GET_TILING_DATA_WITH_STRUCT。dtype 分发沿用
 * A2 的 TilingKey:fp32=100 / fp16=200 / bf16=300。kernel 入口风格对齐 norm/deep_norm。
 */
#include "kernel_operator.h"
#include "apply_adam_w_quant_tiling_data.h"
#include "apply_adam_w_quant_fp32.h"
#include "apply_adam_w_quant_fp16.h"

using namespace ApplyAdamWQuantNS;

extern "C" __global__ __aicore__ void apply_adam_w_quant(GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v,
                                                         GM_ADDR qmap_m, GM_ADDR qmap_v, GM_ADDR absmax_m,
                                                         GM_ADDR absmax_v, GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref,
                                                         GM_ADDR v_ref, GM_ADDR absmax_m_ref, GM_ADDR absmax_v_ref,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ApplyAdamWQuantRegbaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamWQuantRegbaseTilingData, tiling_data_in, tiling);
    if (TILING_KEY_IS(100)) {
        ApplyAdamWQuant<float, int64_t> op;
        op.Init(var, grad, m, v, qmap_m, qmap_v, absmax_m, absmax_v, step, var_ref, m_ref, v_ref, absmax_m_ref,
                absmax_v_ref, &tiling_data_in);
        op.Process();
    } else if (TILING_KEY_IS(200)) {
        ApplyAdamWQuant16<float, int64_t, half> op;
        op.Init(var, grad, m, v, qmap_m, qmap_v, absmax_m, absmax_v, step, var_ref, m_ref, v_ref, absmax_m_ref,
                absmax_v_ref, &tiling_data_in);
        op.Process();
    } else if (TILING_KEY_IS(300)) {
        ApplyAdamWQuant16<float, int64_t, bfloat16_t> op;
        op.Init(var, grad, m, v, qmap_m, qmap_v, absmax_m, absmax_v, step, var_ref, m_ref, v_ref, absmax_m_ref,
                absmax_v_ref, &tiling_data_in);
        op.Process();
    }
}
