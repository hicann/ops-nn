/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file softsign_grad_apt.cpp
 * \brief SoftsignGrad 算子 kernel 入口 (arch35)
 *
 * 模板参数（与 softsign_grad_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 对应）：
 *   - D_T: 数据类型（由 ASCENDC_TPL_DATATYPE_DECL 定义：half / float / bfloat16_t）
 *   - BUFFER_MODE: 缓冲模式 (0=单缓冲, 1=双缓冲)
 *
 * 全量覆盖（迭代三完成）：6 个 TilingKey 组合 (FP16/FP32/BF16 x 单/双缓冲)
 */

#include "softsign_grad.h"

template <typename D_T, int BUFFER_MODE>
__global__ __aicore__ void softsign_grad(
    GM_ADDR gradients,
    GM_ADDR features,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftsignGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftsignGradTilingData, tilingData, tiling);
    NsSoftsignGrad::SoftsignGrad<D_T, BUFFER_MODE> op;
    op.Init(gradients, features, output, &tilingData);
    op.Process();
}
