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
 * \file apply_proximal_gradient_descent.cpp
 * \brief ApplyProximalGradientDescent Kernel 入口
 *
 * 模板参数：
 *   - D_T_X: 数据类型（由 ASCENDC_TPL_DATATYPE_DECL 定义）
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 *
 * 参数顺序（固定）：输入 (var, alpha, l1, l2, delta) → 输出 (varOut) → workspace → tiling
 */

#include "apply_proximal_gradient_descent.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void apply_proximal_gradient_descent(
    GM_ADDR var, GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2, GM_ADDR delta,
    GM_ADDR varOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ApplyProximalGradientDescentTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyProximalGradientDescentTilingData, tilingData, tiling);
    NsApplyProximalGradientDescent::ApplyProximalGradientDescent<D_T_X, BUFFER_MODE> op;
    op.Init(var, alpha, l1, l2, delta, varOut, &tilingData);
    op.Process();
}
