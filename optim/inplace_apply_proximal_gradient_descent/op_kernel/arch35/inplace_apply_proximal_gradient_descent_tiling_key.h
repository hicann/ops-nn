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
 * =============================================================================
 * inplace_apply_proximal_gradient_descent_package/op_kernel/arch35/inplace_apply_proximal_gradient_descent_tiling_key.h
 * =============================================================================
 * Role: DESIGN §6 的 TilingKey 模板声明。dtype 由 OpDef profile 生成的
 *       DTYPE_VAR 宏展开；TilingKey 只保留 BUFFER_MODE 算法维度：
 *   - BUFFER_MODE: 8-bit UINT 模板参数，取值 {0, 1}
 *                  （0=小数据阈值 SB，1=大数据阈值 DB）。
 *
 * 三种 dtype 分别生成外层二进制，每个二进制内只有 key 0/1 两个 sub-kernel。
 * Host 只把 dim0 阈值生成的 bufferMode 传给 ASCENDC_TPL_SEL_PARAM；TilingData
 * 中不存储 dtype/key 字段。
 * =============================================================================
 */

#ifndef INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_TILING_KEY_H_
#define INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(InplaceApplyProximalGradientDescent,
                      ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)));

#endif
