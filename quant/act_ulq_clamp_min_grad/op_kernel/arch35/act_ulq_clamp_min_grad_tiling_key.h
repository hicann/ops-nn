/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file act_ulq_clamp_min_grad_tiling_key.h
 * \brief ActULQClampMinGrad TilingKey 模板参数声明（arch35 / RegBase）。
 *
 * All Reduce 类别（axis_source: implicit_all）→ TilingKey 只含 2 个 bool（去掉 isTailR）：
 *   templateType == 0：
 *     isEmptyTensor == 0 → normal 模板（主路径，全轴 AR reduce_sum）
 *     isEmptyTensor == 1 → empty 模板（0 元素输入 → 输出标量 0）
 *   templateType == 1 → group 模板（A×R 2D 分核 + Phase 2 RA mini-kernel），isEmptyTensor 固定 0
 *
 * dtype 不进 key：4 组合法 dtype 组合走 DTYPE_Y_GRAD（y_grad/x_clamped_loss/输出）+
 *   DTYPE_CLAMP_MIN_MASK（mask 承载 fp16/fp32/uint8）编译期实例化（跨 H4 不变量 #1）。
 *
 * ✅ 使用 ASCENDC_TPL_ARGS_DECL 模板编程方式
 * ❌ 禁止使用废弃的 TILING_KEY_IS 宏
 */
#ifndef OPS_ACT_ULQ_CLAMP_MIN_GRAD_TILING_KEY_H_
#define OPS_ACT_ULQ_CLAMP_MIN_GRAD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(ActULQClampMinGrad, ASCENDC_TPL_BOOL_DECL(templateType, 0, 1), // 0=normal/empty, 1=group
                      ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1)                     // 0=normal, 1=empty
);

ASCENDC_TPL_SEL(
    // normal 模板：全轴 AR reduce_sum
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(templateType, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)),
    // empty 模板：0 元素输入，输出标量 0
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(templateType, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1)),
    // group 模板：A 用不满核 + R 有并行度时的 A×R 2D 分核（isEmptyTensor 固定 0）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(templateType, 1), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)));

#endif // OPS_ACT_ULQ_CLAMP_MIN_GRAD_TILING_KEY_H_
