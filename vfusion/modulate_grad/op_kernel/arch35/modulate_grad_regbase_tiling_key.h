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
 * \file modulate_grad_regbase_tiling_key.h
 * \brief modulate_grad regbase tiling key declaration
 */

#ifndef MODULATE_GRAD_REGBASE_TILING_KEY_H
#define MODULATE_GRAD_REGBASE_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

#define MODULATE_GRAD_TILING_B 0
#define MODULATE_GRAD_TILING_D 1

// 模板参数: 切分策略 + 可选输入存在性
ASCENDC_TPL_ARGS_DECL(ModulateGrad,
                      ASCENDC_TPL_UINT_DECL(tilingStrategy, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
                                            MODULATE_GRAD_TILING_B, MODULATE_GRAD_TILING_D),
                      ASCENDC_TPL_BOOL_DECL(isScale, 0, 1), ASCENDC_TPL_BOOL_DECL(isShift, 0, 1));

// 模板参数组合
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(tilingStrategy, ASCENDC_TPL_UI_LIST, MODULATE_GRAD_TILING_B,
                             MODULATE_GRAD_TILING_D), // LIST模式穷举两种切分策略，确保二进制包含全部策略实例
        ASCENDC_TPL_BOOL_SEL(isScale, 0, 1), ASCENDC_TPL_BOOL_SEL(isShift, 0, 1)), );

#endif // MODULATE_GRAD_REGBASE_TILING_KEY_H
