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
 * \file lp_loss_tiling_key.h
 * \brief lp_loss tiling key declare
 */
#ifndef LP_LOSS_TILING_KEY_H
#define LP_LOSS_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// 定义 Tiling Key 模式
// 0 为默认模式，1 为当前 Tiling 代码中使用的模式 (SetTilingKey(1))
#define LPLOSS_TPL_SCH_MODE_0 0
#define LPLOSS_TPL_SCH_MODE_1 1

// 声明 Tiling Key 模板参数
// OpType: LpLoss
// Param: schMode
// Default Value: 1 (对应 lp_loss_tiling.cpp 中的 SetTilingKey(1))
// Allow List: 0, 1
ASCENDC_TPL_ARGS_DECL(
    LpLoss, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, LPLOSS_TPL_SCH_MODE_0, LPLOSS_TPL_SCH_MODE_1));

// 注册选择器
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, LPLOSS_TPL_SCH_MODE_0, LPLOSS_TPL_SCH_MODE_1)), );

#endif // LP_LOSS_TILING_KEY_H