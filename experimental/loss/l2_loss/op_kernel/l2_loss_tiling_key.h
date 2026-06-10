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
 * \file l2_loss_tiling_key.h
 * \brief l2_loss tiling key declarations
*/

#ifndef L2_LOSS_TILING_KEY_H
#define L2_LOSS_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define ELEMENTWISE_TPL_SCH_MODE_0 0
#define ELEMENTWISE_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(L2Loss,
    ASCENDC_TPL_UINT_DECL(schMode, 1, 
    ASCENDC_TPL_UI_LIST, 
    ELEMENTWISE_TPL_SCH_MODE_0, 
    ELEMENTWISE_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ASCENDC_TPL_UINT_SEL(schMode,
    ASCENDC_TPL_UI_LIST,
    ELEMENTWISE_TPL_SCH_MODE_0)),
    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
    ASCENDC_TPL_UINT_SEL(schMode,
    ASCENDC_TPL_UI_LIST,
    ELEMENTWISE_TPL_SCH_MODE_1)));

#endif // L2_LOSS_TILING_KEY_H
