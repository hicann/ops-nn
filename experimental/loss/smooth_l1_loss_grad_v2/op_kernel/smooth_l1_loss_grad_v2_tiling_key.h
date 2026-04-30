/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef SMOOTH_L1_LOSS_GRAD_V2_TILING_KEY_H_
#define SMOOTH_L1_LOSS_GRAD_V2_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define SMOOTHL1LOSSGRADV2_TPL_SCH_MODE_0 0
#define SMOOTHL1LOSSGRADV2_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    SmoothL1LossGradV2,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST,
                          SMOOTHL1LOSSGRADV2_TPL_SCH_MODE_0,
                          SMOOTHL1LOSSGRADV2_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                             SMOOTHL1LOSSGRADV2_TPL_SCH_MODE_0,
                             SMOOTHL1LOSSGRADV2_TPL_SCH_MODE_1)
    ),
);

#endif