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
 * \file nll_loss_grad_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef NLLLOSSGRAD_TILING_KEY_H
#define NLLLOSSGRAD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// schMode 编码 浮点dtype × target dtype 组合，顺序与 op proto 一致
#define NLLLOSSGRAD_TPL_SCH_MODE_0 0 // float32 / int32
#define NLLLOSSGRAD_TPL_SCH_MODE_1 1 // bf16    / int32
#define NLLLOSSGRAD_TPL_SCH_MODE_2 2 // float32 / int64
#define NLLLOSSGRAD_TPL_SCH_MODE_3 3 // bf16    / int64
#define NLLLOSSGRAD_TPL_SCH_MODE_4 4 // float32 / uint8
#define NLLLOSSGRAD_TPL_SCH_MODE_5 5 // bf16    / uint8
#define NLLLOSSGRAD_TPL_SCH_MODE_6 6 // float16 / int32
#define NLLLOSSGRAD_TPL_SCH_MODE_7 7 // float16 / int64
#define NLLLOSSGRAD_TPL_SCH_MODE_8 8 // float16 / uint8

ASCENDC_TPL_ARGS_DECL(NllLossGrad, ASCENDC_TPL_UINT_DECL(schMode, 4, ASCENDC_TPL_UI_LIST, NLLLOSSGRAD_TPL_SCH_MODE_0,
                                                         NLLLOSSGRAD_TPL_SCH_MODE_1, NLLLOSSGRAD_TPL_SCH_MODE_2,
                                                         NLLLOSSGRAD_TPL_SCH_MODE_3, NLLLOSSGRAD_TPL_SCH_MODE_4,
                                                         NLLLOSSGRAD_TPL_SCH_MODE_5, NLLLOSSGRAD_TPL_SCH_MODE_6,
                                                         NLLLOSSGRAD_TPL_SCH_MODE_7, NLLLOSSGRAD_TPL_SCH_MODE_8));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, NLLLOSSGRAD_TPL_SCH_MODE_0,
                                                          NLLLOSSGRAD_TPL_SCH_MODE_1, NLLLOSSGRAD_TPL_SCH_MODE_2,
                                                          NLLLOSSGRAD_TPL_SCH_MODE_3, NLLLOSSGRAD_TPL_SCH_MODE_4,
                                                          NLLLOSSGRAD_TPL_SCH_MODE_5, NLLLOSSGRAD_TPL_SCH_MODE_6,
                                                          NLLLOSSGRAD_TPL_SCH_MODE_7, NLLLOSSGRAD_TPL_SCH_MODE_8)));

#endif
