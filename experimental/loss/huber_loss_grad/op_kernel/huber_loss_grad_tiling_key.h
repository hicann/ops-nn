/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file huber_loss_grad_tiling_key.h
 * \brief HuberLossGrad tiling key declare
 */

#ifndef HUBER_LOSS_GRAD_TILING_KEY_H_
#define HUBER_LOSS_GRAD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// 数据类型由 def 的 DataType 编译变体配合 DTYPE_PREDICTIONS 宏决定，
#define HUBER_LOSS_GRAD_TPL_SCH_MODE_0 0
#define HUBER_LOSS_GRAD_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(HuberLossGrad,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, HUBER_LOSS_GRAD_TPL_SCH_MODE_0,
                                            HUBER_LOSS_GRAD_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, HUBER_LOSS_GRAD_TPL_SCH_MODE_0,
                                                          HUBER_LOSS_GRAD_TPL_SCH_MODE_1)));

#endif // HUBER_LOSS_GRAD_TILING_KEY_H_
