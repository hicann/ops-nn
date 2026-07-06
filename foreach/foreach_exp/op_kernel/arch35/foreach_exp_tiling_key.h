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
 * \file foreach_exp_tiling_key.h
 * \brief foreach_exp tiling key declare
 */

#ifndef FOREACH_EXP_TILING_KEY_H
#define FOREACH_EXP_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define FOREACH_EXP_TPL_SCH_MODE_FLOAT 0
#define FOREACH_EXP_TPL_SCH_MODE_FLOAT16 1
#define FOREACH_EXP_TPL_SCH_MODE_BF16 2

ASCENDC_TPL_ARGS_DECL(ForeachExp,
                      ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST, FOREACH_EXP_TPL_SCH_MODE_FLOAT,
                                            FOREACH_EXP_TPL_SCH_MODE_FLOAT16, FOREACH_EXP_TPL_SCH_MODE_BF16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, FOREACH_EXP_TPL_SCH_MODE_FLOAT,
                                                          FOREACH_EXP_TPL_SCH_MODE_FLOAT16,
                                                          FOREACH_EXP_TPL_SCH_MODE_BF16)));

#endif // FOREACH_EXP_TILING_KEY_H