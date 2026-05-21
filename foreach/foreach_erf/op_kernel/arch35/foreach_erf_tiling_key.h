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
 * \file foreach_erf_tiling_key.h
 * \brief foreach_erf tiling key declare
 */

#ifndef FOREACH_ERF_TILING_KEY_H
#define FOREACH_ERF_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define FOREACH_ERF_TPL_SCH_MODE_FLOAT 0
#define FOREACH_ERF_TPL_SCH_MODE_FLOAT16 1
#define FOREACH_ERF_TPL_SCH_MODE_BF16 2

ASCENDC_TPL_ARGS_DECL(
    ForeachErf,
    ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST,
        FOREACH_ERF_TPL_SCH_MODE_FLOAT,
        FOREACH_ERF_TPL_SCH_MODE_FLOAT16,
        FOREACH_ERF_TPL_SCH_MODE_BF16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
        FOREACH_ERF_TPL_SCH_MODE_FLOAT,
        FOREACH_ERF_TPL_SCH_MODE_FLOAT16,
        FOREACH_ERF_TPL_SCH_MODE_BF16)));

#endif // FOREACH_ERF_TILING_KEY_H