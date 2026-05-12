/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
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
 * \file lp_norm_v3_tiling_key.h
 * \brief lp_norm_v3 tiling key declare
 */

#ifndef __LP_NORM_V3_TILING_KEY_H__
#define __LP_NORM_V3_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

/* Mode场景定义 */
#define LP_NORM_AXIS_NONE 0
#define LP_NORM_AXIS_0 1
#define LP_NORM_AXIS_1 2
/* 继续定义其他Mode场景... */

/* 模板参数 */
ASCENDC_TPL_ARGS_DECL(
    LpNormV3,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST, LP_NORM_AXIS_NONE, LP_NORM_AXIS_0, LP_NORM_AXIS_1));

/* 模板参数组合 */
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, LP_NORM_AXIS_NONE, LP_NORM_AXIS_0, LP_NORM_AXIS_1)));

#endif