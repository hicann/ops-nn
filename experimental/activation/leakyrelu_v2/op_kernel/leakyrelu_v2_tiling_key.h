/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file leakyrelu_v2_tiling_key.h
 * \brief leakyrelu_v2 tiling key declare
 */

#ifndef __LEAKY_RELU_V2_TILING_KEY_H__
#define __LEAKY_RELU_V2_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define ELEMENTWISE_TPL_SCH_MODE_0 0
#define ELEMENTWISE_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(LeakyreluV2,
    ASCENDC_TPL_UINT_DECL(schMode, 1, 
    ASCENDC_TPL_UI_LIST, 
    ELEMENTWISE_TPL_SCH_MODE_0, 
    ELEMENTWISE_TPL_SCH_MODE_1),);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, 
    ASCENDC_TPL_UI_LIST, 
    ELEMENTWISE_TPL_SCH_MODE_0, 
    ELEMENTWISE_TPL_SCH_MODE_1)),);

#endif