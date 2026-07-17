/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multi_scale_deformable_attn_function_tiling_key.h
 * \brief tiling key declare for multi_scale_deformable_attn_function (950 regbase)
 */

#ifndef __MSDA_TILING_KEY_H__
#define __MSDA_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define MSDA_MODE_GENERIC 0
#define MSDA_MODE_SIMT 1

ASCENDC_TPL_ARGS_DECL(MultiScaleDeformableAttnFunction,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, MSDA_MODE_GENERIC, MSDA_MODE_SIMT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, MSDA_MODE_GENERIC,
                                                          MSDA_MODE_SIMT)));

#endif // __MSDA_TILING_KEY_H__
