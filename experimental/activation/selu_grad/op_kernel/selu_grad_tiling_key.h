/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file selu_grad_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __SELUGRAD_TILING_KEY_H__
#define __SELUGRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define SELUGRAD_TPL_SCH_MODE_FP16 0
#define SELUGRAD_TPL_SCH_MODE_FP32 1
#define SELUGRAD_TPL_SCH_MODE_BF16 2
#define SELUGRAD_TPL_SCH_MODE_INT32 3
#define SELUGRAD_TPL_SCH_MODE_INT8 4
#define SELUGRAD_TPL_SCH_MODE_UINT8 5

ASCENDC_TPL_ARGS_DECL(SeluGrad, ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST, SELUGRAD_TPL_SCH_MODE_FP16,
                                                      SELUGRAD_TPL_SCH_MODE_FP32, SELUGRAD_TPL_SCH_MODE_BF16,
                                                      SELUGRAD_TPL_SCH_MODE_INT32, SELUGRAD_TPL_SCH_MODE_INT8,
                                                      SELUGRAD_TPL_SCH_MODE_UINT8));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SELUGRAD_TPL_SCH_MODE_FP16,
                                                          SELUGRAD_TPL_SCH_MODE_FP32, SELUGRAD_TPL_SCH_MODE_BF16,
                                                          SELUGRAD_TPL_SCH_MODE_INT32, SELUGRAD_TPL_SCH_MODE_INT8,
                                                          SELUGRAD_TPL_SCH_MODE_UINT8)));

#endif
