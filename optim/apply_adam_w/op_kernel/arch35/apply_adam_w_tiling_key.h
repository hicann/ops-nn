/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_adam_w_tiling_key.h
 * \brief
 */

#ifndef APPLT_ADAM_W_STRUCT_H
#define APPLT_ADAM_W_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

#define ELEMENTWISE_TPL_SCH_MODE_0 0
#define ELEMENTWISE_TPL_SCH_MODE_1 1 
#define APPLY_ADAM_W_TPL_FP16 1
#define APPLY_ADAM_W_TPL_BF16 2
#define APPLY_ADAM_W_TPL_FP32 3
#define ATTR_BIT_WIDTH 1
#define ATTR_IS_TRUE 1

ASCENDC_TPL_ARGS_DECL(ApplyAdamW,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1),
    ASCENDC_TPL_UINT_DECL(amsgrad, ATTR_BIT_WIDTH, ASCENDC_TPL_UI_LIST, 0, ATTR_IS_TRUE),
    ASCENDC_TPL_DTYPE_DECL(dType, APPLY_ADAM_W_TPL_FP16, APPLY_ADAM_W_TPL_BF16, APPLY_ADAM_W_TPL_FP32)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1),
        ASCENDC_TPL_UINT_SEL(amsgrad, ASCENDC_TPL_UI_LIST, 0, ATTR_IS_TRUE),
        ASCENDC_TPL_DTYPE_SEL(dType, APPLY_ADAM_W_TPL_FP16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1),
        ASCENDC_TPL_UINT_SEL(amsgrad, ASCENDC_TPL_UI_LIST, 0, ATTR_IS_TRUE),
        ASCENDC_TPL_DTYPE_SEL(dType, APPLY_ADAM_W_TPL_BF16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1),
        ASCENDC_TPL_UINT_SEL(amsgrad, ASCENDC_TPL_UI_LIST, 0, ATTR_IS_TRUE),
        ASCENDC_TPL_DTYPE_SEL(dType, APPLY_ADAM_W_TPL_FP32)
    )
);
#endif // APPLT_ADAM_W_STRUCT_H