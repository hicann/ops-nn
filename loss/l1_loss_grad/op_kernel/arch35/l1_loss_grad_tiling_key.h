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
 * \file l1_loss_grad_tiling_key.h
 * \brief l1_loss_grad_tiling_key
 */

#ifndef L1_LOSS_GRAD_STRUCT_H
#define L1_LOSS_GRAD_STRUCT_H
 
#include "atvoss/broadcast/broadcast_base_struct.h"

#define ATTR_BIT_WIDTH 1
#define ATTR_IS_TRUE 1
// 算子自定义的tiling key字段
ASCENDC_TPL_ARGS_DECL(L1LossGrad, 
    BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
    ASCENDC_TPL_UINT_DECL(inputGradsIsScalar, ATTR_BIT_WIDTH, ASCENDC_TPL_UI_LIST, 0, ATTR_IS_TRUE));
 
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
    ASCENDC_TPL_UINT_SEL(inputGradsIsScalar, ASCENDC_TPL_UI_LIST, 0, ATTR_IS_TRUE))
);
 
#endif // L1_LOSS_GRAD_STRUCT_H