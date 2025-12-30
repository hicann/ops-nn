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
 * \file leaky_relu_grad_struct.h
 * \brief leaky_relu_grad_struct
 */
#ifndef LEAKY_RELU_GRAD_STRUCT_H_
#define LEAKY_RELU_GRAD_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"

namespace LeakyReluGradOp
{
#define LEAKY_RELU_GRAD_TPL_FP16  1
#define LEAKY_RELU_GRAD_TPL_BF16  2
#define LEAKY_RELU_GRAD_TPL_FP32  3

#define LEAKY_RELU_GRAD_TPL_SCH_MODE_0 0
#define LEAKY_RELU_GRAD_TPL_SCH_MODE_1 1
// 算子自定义的tiling key字段
ASCENDC_TPL_ARGS_DECL(LeakyReluGrad, BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
                      ASCENDC_TPL_DTYPE_DECL(dType, LEAKY_RELU_GRAD_TPL_FP16, LEAKY_RELU_GRAD_TPL_BF16, LEAKY_RELU_GRAD_TPL_FP32));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, LEAKY_RELU_GRAD_TPL_FP16)),
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, LEAKY_RELU_GRAD_TPL_BF16)),
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_DTYPE_SEL(dType, LEAKY_RELU_GRAD_TPL_FP32)));
}

#endif  // LEAKY_RELU_GRAD_STRUCT_H_
