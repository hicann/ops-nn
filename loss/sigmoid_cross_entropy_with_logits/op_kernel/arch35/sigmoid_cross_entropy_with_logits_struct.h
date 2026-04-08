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
 * \file sigmoid_cross_entropy_with_logits_struct.h
 * \brief SigmoidCrossEntropyWithLogits 算子模板参数定义
 */

#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_STRUCT_H_
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_FP16 1
#define TPL_BF16 2
#define TPL_FP32 3

ASCENDC_TPL_ARGS_DECL(SigmoidCrossEntropyWithLogits,
    ASCENDC_TPL_DTYPE_DECL(dType, TPL_FP16, TPL_BF16, TPL_FP32)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_FP16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_BF16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_FP32)
    )
);

#endif // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_STRUCT_H_