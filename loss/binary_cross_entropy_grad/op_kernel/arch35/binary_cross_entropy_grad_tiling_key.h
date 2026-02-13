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
 * \file binary_cross_entropy_grad_tiling_key.h
 * \brief binary_cross_entropy_grad tiling key
 */

#ifndef _BINARY_CORSS_ENTROPY_GRAD_TILING_KEY_H_
#define _BINARY_CORSS_ENTROPY_GRAD_TILING_KEY_H_

#define BCE_NONE 0
#define BCE_SUM 1
#define BCE_MEAN 2
#define BCE_HAS_WEIGHT 1
#define REDUCTION_BIT_WIDTH 2
#define WEIGHT_BIT_WIDTH 1
#include "atvoss/broadcast/broadcast_base_struct.h"
#include "ascendc/host_api/tiling/template_argument.h"

// 算子自定义的tiling key字段
ASCENDC_TPL_ARGS_DECL(BinaryCrossEntropyGrad, BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
    ASCENDC_TPL_UINT_DECL(Reduction, REDUCTION_BIT_WIDTH, ASCENDC_TPL_UI_LIST, BCE_NONE, BCE_SUM, BCE_MEAN),
    ASCENDC_TPL_UINT_DECL(HasWeight, WEIGHT_BIT_WIDTH, ASCENDC_TPL_UI_LIST, 0, BCE_HAS_WEIGHT)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
        ASCENDC_TPL_UINT_SEL(Reduction, ASCENDC_TPL_UI_LIST, BCE_NONE, BCE_SUM, BCE_MEAN),
        ASCENDC_TPL_UINT_SEL(HasWeight, ASCENDC_TPL_UI_LIST, 0, BCE_HAS_WEIGHT)
    ),
);

#endif