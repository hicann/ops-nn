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
 * \file sigmoid_cross_entropy_with_logits_v2_tiling_key.h
 * \brief sigmoid_cross_entropy_with_logits_v2 tiling key
 */

#ifndef ASCENDC_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_KEY_H_
#define ASCENDC_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_TILING_KEY_H_

#define SIGMOIDCEV2_NONE 0
#define SIGMOIDCEV2_NO_WEIGHT 0
#define SIGMOIDCEV2_HAS_WEIGHT 1
#define REDUCTION_BIT_WIDTH 2
#define WEIGHT_BIT_WIDTH 1

#include "atvoss/broadcast/broadcast_base_struct.h"
#include "ascendc/host_api/tiling/template_argument.h"
ASCENDC_TPL_ARGS_DECL(SigmoidCrossEntropyWithLogitsV2,
    BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
    ASCENDC_TPL_UINT_DECL(Reduction, REDUCTION_BIT_WIDTH, ASCENDC_TPL_UI_LIST, SIGMOIDCEV2_NONE),
    ASCENDC_TPL_UINT_DECL(HasWeight, WEIGHT_BIT_WIDTH, ASCENDC_TPL_UI_LIST, SIGMOIDCEV2_NO_WEIGHT, SIGMOIDCEV2_HAS_WEIGHT),
    ASCENDC_TPL_UINT_DECL(HasPosWeight, WEIGHT_BIT_WIDTH, ASCENDC_TPL_UI_LIST, SIGMOIDCEV2_NO_WEIGHT, SIGMOIDCEV2_HAS_WEIGHT)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
        ASCENDC_TPL_UINT_SEL(Reduction, ASCENDC_TPL_UI_LIST, SIGMOIDCEV2_NONE),
        ASCENDC_TPL_UINT_SEL(HasWeight, ASCENDC_TPL_UI_LIST, SIGMOIDCEV2_NO_WEIGHT, SIGMOIDCEV2_HAS_WEIGHT),
        ASCENDC_TPL_UINT_SEL(HasPosWeight, ASCENDC_TPL_UI_LIST, SIGMOIDCEV2_NO_WEIGHT, SIGMOIDCEV2_HAS_WEIGHT)
    ),
);

#endif