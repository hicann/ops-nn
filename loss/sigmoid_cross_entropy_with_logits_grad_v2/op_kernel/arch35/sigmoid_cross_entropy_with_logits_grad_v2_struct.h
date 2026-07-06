/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sigmoid_cross_entropy_with_logits_grad_v2_struct.h
 * \brief
 */

#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_STRUCT_H_
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"
namespace SigmoidCrossEntropyWithLogitsGradV2Struct {
ASCENDC_TPL_ARGS_DECL(SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2, BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
                      ASCENDC_TPL_BOOL_DECL(HAS_WEIGHT, 0, 1), ASCENDC_TPL_BOOL_DECL(HAS_POS_WEIGHT, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(IS_MEAN, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_BOOL_SEL(HAS_WEIGHT, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(HAS_POS_WEIGHT, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_MEAN, 0, 1)));
} // namespace SigmoidCrossEntropyWithLogitsGradV2Struct

#endif // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_STRUCT_H_
