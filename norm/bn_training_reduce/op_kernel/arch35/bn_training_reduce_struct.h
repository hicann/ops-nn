/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_TRAINING_REDUCE_STRUCT_H_
#define BN_TRAINING_REDUCE_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(BNTrainingReduce, ASCENDC_TPL_BOOL_DECL(templateType, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1), ASCENDC_TPL_BOOL_DECL(isTailR, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(isDeterministic, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(templateType, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0),
                                     ASCENDC_TPL_BOOL_SEL(isTailR, 0, 1), ASCENDC_TPL_BOOL_SEL(isDeterministic, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(templateType, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1),
                                     ASCENDC_TPL_BOOL_SEL(isTailR, 0, 1), ASCENDC_TPL_BOOL_SEL(isDeterministic, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(templateType, 1), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0),
                                     ASCENDC_TPL_BOOL_SEL(isTailR, 0, 1), ASCENDC_TPL_BOOL_SEL(isDeterministic, 0, 1)));

#endif // BN_TRAINING_REDUCE_STRUCT_H_
