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
 * \file gn_training_reduce_struct.h
 * \brief GNTrainingReduce tiling key template declarations.
 */

#ifndef GN_TRAINING_REDUCE_STRUCT_H
#define GN_TRAINING_REDUCE_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(GNTrainingReduce, ASCENDC_TPL_BOOL_DECL(isGroup, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 0), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isGroup, 1), ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)));

#endif // GN_TRAINING_REDUCE_STRUCT_H
