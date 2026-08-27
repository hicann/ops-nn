/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_REQUANT_STRUCT_H_
#define ASCEND_REQUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define ASCEND_REQUANT_RANK_4 4
#define ASCEND_REQUANT_RANK_8 8

#define ASCEND_REQUANT_RELU_FALSE 0
#define ASCEND_REQUANT_RELU_TRUE 1

ASCENDC_TPL_ARGS_DECL(AscendRequant,
                      ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST, ASCEND_REQUANT_RANK_4, ASCEND_REQUANT_RANK_8),
                      ASCENDC_TPL_BOOL_DECL(DO_RELU, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ASCEND_REQUANT_RANK_4),
                                     ASCENDC_TPL_BOOL_SEL(DO_RELU, ASCEND_REQUANT_RELU_FALSE)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ASCEND_REQUANT_RANK_4),
                                     ASCENDC_TPL_BOOL_SEL(DO_RELU, ASCEND_REQUANT_RELU_TRUE)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ASCEND_REQUANT_RANK_8),
                                     ASCENDC_TPL_BOOL_SEL(DO_RELU, ASCEND_REQUANT_RELU_FALSE)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ASCEND_REQUANT_RANK_8),
                                     ASCENDC_TPL_BOOL_SEL(DO_RELU, ASCEND_REQUANT_RELU_TRUE)));

#endif
