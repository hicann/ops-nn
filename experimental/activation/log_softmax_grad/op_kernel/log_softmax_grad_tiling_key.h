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
 * \file log_softmax_grad_tiling_key.h
 * \brief log_softmax_grad tiling key declare
 */

#ifndef __LOG_SOFTMAX_GRAD_TILING_KEY_H__
#define __LOG_SOFTMAX_GRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

/* Mode场景定义 */
#define NO_NEED_REDUCE 0
#define REDUCE_TAIL 1
#define REDUCE_MID 2

/* 模板参数 */
ASCENDC_TPL_ARGS_DECL(LogSoftmaxGrad,
                      ASCENDC_TPL_UINT_DECL(SCH_MOD, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, NO_NEED_REDUCE, REDUCE_TAIL,
                                            REDUCE_MID),
                      ASCENDC_TPL_BOOL_DECL(IS_SMALL, 0, 1), ASCENDC_TPL_BOOL_DECL(IS_CONTIGUOUS, 0, 1));

/* 模板参数组合 */
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(SCH_MOD, ASCENDC_TPL_UI_LIST, NO_NEED_REDUCE, REDUCE_TAIL,
                                                          REDUCE_MID),
                                     ASCENDC_TPL_BOOL_SEL(IS_SMALL, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_CONTIGUOUS, 0, 1)));

#endif
