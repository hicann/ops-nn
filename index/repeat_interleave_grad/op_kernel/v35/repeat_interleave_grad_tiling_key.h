/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_tiling_key.h
 * \brief repeat interleave grad tiling key
 */

#ifndef _REPEAT_INTERLEAVE_GRAD_TILING_KEY_H_
#define _REPEAT_INTERLEAVE_GRAD_TILING_KEY_H_

#include "atvoss/reduce/reduce_tiling_key_decl.h"

#define REPEAT_INTERLEAVE_GRAD_BIT_WIDTH 4


ASCENDC_TPL_ARGS_DECL(
    repeatInterleaveGrad, REDUCE_TPL_KEY_DECL(),
    ASCENDC_TPL_UINT_DECL(TemplateNum, REPEAT_INTERLEAVE_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 4));

ASCENDC_TPL_SEL(
    // empty
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_EMPTY(),
        ASCENDC_TPL_UINT_DECL(TemplateNum, REPEAT_INTERLEAVE_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 4)),
    // A
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_A(),
        ASCENDC_TPL_UINT_DECL(TemplateNum, REPEAT_INTERLEAVE_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 4)),
    // AR
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_AR_NORMAL(),
        ASCENDC_TPL_UINT_DECL(TemplateNum, REPEAT_INTERLEAVE_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 4)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_AR_GROUP(),
        ASCENDC_TPL_UINT_DECL(TemplateNum, REPEAT_INTERLEAVE_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 4)),

    // ARA
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_NORMAL(),
        ASCENDC_TPL_UINT_DECL(TemplateNum, REPEAT_INTERLEAVE_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 4)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_GROUP(),
        ASCENDC_TPL_UINT_DECL(TemplateNum, REPEAT_INTERLEAVE_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 4)), );

#endif