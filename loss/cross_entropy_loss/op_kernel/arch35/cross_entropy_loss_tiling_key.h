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
 * \file resize_linear_tiling_key.h
 * \brief resize_linear tiling key declare
 */

#ifndef _CROSS_ENTROPY_LOSS_TILING_KEY_DECL_H_
#define _CROSS_ENTROPY_LOSS_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

#define CROSS_ENTROPY_LOSS_MODE_0 0
#define CROSS_ENTROPY_LOSS_MODE_1 1
#define CROSS_ENTROPY_LOSS_MODE_2 2

#define CROSS_ENTROPY_LOSS_TPL_KEY_DECL()                                                                     \
    ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),                             \
        ASCENDC_TPL_UINT_DECL(                                                                                \
            reduction, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, CROSS_ENTROPY_LOSS_MODE_0, 1,                   \
            CROSS_ENTROPY_LOSS_MODE_2),                                                                       \
        ASCENDC_TPL_UINT_DECL(isWeight, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, CROSS_ENTROPY_LOSS_MODE_0, 1), \
        ASCENDC_TPL_UINT_DECL(labelS, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, CROSS_ENTROPY_LOSS_MODE_0, 1),   \
        ASCENDC_TPL_UINT_DECL(ignorex, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, CROSS_ENTROPY_LOSS_MODE_0, 1)

#define CROSS_ENTROPY_LOSS_TPL_NO_EMPTY_KEY_SEL()                                              \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, 0, 2),                                    \
        ASCENDC_TPL_UINT_SEL(reduction, ASCENDC_TPL_UI_LIST, 0, 1, CROSS_ENTROPY_LOSS_MODE_2), \
        ASCENDC_TPL_UINT_SEL(isWeight, ASCENDC_TPL_UI_LIST, 0, 1),                             \
        ASCENDC_TPL_UINT_SEL(labelS, ASCENDC_TPL_UI_LIST, 0, 1),                               \
        ASCENDC_TPL_UINT_SEL(ignorex, ASCENDC_TPL_UI_LIST, 0, 1)

#define CROSS_ENTROPY_LOSS_TPL_EMPTY_KEY_SEL()                                                                        \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, 1),                                                              \
        ASCENDC_TPL_UINT_SEL(reduction, ASCENDC_TPL_UI_LIST, 0, 1, CROSS_ENTROPY_LOSS_MODE_2),                        \
        ASCENDC_TPL_UINT_SEL(isWeight, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(labelS, ASCENDC_TPL_UI_LIST, 0), \
        ASCENDC_TPL_UINT_SEL(ignorex, ASCENDC_TPL_UI_LIST, 0)

ASCENDC_TPL_ARGS_DECL(CrossEntropyLoss, CROSS_ENTROPY_LOSS_TPL_KEY_DECL());

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(CROSS_ENTROPY_LOSS_TPL_NO_EMPTY_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(CROSS_ENTROPY_LOSS_TPL_EMPTY_KEY_SEL()));
#endif
