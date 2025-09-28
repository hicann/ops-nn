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
 * \file cross_entropy_loss_grad_tiling_key.h
 * \brief cross_entropy_loss_grad_tiling_key declare
 */

#ifndef CROSS_ENTROPY_LOSS_GRAD_TILING_KEY_DECL_H
#define CROSS_ENTROPY_LOSS_GRAD_TILING_KEY_DECL_H
#include "ascendc/host_api/tiling/template_argument.h"

#define CROSS_ENTROPY_LOSS_GRAD_MODE_0 0
#define CROSS_ENTROPY_LOSS_GRAD_MODE_1 1
#define CROSS_ENTROPY_LOSS_GRAD_MODE_2 2


#define CROSS_ENTROPY_LOSS_GRAD_TPL_KEY_DECL()                                                  \
    ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, CROSS_ENTROPY_LOSS_GRAD_MODE_2),             \
        ASCENDC_TPL_UINT_DECL(reductionKey, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,            \
        CROSS_ENTROPY_LOSS_GRAD_MODE_0, 1, CROSS_ENTROPY_LOSS_GRAD_MODE_2),                          \
        ASCENDC_TPL_UINT_DECL(isWeight, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,             \
        CROSS_ENTROPY_LOSS_GRAD_MODE_0, 1),                                                     \
        ASCENDC_TPL_UINT_DECL(labelS, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,               \
        CROSS_ENTROPY_LOSS_GRAD_MODE_0, 1),                 \
        ASCENDC_TPL_UINT_DECL(isIgnore, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,               \
        CROSS_ENTROPY_LOSS_GRAD_MODE_0, 1)

#define CROSS_ENTROPY_LOSS_GRAD_TPL_AR_KEY_SEL()                                           \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, 0),                                    \
    ASCENDC_TPL_UINT_SEL(reductionKey, ASCENDC_TPL_UI_LIST, 0, 1, CROSS_ENTROPY_LOSS_GRAD_MODE_2),  \
    ASCENDC_TPL_UINT_SEL(isWeight, ASCENDC_TPL_UI_LIST, 0, 1),                              \
    ASCENDC_TPL_UINT_SEL(labelS, ASCENDC_TPL_UI_LIST, 0, 1),      \
    ASCENDC_TPL_UINT_SEL(isIgnore, ASCENDC_TPL_UI_LIST, 0, 1)                                   \

#define CROSS_ENTROPY_LOSS_GRAD_TPL_FULL_LOAD_KEY_SEL()                                           \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, 1),                                    \
    ASCENDC_TPL_UINT_SEL(reductionKey, ASCENDC_TPL_UI_LIST, 0, 1, CROSS_ENTROPY_LOSS_GRAD_MODE_2),  \
    ASCENDC_TPL_UINT_SEL(isWeight, ASCENDC_TPL_UI_LIST, 0, 1),                              \
    ASCENDC_TPL_UINT_SEL(labelS, ASCENDC_TPL_UI_LIST, 0, 1),      \
    ASCENDC_TPL_UINT_SEL(isIgnore, ASCENDC_TPL_UI_LIST, 0, 1)                                 \

#define CROSS_ENTROPY_LOSS_GRAD_TPL_EMPTY_KEY_SEL()                                              \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, CROSS_ENTROPY_LOSS_GRAD_MODE_2),            \
    ASCENDC_TPL_UINT_SEL(reductionKey, ASCENDC_TPL_UI_LIST, 0), \
    ASCENDC_TPL_UINT_SEL(isWeight, ASCENDC_TPL_UI_LIST, 0),                                 \
    ASCENDC_TPL_UINT_SEL(labelS, ASCENDC_TPL_UI_LIST, 0),                                    \
    ASCENDC_TPL_UINT_SEL(isIgnore, ASCENDC_TPL_UI_LIST, 0)                                    \

ASCENDC_TPL_ARGS_DECL(CrossEntropyLossGrad, CROSS_ENTROPY_LOSS_GRAD_TPL_KEY_DECL());

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(CROSS_ENTROPY_LOSS_GRAD_TPL_AR_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(CROSS_ENTROPY_LOSS_GRAD_TPL_FULL_LOAD_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(CROSS_ENTROPY_LOSS_GRAD_TPL_EMPTY_KEY_SEL())
);
#endif