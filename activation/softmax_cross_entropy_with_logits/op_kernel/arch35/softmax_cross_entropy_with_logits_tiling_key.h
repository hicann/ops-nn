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
 * \file softmax_cross_entropy_with_logits_tiling_key.h
 * \brief softmax_cross_entropy_with_logits_tiling_key declare
 */

#ifndef SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TILING_KEY_H
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_0 0
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_1 1
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2 2

//template <uint64_t schId, uint64_t featuresBrc, uint64_t featuresBrcDim, uint64_t labelsBrc, uint64_t labelsBrcDim, uint64_t db>

#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TPL_KEY_DECL()                                                  \
    ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2),             \
    ASCENDC_TPL_UINT_DECL(featuresBrc, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,            \
    SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_0, 1, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2),                          \
    ASCENDC_TPL_UINT_DECL(labelsBrc, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,             \
    SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_0, 1, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2),                                  \
    ASCENDC_TPL_UINT_DECL(db, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,               \
    SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_0, 1)

#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TPL_SPLITR_KEY_SEL()                                           \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, 1),                                    \
    ASCENDC_TPL_UINT_SEL(featuresBrc, ASCENDC_TPL_UI_LIST, 0, 1, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2),                           \
    ASCENDC_TPL_UINT_SEL(labelsBrc, ASCENDC_TPL_UI_LIST, 0, 1, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2),                              \
    ASCENDC_TPL_UINT_SEL(db, ASCENDC_TPL_UI_LIST, 0, 1)                        \

#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TPL_EMPTY_KEY_SEL()                                           \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, 2),                                    \
    ASCENDC_TPL_UINT_SEL(featuresBrc, ASCENDC_TPL_UI_LIST, 0, 1, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2),                           \
    ASCENDC_TPL_UINT_SEL(labelsBrc, ASCENDC_TPL_UI_LIST, 0, 1, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2),                              \
    ASCENDC_TPL_UINT_SEL(db, ASCENDC_TPL_UI_LIST, 0, 1)                                                \

#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TPL_FULLLOAD_KEY_SEL()                                           \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, 0),                                    \
    ASCENDC_TPL_UINT_SEL(featuresBrc, ASCENDC_TPL_UI_LIST, 0, 1, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2),                           \
    ASCENDC_TPL_UINT_SEL(labelsBrc, ASCENDC_TPL_UI_LIST, 0, 1, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_MODE_2),                              \
    ASCENDC_TPL_UINT_SEL(db, ASCENDC_TPL_UI_LIST, 0, 1)                                                \

ASCENDC_TPL_ARGS_DECL(SoftmaxCrossEntropyWithLogits, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TPL_KEY_DECL());

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TPL_FULLLOAD_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TPL_SPLITR_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TPL_EMPTY_KEY_SEL())
);
#endif