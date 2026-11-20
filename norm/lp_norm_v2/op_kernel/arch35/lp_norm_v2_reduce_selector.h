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
 * \file lp_norm_v2_reduce_selector.h
 * \brief Private deterministic-level Reduce selectors for LpNormV2.
 * Public REDUCE_TPL_KEY_SEL_* selectors lack deterministic-level metadata; keep private
 * selectors to attach that metadata while preserving the public Reduce instance domains.
 */

#ifndef LP_NORM_V2_REDUCE_SELECTOR_H_
#define LP_NORM_V2_REDUCE_SELECTOR_H_

#include "atvoss/reduce/reduce_tiling_key_decl.h"

#if !defined(ASCENDC_TPL_ENABLE_DETERMINISTIC_LEVEL) || !ASCENDC_TPL_ENABLE_DETERMINISTIC_LEVEL
#error "Include LpNormV2 level selectors only for an enabled deterministic-level compile task"
#endif

// Index contract: deterministic d == int(level != 0), with groups [[0], [1, 2], [3]].
// BOOL uses the first level of each group, so all levels in that group must share d.
// Keep this table and LPNORM_LEVEL_GROUPS consistent; deterministic_levels.py validates
// the keys and group layout during packaging, not when these macros are defined.
#define LPNORM_REDUCE_LEVEL_BOOL_0 false
#define LPNORM_REDUCE_LEVEL_BOOL_1 true
#define LPNORM_REDUCE_LEVEL_BOOL_2 true
#define LPNORM_REDUCE_LEVEL_BOOL_3 true
#define LPNORM_REDUCE_LEVEL_BOOL_IMPL(level) LPNORM_REDUCE_LEVEL_BOOL_##level
#define LPNORM_REDUCE_LEVEL_BOOL(level, ...) LPNORM_REDUCE_LEVEL_BOOL_IMPL(level)
#define LPNORM_REDUCE_LEVEL_METADATA(...)                                 \
    ASCENDC_TPL_DETERMINISTIC_SEL(LPNORM_REDUCE_LEVEL_BOOL(__VA_ARGS__)), \
        ASCENDC_TPL_DETERMINISTIC_LEVEL_SEL(__VA_ARGS__)

// Match the selector domains of the validated opbase de414d06 Reduce implementation.
// Keep these private rules in sync when adopting another public Reduce version.
#define LPNORM_REDUCE_MODE(kernel_type, batch, pattern, ...)                                                        \
    ASCENDC_TPL_KERNEL_TYPE_SEL(kernel_type), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                               \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, batch), ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, pattern), \
        __VA_ARGS__

#define LPNORM_REDUCE_EMPTY(batch, ...)                                                                           \
    LPNORM_REDUCE_MODE(ASCENDC_TPL_AIV_ONLY, batch, 0, ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, 0), \
                       ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0),                            \
                       LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_A(batch, ...)                                                                      \
    LPNORM_REDUCE_MODE(                                                                                  \
        ASCENDC_TPL_AIV_ONLY, batch, C100, ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0), \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0), LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_AR_NORMAL(batch, ...)                                                     \
    LPNORM_REDUCE_MODE(ASCENDC_TPL_AIV_ONLY, batch, AR_PATTERN,                                 \
                       ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0),            \
                       ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C1), \
                       LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_AR_GROUP(batch, ...)                                                                        \
    LPNORM_REDUCE_MODE(                                                                                           \
        ASCENDC_TPL_MIX_AIV_1_0, batch, AR_PATTERN, ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R2), \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0), LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_ARA_NORMAL(batch, ...)                                                    \
    LPNORM_REDUCE_MODE(ASCENDC_TPL_AIV_ONLY, batch, ARA_PATTERN,                                \
                       ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0, A2R0),      \
                       ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C1), \
                       LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_ARA_GROUP(batch, ...)                                                \
    LPNORM_REDUCE_MODE(ASCENDC_TPL_MIX_AIV_1_0, batch, ARA_PATTERN,                        \
                       ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R2, A2R3), \
                       ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0),    \
                       LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_ARAR_NORMAL(batch, ...)                                                   \
    LPNORM_REDUCE_MODE(ASCENDC_TPL_AIV_ONLY, batch, ARAR_PATTERN,                               \
                       ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0, A2R0),      \
                       ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C2), \
                       LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_ARAR_GROUP(batch, ...)                                                           \
    LPNORM_REDUCE_MODE(ASCENDC_TPL_MIX_AIV_1_0, batch, ARAR_PATTERN,                                   \
                       ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R2, A1R3, A2R3, A2R4), \
                       ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0),                \
                       LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_ARARARAR_NORMAL(batch, ...)                                                      \
    LPNORM_REDUCE_MODE(ASCENDC_TPL_AIV_ONLY, batch, ARARARAR_PATTERN,                                  \
                       ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0, A2R0, A3R0, A4R0), \
                       ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C4),        \
                       LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_ARARARAR_GROUP(batch, ...)                                                                  \
    LPNORM_REDUCE_MODE(ASCENDC_TPL_MIX_AIV_1_0, batch, ARARARAR_PATTERN,                                          \
                       ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R2, A1R3, A1R4, A1R5, A2R3, A2R4, \
                                            A2R5, A2R6, A3R4, A3R5, A3R6, A3R7, A4R5, A4R6, A4R7, A4R8),          \
                       ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0),                           \
                       LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_ARARARARA_NORMAL(batch, ...)                                                           \
    LPNORM_REDUCE_MODE(ASCENDC_TPL_AIV_ONLY, batch, ARARARARA_PATTERN,                                       \
                       ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0, A2R0, A3R0, A4R0, A5R0), \
                       ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C4),              \
                       LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#define LPNORM_REDUCE_ARARARARA_GROUP(batch, ...)                                                                    \
    LPNORM_REDUCE_MODE(                                                                                              \
        ASCENDC_TPL_MIX_AIV_1_0, batch, ARARARARA_PATTERN,                                                           \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R2, A1R3, A1R4, A1R5, A2R3, A2R4, A2R5, A2R6, A3R4, \
                             A3R5, A3R6, A3R7, A4R5, A4R6, A4R7, A4R8, A5R6, A5R7, A5R8, A5R9),                      \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0), LPNORM_REDUCE_LEVEL_METADATA(__VA_ARGS__))

#endif
