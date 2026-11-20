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
 * \file lp_norm_v2_tiling_key.h
 * \brief lp_norm_v2 tiling key
 */

#ifndef LP_NORM_V2_TILING_KEY_H_
#define LP_NORM_V2_TILING_KEY_H_

#define LP_NORM_V2_BIT_WIDTH 4

#include "atvoss/reduce/reduce_tiling_key_decl.h"
#include "atvoss/reduce/reduce_tiling_key_decl_batch_invariant.h"

ASCENDC_TPL_ARGS_DECL(LpNormV2, REDUCE_TPL_KEY_DECL_BATCH_INVARIANT(),
                      ASCENDC_TPL_UINT_DECL(TemplateNum, LP_NORM_V2_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 0, 6));

// The selected branch defines LPNORM_LEGACY_METADATA before this macro is expanded.
#define LPNORM_LEGACY_SELECTORS()                                                                                      \
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY_BATCH_INVARIANT(),                                                   \
                         LPNORM_LEGACY_METADATA() ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),   \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A_BATCH_INVARIANT(), LPNORM_LEGACY_METADATA() ASCENDC_TPL_UINT_SEL(    \
                                                                         TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)), \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_NORMAL_BATCH_INVARIANT(),                                           \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),                    \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_GROUP_BATCH_INVARIANT(),                                            \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),                    \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_NORMAL_BATCH_INVARIANT(),                                          \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),                    \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_GROUP_BATCH_INVARIANT(),                                           \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),                    \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARAR_NORMAL_BATCH_INVARIANT(),                                         \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),                    \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARAR_GROUP_BATCH_INVARIANT(),                                          \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),                    \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARAR_NORMAL_BATCH_INVARIANT(),                                     \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),                    \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARAR_GROUP_BATCH_INVARIANT(),                                      \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),                    \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_NORMAL_BATCH_INVARIANT(),                                    \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6)),                    \
        ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_GROUP_BATCH_INVARIANT(),                                     \
                             LPNORM_LEGACY_METADATA()                                                                  \
                                 ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6))

#if defined(ASCENDC_TPL_ENABLE_DETERMINISTIC_LEVEL) && ASCENDC_TPL_ENABLE_DETERMINISTIC_LEVEL
// Enabled tasks declare level groups [[0], [1, 2], [3]] and a separate legacy family.
#include "lp_norm_v2_reduce_selector.h"
#define LPNORM_LEGACY_METADATA() ASCENDC_TPL_LEGACY_SEL(),
// Apply the same TemplateNum range to every Reduce mode.
#define LPNORM_INSTANCE(selector, batch, ...) \
    ASCENDC_TPL_ARGS_SEL(selector(batch, __VA_ARGS__), ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 6))

// Match Host LpNormV2Tiling::TilingReduce: BatchInvariant is C1 only at level 3.
// Levels 0/1/2 must use C0; a different Host policy can produce keys absent from the selected object.
// Keep this mapping aligned with Host; legacy lookup fallback does not repair an incompatible tilingKey.
#define LPNORM_LEVEL_GROUPS(selector) \
    LPNORM_INSTANCE(selector, C0, 0), LPNORM_INSTANCE(selector, C0, 1, 2), LPNORM_INSTANCE(selector, C1, 3)

ASCENDC_TPL_SEL(
    // Empty
    LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_EMPTY),
    // A
    LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_A),
    // AR
    LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_AR_NORMAL), LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_AR_GROUP),
    // ARA
    LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_ARA_NORMAL), LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_ARA_GROUP),
    // ARAR
    LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_ARAR_NORMAL), LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_ARAR_GROUP),
    // ARARARAR
    LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_ARARARAR_NORMAL), LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_ARARARAR_GROUP),
    // ARARARARA
    LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_ARARARARA_NORMAL), LPNORM_LEVEL_GROUPS(LPNORM_REDUCE_ARARARARA_GROUP),
    // Explicit LEGACY_SEL declarations own the original boolean instance set.
    LPNORM_LEGACY_SELECTORS());

#undef LPNORM_LEVEL_GROUPS
#undef LPNORM_INSTANCE

#else
// Traditional tasks keep the original selectors and both BatchInvariant values.
#define LPNORM_LEGACY_METADATA()
ASCENDC_TPL_SEL(LPNORM_LEGACY_SELECTORS());
#endif

#undef LPNORM_LEGACY_SELECTORS
#undef LPNORM_LEGACY_METADATA

#endif
