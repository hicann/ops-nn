/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_ftrl_tiling_key.h
 * \brief ApplyFtrl TilingKey template-argument declaration (ascend910b / DAV_2201).
 *
 * Single Elementwise tiling strategy; TilingKey carries three template params
 * (DESIGN s3.1, schema 3x2x2 = 12 combinations):
 *   - D_T_VAR  : input/output dtype (8 inputs share dtype, taken from input0=var):
 *                C_DT_FLOAT16 / C_DT_BF16 / C_DT_FLOAT.
 *   - PAD_TAIL : whether totalElements is a non-32B-aligned tail (0/1). The kernel
 *                always uses DataCopyPad (alignment-tolerant), so this flag is the
 *                TilingKey discriminator + reserved aligned-only fast-path hint.
 *   - HAS_L1   : compile-time hint that l1 may be non-zero (0/1). When 0 the kernel
 *                statically drops the sign + soft-threshold + L1-gate branch
 *                (var = -linear_t / quadratic). Host cannot read the Device GM
 *                scalar l1 at tiling time, so it defaults to 1 and the kernel takes
 *                a runtime fast-path when l1Scalar_ == 0.
 *
 * Iteration-1 validates the main combination (C_DT_FLOAT16, PAD_TAIL=0, HAS_L1=1);
 * all 12 combinations are declared so every binary is compilable/instantiable.
 *
 * Uses ASCENDC_TPL_ARGS_DECL template-argument mechanism. TILING_KEY_IS forbidden
 * (DESIGN s3.1 / REQUIREMENTS s8.4).
 */

#ifndef APPLY_FTRL_TILING_KEY_H
#define APPLY_FTRL_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(ApplyFtrl,
    ASCENDC_TPL_DATATYPE_DECL(D_T_VAR, C_DT_FLOAT16, C_DT_BF16, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(PAD_TAIL, 8, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(HAS_L1,   8, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_VAR, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(PAD_TAIL, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HAS_L1,   ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_VAR, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(PAD_TAIL, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HAS_L1,   ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_VAR, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(PAD_TAIL, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(HAS_L1,   ASCENDC_TPL_UI_LIST, 0, 1)
    )
);

#endif  // APPLY_FTRL_TILING_KEY_H
