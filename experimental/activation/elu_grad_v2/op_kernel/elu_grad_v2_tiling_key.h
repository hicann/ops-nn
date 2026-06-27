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
 * @file elu_grad_v2_tiling_key.h
 */
 #ifndef ELU_GRAD_V2_TILING_KEY_H
 #define ELU_GRAD_V2_TILING_KEY_H
 
 #include "ascendc/host_api/tiling/template_argument.h"
 
#define ELU_GRAD_V2_TPL_SCH_MODE_SAFE_EXP 0
#define ELU_GRAD_V2_TPL_SCH_MODE_SAFE_RESULT 1
#define ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_EXP 2
#define ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_RESULT 3
#define ELU_GRAD_V2_TPL_SCH_MODE_MIXED_EXP_FAST 4
#define ELU_GRAD_V2_TPL_SCH_MODE_MIXED_RESULT_FAST 5
#define ELU_GRAD_V2_TPL_SCH_MODE_FLOAT16_EXP_FAST 6
#define ELU_GRAD_V2_TPL_SCH_MODE_FLOAT16_RESULT_FAST 7

ASCENDC_TPL_ARGS_DECL(
    EluGradV2,
    ASCENDC_TPL_UINT_DECL(
        schMode,
        ASCENDC_TPL_4_BW,
        ASCENDC_TPL_UI_LIST,
        ELU_GRAD_V2_TPL_SCH_MODE_SAFE_EXP,
        ELU_GRAD_V2_TPL_SCH_MODE_SAFE_RESULT,
        ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_EXP,
        ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_RESULT,
        ELU_GRAD_V2_TPL_SCH_MODE_MIXED_EXP_FAST,
        ELU_GRAD_V2_TPL_SCH_MODE_MIXED_RESULT_FAST,
        ELU_GRAD_V2_TPL_SCH_MODE_FLOAT16_EXP_FAST,
        ELU_GRAD_V2_TPL_SCH_MODE_FLOAT16_RESULT_FAST));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(
            schMode,
            ASCENDC_TPL_UI_LIST,
            ELU_GRAD_V2_TPL_SCH_MODE_SAFE_EXP,
            ELU_GRAD_V2_TPL_SCH_MODE_SAFE_RESULT,
            ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_EXP,
            ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_RESULT,
            ELU_GRAD_V2_TPL_SCH_MODE_MIXED_EXP_FAST,
            ELU_GRAD_V2_TPL_SCH_MODE_MIXED_RESULT_FAST,
            ELU_GRAD_V2_TPL_SCH_MODE_FLOAT16_EXP_FAST,
            ELU_GRAD_V2_TPL_SCH_MODE_FLOAT16_RESULT_FAST)));
 
 #endif // ELU_GRAD_V2_TILING_KEY_H
 
