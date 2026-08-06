/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

#define MATMUL_EMU_SPLIT_WEIGHT_NO_TRANS 0
#define MATMUL_EMU_SPLIT_WEIGHT_TRANS 1

ASCENDC_TPL_ARGS_DECL(MatmulEmuSplitWeight,
                      ASCENDC_TPL_UINT_DECL(API_LEVEL, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(A_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            MATMUL_EMU_SPLIT_WEIGHT_NO_TRANS, MATMUL_EMU_SPLIT_WEIGHT_TRANS),
                      ASCENDC_TPL_UINT_DECL(B_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            MATMUL_EMU_SPLIT_WEIGHT_NO_TRANS, MATMUL_EMU_SPLIT_WEIGHT_TRANS));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_UINT_SEL(API_LEVEL, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(A_TRANS, ASCENDC_TPL_UI_LIST, MATMUL_EMU_SPLIT_WEIGHT_NO_TRANS,
                                              MATMUL_EMU_SPLIT_WEIGHT_TRANS),
                         ASCENDC_TPL_UINT_SEL(B_TRANS, ASCENDC_TPL_UI_LIST, MATMUL_EMU_SPLIT_WEIGHT_NO_TRANS,
                                              MATMUL_EMU_SPLIT_WEIGHT_TRANS),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MatmulEmuSplitWeightTilingData)), );
