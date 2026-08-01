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
 * \file add_rms_norm_dynamic_quant_tiling_key.h
 * \brief
 */

#ifndef OP_KERNEL_ADD_RMS_NORM_DYNAMIC_QUANT_TILING_KEY_H
#define OP_KERNEL_ADD_RMS_NORM_DYNAMIC_QUANT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define COMPUTE_MODE_PERF 0
#define COMPUTE_MODE_NORMAL 1
#define COMPUTE_MODE_SINGLE_ROW 2
#define COMPUTE_MODE_SPLIT 3
#define COMPUTE_MODE_REDUCE_EMPTY 4

#define TPL_NO_Y3 0
#define TPL_HAS_Y3 1
#define TPL_NO_Y4 0
#define TPL_HAS_Y4 1

ASCENDC_TPL_ARGS_DECL(AddRmsNormDynamicQuant,
                      ASCENDC_TPL_UINT_DECL(COMPUTE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, COMPUTE_MODE_PERF,
                                            COMPUTE_MODE_NORMAL, COMPUTE_MODE_SINGLE_ROW, COMPUTE_MODE_SPLIT,
                                            COMPUTE_MODE_REDUCE_EMPTY),
                      ASCENDC_TPL_UINT_DECL(Y3_MODE, 1, ASCENDC_TPL_UI_LIST, TPL_NO_Y3, TPL_HAS_Y3),
                      ASCENDC_TPL_UINT_DECL(Y4_MODE, 1, ASCENDC_TPL_UI_LIST, TPL_NO_Y4, TPL_HAS_Y4));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(COMPUTE_MODE, ASCENDC_TPL_UI_LIST, COMPUTE_MODE_PERF,
                                                          COMPUTE_MODE_NORMAL, COMPUTE_MODE_SINGLE_ROW,
                                                          COMPUTE_MODE_SPLIT),
                                     ASCENDC_TPL_UINT_SEL(Y3_MODE, ASCENDC_TPL_UI_LIST, TPL_NO_Y3, TPL_HAS_Y3),
                                     ASCENDC_TPL_UINT_SEL(Y4_MODE, ASCENDC_TPL_UI_LIST, TPL_NO_Y4, TPL_HAS_Y4),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(AddRmsNormDynamicQuantRegbaseTilingData)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(COMPUTE_MODE, ASCENDC_TPL_UI_LIST, COMPUTE_MODE_REDUCE_EMPTY),
                                     ASCENDC_TPL_UINT_SEL(Y3_MODE, ASCENDC_TPL_UI_LIST, TPL_NO_Y3, TPL_HAS_Y3),
                                     ASCENDC_TPL_UINT_SEL(Y4_MODE, ASCENDC_TPL_UI_LIST, TPL_NO_Y4, TPL_HAS_Y4),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(AddRmsNormDynamicQuantEmptyTilingData)));

#endif
