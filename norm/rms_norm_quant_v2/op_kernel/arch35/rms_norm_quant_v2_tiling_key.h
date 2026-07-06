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
 * \file rms_norm_quant_v2_tiling_key.h
 * \brief
 */

#ifndef OP_KERNEL_RMS_NORM_QUANT_V2_TILING_KEY_H
#define OP_KERNEL_RMS_NORM_QUANT_V2_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define COMPUTE_MODE_FULL_LOAD 0
#define COMPUTE_MODE_RECOMPUTE 1

ASCENDC_TPL_ARGS_DECL(RmsNormQuantV2, ASCENDC_TPL_UINT_DECL(COMPUTE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                                            COMPUTE_MODE_FULL_LOAD, COMPUTE_MODE_RECOMPUTE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(COMPUTE_MODE, ASCENDC_TPL_UI_LIST, COMPUTE_MODE_FULL_LOAD),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(RmsNormQuantV2RegbaseFullLoadTilingData)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(COMPUTE_MODE, ASCENDC_TPL_UI_LIST, COMPUTE_MODE_RECOMPUTE),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(RmsNormQuantV2RegbaseRecomputeTilingData)));

#endif
