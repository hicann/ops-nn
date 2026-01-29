/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dual_level_quant_batch_matmul_tiling_key.h
 * \brief
 */

#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_TILING_KEY_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define DLQBMM_SOC_RESERVERD 0 // 平台大类
#define DLQBMM_SOC_SUPPORT_L0C_TO_OUT 1

// tilingKey变量的定义
ASCENDC_TPL_ARGS_DECL(
    DualLevelQuantBatchMatmul,
    ASCENDC_TPL_UINT_DECL( // 平台大类
        SOC_VERSION_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, DLQBMM_SOC_RESERVERD,
        DLQBMM_SOC_SUPPORT_L0C_TO_OUT), 
);

// 可行的一个tilingKey的取值，用于做校验的
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(SOC_VERSION_TYPE, ASCENDC_TPL_UI_LIST, DLQBMM_SOC_RESERVERD), 
    ), 
);

#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_TILING_KEY_H