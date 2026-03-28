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
 * Our normal copyright notice. Below are our remarks.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#ifndef __MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_H__
#define __MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define MAXPOOL_TPL_SCH_MODE_FP16 0
#define MAXPOOL_TPL_SCH_MODE_FP32 1
#define MAXPOOL_TPL_SCH_MODE_BF16 2

ASCENDC_TPL_ARGS_DECL(
    MaxPoolWithArgmaxV3,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
        MAXPOOL_TPL_SCH_MODE_FP16,
        MAXPOOL_TPL_SCH_MODE_FP32,
        MAXPOOL_TPL_SCH_MODE_BF16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
        MAXPOOL_TPL_SCH_MODE_FP16,
        MAXPOOL_TPL_SCH_MODE_FP32,
        MAXPOOL_TPL_SCH_MODE_BF16)));

#endif
