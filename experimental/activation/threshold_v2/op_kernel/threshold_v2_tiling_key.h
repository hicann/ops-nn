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
/**
 * \file threshold_v2_tiling_key.h
 * \brief ThresholdV2 TilingKey
 *
 * Template params: BUFFER_MODE (0=single, 1=double buffer), DTYPE_SELF (DT_FLOAT/DT_FLOAT16/DT_BF16)
 */

#ifndef __THRESHOLD_V2_TILING_KEY_H__
#define __THRESHOLD_V2_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef DT_FLOAT
#define DT_FLOAT 0
#endif
#ifndef DT_FLOAT16
#define DT_FLOAT16 1
#endif
#ifndef DT_BF16
#define DT_BF16 27
#endif

ASCENDC_TPL_ARGS_DECL(ThresholdV2,
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_DTYPE_DECL(DTYPE_SELF, DT_FLOAT, DT_FLOAT16, DT_BF16)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_DTYPE_SEL(DTYPE_SELF, DT_FLOAT, DT_FLOAT16, DT_BF16)
    ),
);

#endif
