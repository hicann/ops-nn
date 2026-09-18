/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file hard_swish_grad_v2_tiling_key.h
 * \brief HardSwishGradV2 tiling key definition
 *
 * Template parameters:
 *   - SCH_MODE: integer schedule/data-path selector (0=float32, 1=float16, 2=bfloat16)
 *   - BUFFER_MODE: buffer mode (0=single buffer, 1=double buffer)
 */

#ifndef __HARD_SWISH_GRAD_V2_TILING_KEY_H__
#define __HARD_SWISH_GRAD_V2_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// Keep dtype out of the template argument declaration. The integer schedule
// selector is chosen by host tiling from the validated input dtype.
#define HARD_SWISH_GRAD_V2_SCH_MODE_FP32 0
#define HARD_SWISH_GRAD_V2_SCH_MODE_FP16 1
#define HARD_SWISH_GRAD_V2_SCH_MODE_BF16 2

ASCENDC_TPL_ARGS_DECL(HardSwishGradV2,
                      ASCENDC_TPL_UINT_DECL(SCH_MODE, 2, ASCENDC_TPL_UI_LIST, HARD_SWISH_GRAD_V2_SCH_MODE_FP32,
                                            HARD_SWISH_GRAD_V2_SCH_MODE_FP16, HARD_SWISH_GRAD_V2_SCH_MODE_BF16),
                      ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, HARD_SWISH_GRAD_V2_SCH_MODE_FP32),
                         ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, HARD_SWISH_GRAD_V2_SCH_MODE_FP16),
                         ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, HARD_SWISH_GRAD_V2_SCH_MODE_BF16),
                         ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)), );

#endif
