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

#ifndef __HARD_SIGMOID_GRAD_TILING_KEY_H__
#define __HARD_SIGMOID_GRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// TilingKey modes:
// 0 = float32 (native precision)
// 1 = float16 (native precision)
// 2 = bfloat16 (cast to fp32 for computation)
#define HARD_SIGMOID_GRAD_MODE_FLOAT32  0
#define HARD_SIGMOID_GRAD_MODE_FLOAT16  1
#define HARD_SIGMOID_GRAD_MODE_BFLOAT16 2

ASCENDC_TPL_ARGS_DECL(HardSigmoidGrad,
    ASCENDC_TPL_UINT_DECL(schMode, 8, ASCENDC_TPL_UI_LIST,
                          HARD_SIGMOID_GRAD_MODE_FLOAT32, HARD_SIGMOID_GRAD_MODE_FLOAT16,
                          HARD_SIGMOID_GRAD_MODE_BFLOAT16)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                             HARD_SIGMOID_GRAD_MODE_FLOAT32, HARD_SIGMOID_GRAD_MODE_FLOAT16,
                             HARD_SIGMOID_GRAD_MODE_BFLOAT16)
    ),
);

#endif
