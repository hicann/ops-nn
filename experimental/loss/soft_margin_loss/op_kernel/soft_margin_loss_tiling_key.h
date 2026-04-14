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
 * \file soft_margin_loss_tiling_key.h
 * \brief SoftMarginLoss TilingKey definition
 *
 * TilingKey mapping:
 * - SML_TPL_SCH_MODE_0 (0): FLOAT32 + NONE (elementwise output)
 * - SML_TPL_SCH_MODE_1 (1): FLOAT32 + REDUCE (mean/sum)
 * - SML_TPL_SCH_MODE_2 (2): FLOAT16 + NONE (elementwise output)
 * - SML_TPL_SCH_MODE_3 (3): FLOAT16 + REDUCE (mean/sum)
 */

#ifndef __SOFT_MARGIN_LOSS_TILING_KEY_H__
#define __SOFT_MARGIN_LOSS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define SML_TPL_SCH_MODE_0 0   // FLOAT32 + NONE
#define SML_TPL_SCH_MODE_1 1   // FLOAT32 + REDUCE (mean/sum)
#define SML_TPL_SCH_MODE_2 2   // FLOAT16 + NONE
#define SML_TPL_SCH_MODE_3 3   // FLOAT16 + REDUCE (mean/sum)

ASCENDC_TPL_ARGS_DECL(
    SoftMarginLoss,
    ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST,
                          SML_TPL_SCH_MODE_0, SML_TPL_SCH_MODE_1,
                          SML_TPL_SCH_MODE_2, SML_TPL_SCH_MODE_3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                         SML_TPL_SCH_MODE_0, SML_TPL_SCH_MODE_1,
                         SML_TPL_SCH_MODE_2, SML_TPL_SCH_MODE_3)));

#endif
