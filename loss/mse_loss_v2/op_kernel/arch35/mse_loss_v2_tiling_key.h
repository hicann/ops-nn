/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
 * \file mse_loss_v2_tiling_key.h
 * \brief MSELossV2 arch35 tiling key definition
 *
 * Template parameters:
 *   - D_T_X: data type (C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16)
 *   - BUFFER_MODE: buffer mode (0=single buffer, 1=double buffer)
 *
 * reduction (none/sum/mean) is NOT part of the tiling key: it is carried as a runtime
 * field in the tiling data and dispatched inside the kernel, so a single template instance
 * per (dtype, buffer_mode) covers all three reductions.
 */

#ifndef MSE_LOSS_V2_ARCH35_TILING_KEY_H
#define MSE_LOSS_V2_ARCH35_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(MSELossV2,
                      ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
                      ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0, 1)), );

#endif // MSE_LOSS_V2_ARCH35_TILING_KEY_H
