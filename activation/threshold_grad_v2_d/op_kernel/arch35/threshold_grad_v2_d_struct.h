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
 * \file threshold_grad_v2_d_struct.h
 * \brief threshold_grad_v2_d_struct
 */
#ifndef THRESHOLD_GRAD_V2_D_STRUCT_H_
#define THRESHOLD_GRAD_V2_D_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"

namespace ThresholdGradV2DOp {
#define THRESHOLD_GRAD_V2_D_TPL_FP16 1
#define THRESHOLD_GRAD_V2_D_TPL_BF16 2
#define THRESHOLD_GRAD_V2_D_TPL_FP32 3
#define THRESHOLD_GRAD_V2_D_TPL_INT32 4
#define THRESHOLD_GRAD_V2_D_TPL_INT8 5
#define THRESHOLD_GRAD_V2_D_TPL_UINT8 6

#define THRESHOLD_GRAD_V2_D_TPL_SCH_MODE_0 0
#define THRESHOLD_GRAD_V2_D_TPL_SCH_MODE_1 1
// 算子自定义的tiling key字段
ASCENDC_TPL_ARGS_DECL(THRESHOLD_GRAD_V2_D_TPL_FP32V2D, BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
                      ASCENDC_TPL_DTYPE_DECL(dType, THRESHOLD_GRAD_V2_D_TPL_FP16, THRESHOLD_GRAD_V2_D_TPL_BF16,
                                             THRESHOLD_GRAD_V2_D_TPL_FP32, THRESHOLD_GRAD_V2_D_TPL_INT32,
                                             THRESHOLD_GRAD_V2_D_TPL_INT8, THRESHOLD_GRAD_V2_D_TPL_UINT8));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_DTYPE_SEL(dType, THRESHOLD_GRAD_V2_D_TPL_FP16)),
                ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_DTYPE_SEL(dType, THRESHOLD_GRAD_V2_D_TPL_BF16)),
                ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_DTYPE_SEL(dType, THRESHOLD_GRAD_V2_D_TPL_FP32)),
                ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_DTYPE_SEL(dType, THRESHOLD_GRAD_V2_D_TPL_INT32)),
                ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_DTYPE_SEL(dType, THRESHOLD_GRAD_V2_D_TPL_INT8)),
                ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_DTYPE_SEL(dType, THRESHOLD_GRAD_V2_D_TPL_UINT8)), );
} // namespace ThresholdGradV2DOp

#endif // THRESHOLD_GRAD_V2_D_STRUCT_H_
