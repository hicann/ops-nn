/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IN_TRAINING_UPDATE_V2_TILING_KEY_H
#define IN_TRAINING_UPDATE_V2_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define IN_TRAINING_UPDATE_V2_EMPTY_KEY 1000
#define IN_TRAINING_UPDATE_V2_NCHW_KEY 2000
#define IN_TRAINING_UPDATE_V2_NHWC_KEY 3000

ASCENDC_TPL_ARGS_DECL(INTrainingUpdateV2,
                      ASCENDC_TPL_UINT_DECL(TILING_MODE, 16, ASCENDC_TPL_UI_LIST, IN_TRAINING_UPDATE_V2_EMPTY_KEY,
                                            IN_TRAINING_UPDATE_V2_NCHW_KEY, IN_TRAINING_UPDATE_V2_NHWC_KEY));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, IN_TRAINING_UPDATE_V2_EMPTY_KEY)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, IN_TRAINING_UPDATE_V2_NCHW_KEY)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, IN_TRAINING_UPDATE_V2_NHWC_KEY)));

#endif // IN_TRAINING_UPDATE_V2_TILING_KEY_H
