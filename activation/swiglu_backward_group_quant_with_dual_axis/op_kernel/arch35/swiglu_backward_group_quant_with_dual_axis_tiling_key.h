/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_TILING_KEY_H
#define SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_MODE_ROTATE 0
#define TPL_MX_MODE 1
#define TPL_NO_GROUP_INDEX 0
#define TPL_GROUP_INDEX 1
#define TPL_NO_WEIGHT 0
#define TPL_WEIGHT 1
#define TPL_WEIGHT_FP16 1
#define TPL_WEIGHT_BF16 2
#define TPL_WEIGHT_FP32 3
#define TPL_NO_CLAMP 0
#define TPL_CLAMP 1

namespace SwigluBackwardGroupQuantWithDualAxisOp {
ASCENDC_TPL_ARGS_DECL(SwigluBackwardGroupQuantWithDualAxis,
                      ASCENDC_TPL_UINT_DECL(mode, 1, ASCENDC_TPL_UI_LIST, TPL_MODE_ROTATE),
                      ASCENDC_TPL_UINT_DECL(quantMode, 1, ASCENDC_TPL_UI_LIST, TPL_MX_MODE),
                      ASCENDC_TPL_UINT_DECL(hasGroupIndex, 1, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_GROUP_INDEX),
                      ASCENDC_TPL_UINT_DECL(hasWeight, 1, ASCENDC_TPL_UI_LIST, TPL_NO_WEIGHT, TPL_WEIGHT),
                      ASCENDC_TPL_DTYPE_DECL(weightDtype, TPL_WEIGHT_FP16, TPL_WEIGHT_BF16, TPL_WEIGHT_FP32),
                      ASCENDC_TPL_UINT_DECL(hasClampLimit, 1, ASCENDC_TPL_UI_LIST, TPL_NO_CLAMP, TPL_CLAMP));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(mode, ASCENDC_TPL_UI_LIST, TPL_MODE_ROTATE),
                         ASCENDC_TPL_UINT_SEL(quantMode, ASCENDC_TPL_UI_LIST, TPL_MX_MODE),
                         ASCENDC_TPL_UINT_SEL(hasGroupIndex, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX),
                         ASCENDC_TPL_UINT_SEL(hasWeight, ASCENDC_TPL_UI_LIST, TPL_NO_WEIGHT),
                         ASCENDC_TPL_DTYPE_SEL(weightDtype, TPL_WEIGHT_FP16),
                         ASCENDC_TPL_UINT_SEL(hasClampLimit, ASCENDC_TPL_UI_LIST, TPL_NO_CLAMP, TPL_CLAMP)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(mode, ASCENDC_TPL_UI_LIST, TPL_MODE_ROTATE),
                         ASCENDC_TPL_UINT_SEL(quantMode, ASCENDC_TPL_UI_LIST, TPL_MX_MODE),
                         ASCENDC_TPL_UINT_SEL(hasGroupIndex, ASCENDC_TPL_UI_LIST, TPL_GROUP_INDEX),
                         ASCENDC_TPL_UINT_SEL(hasWeight, ASCENDC_TPL_UI_LIST, TPL_NO_WEIGHT),
                         ASCENDC_TPL_DTYPE_SEL(weightDtype, TPL_WEIGHT_FP16),
                         ASCENDC_TPL_UINT_SEL(hasClampLimit, ASCENDC_TPL_UI_LIST, TPL_NO_CLAMP, TPL_CLAMP)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(mode, ASCENDC_TPL_UI_LIST, TPL_MODE_ROTATE),
                         ASCENDC_TPL_UINT_SEL(quantMode, ASCENDC_TPL_UI_LIST, TPL_MX_MODE),
                         ASCENDC_TPL_UINT_SEL(hasGroupIndex, ASCENDC_TPL_UI_LIST, TPL_GROUP_INDEX),
                         ASCENDC_TPL_UINT_SEL(hasWeight, ASCENDC_TPL_UI_LIST, TPL_WEIGHT),
                         ASCENDC_TPL_DTYPE_SEL(weightDtype, TPL_WEIGHT_FP16, TPL_WEIGHT_BF16, TPL_WEIGHT_FP32),
                         ASCENDC_TPL_UINT_SEL(hasClampLimit, ASCENDC_TPL_UI_LIST, TPL_NO_CLAMP, TPL_CLAMP)));
} // namespace SwigluBackwardGroupQuantWithDualAxisOp
#endif
