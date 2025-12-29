/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef CANN_CUSTOM_OPS_FAST_GELU_STRUCT_H_
#define CANN_CUSTOM_OPS_FAST_GELU_STRUCT_H_

namespace FastGeluOp {
#define TPL_FP16 1
#define TPL_BF16 2
#define TPL_FP32 3
#define FastGelu_TPL_SCH_MODE_0 0
#define FastGelu_TPL_SCH_MODE_1 1

ASCENDC_TPL_ARGS_DECL(
    FastGelu, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, FastGelu_TPL_SCH_MODE_0, FastGelu_TPL_SCH_MODE_1),
    ASCENDC_TPL_DTYPE_DECL(dType, TPL_FP16, TPL_BF16, TPL_FP32));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, FastGelu_TPL_SCH_MODE_0, FastGelu_TPL_SCH_MODE_1),
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_FP16)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, FastGelu_TPL_SCH_MODE_0, FastGelu_TPL_SCH_MODE_1),
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_BF16)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, FastGelu_TPL_SCH_MODE_0, FastGelu_TPL_SCH_MODE_1),
        ASCENDC_TPL_DTYPE_SEL(dType, TPL_FP32)));
} // namespace FastGeluOp

#endif // CANN_CUSTOM_OPS_FAST_GELU_STRUCT_H_