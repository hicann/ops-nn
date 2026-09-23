/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_KEY_H
#define SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_MODE_ROTATE 0
#define TPL_MODE_BLOCK 1
#define TPL_NO_GROUP_INDEX 0
#define TPL_GROUP_INDEX 1

ASCENDC_TPL_ARGS_DECL(SwigluGroupQuantWithDualAxis,
                      ASCENDC_TPL_UINT_DECL(MODE, 1, ASCENDC_TPL_UI_LIST, TPL_MODE_ROTATE, TPL_MODE_BLOCK),
                      ASCENDC_TPL_UINT_DECL(IS_GROUP, 1, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_GROUP_INDEX),
                      ASCENDC_TPL_UINT_DECL(HAS_ATTRS, 1, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(HAS_CLAMP, 1, ASCENDC_TPL_UI_LIST, 0, 1));
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_ROTATE, TPL_MODE_BLOCK),
                         ASCENDC_TPL_UINT_SEL(IS_GROUP, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_GROUP_INDEX),
                         ASCENDC_TPL_UINT_SEL(HAS_ATTRS, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(HAS_CLAMP, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_ROTATE, TPL_MODE_BLOCK),
                         ASCENDC_TPL_UINT_SEL(IS_GROUP, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_GROUP_INDEX),
                         ASCENDC_TPL_UINT_SEL(HAS_ATTRS, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(HAS_CLAMP, ASCENDC_TPL_UI_LIST, 0, 1)));
#endif
