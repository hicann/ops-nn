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
 * \file swiglu_mx_quant_tiling_key.h
 * \brief TPL tiling key template argument declarations for SwigluMxQuant
 */

#ifndef SWIGLU_MX_QUANT_TILING_KEY_H
#define SWIGLU_MX_QUANT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_NO_GROUP_INDEX     0
#define TPL_GROUP_INDEX_INT32  1
#define TPL_GROUP_INDEX_INT64  2

#define TPL_AXIS_NOT_LAST    0  // axis == -2
#define TPL_AXIS_LAST        1  // axis == -1

#define TPL_ACTIVATE_NOT_LAST 0  // activate_dim == -2
#define TPL_ACTIVATE_LAST     1  // activate_dim == -1

#define TPL_RINT 4
#define TPL_ROUND 0
#define TPL_FLOOR 1

namespace SwigluMxQuantOp {
ASCENDC_TPL_ARGS_DECL(
    SwigluMxQuant,
    ASCENDC_TPL_UINT_DECL(groupIndexType, 3, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_GROUP_INDEX_INT32, TPL_GROUP_INDEX_INT64),
    ASCENDC_TPL_UINT_DECL(axisLast, 2, ASCENDC_TPL_UI_LIST, TPL_AXIS_NOT_LAST, TPL_AXIS_LAST),
    ASCENDC_TPL_UINT_DECL(activateDimLast, 2, ASCENDC_TPL_UI_LIST, TPL_ACTIVATE_NOT_LAST, TPL_ACTIVATE_LAST),
    ASCENDC_TPL_UINT_DECL(roundMode, 3, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(groupIndexType, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_GROUP_INDEX_INT32, TPL_GROUP_INDEX_INT64),
        ASCENDC_TPL_UINT_SEL(axisLast, ASCENDC_TPL_UI_LIST, TPL_AXIS_NOT_LAST, TPL_AXIS_LAST),
        ASCENDC_TPL_UINT_SEL(activateDimLast, ASCENDC_TPL_UI_LIST, TPL_ACTIVATE_NOT_LAST, TPL_ACTIVATE_LAST),
        ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR)));
} // namespace SwigluMxQuantOp

#endif // SWIGLU_MX_QUANT_TILING_KEY_H
