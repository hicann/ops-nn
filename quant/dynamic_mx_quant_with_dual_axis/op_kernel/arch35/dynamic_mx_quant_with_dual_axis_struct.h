/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_with_dual_axis_struct.h
 * \brief
 */

#ifndef _DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_WITH_STRUCT_H_
#define _DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_WITH_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_SCALE_ALG_0 0
#define TPL_SCALE_ALG_1 1

#define TPL_RINT 4
#define TPL_ROUND 0
#define TPL_FLOOR 1

#define TPL_MODE_0 0
#define TPL_MODE_1 1

namespace DynamicMxQuantWithDualAxisOp {
ASCENDC_TPL_ARGS_DECL(
    DynamicMxQuantWithDualAxis,
    ASCENDC_TPL_UINT_DECL(mode, 1, ASCENDC_TPL_UI_LIST, TPL_MODE_0, TPL_MODE_1),
    ASCENDC_TPL_UINT_DECL(roundMode, 3, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR),
    ASCENDC_TPL_UINT_DECL(scaleAlg, 1, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(mode, ASCENDC_TPL_UI_LIST, TPL_MODE_0, TPL_MODE_1),
    ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR),
    ASCENDC_TPL_UINT_SEL(scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1)));

} // namespace DynamicMxQuantWithDualAxisOp

#endif // _DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_WITH_STRUCT_H_