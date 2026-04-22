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
 * \file swiglu_mx_quant_with_dual_axis_tiling_key.h
 * \brief TPL tiling key template argument declarations for SwigluMxQuantWithDualAxis
 */

#ifndef _SWIGLU_MX_QUANT_WITH_DUAL_AXIS_TILING_KEY_H_
#define _SWIGLU_MX_QUANT_WITH_DUAL_AXIS_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_SCALE_ALG_0 0
#define TPL_SCALE_ALG_1 1

#define TPL_RINT 4
#define TPL_ROUND 0
#define TPL_FLOOR 1

#define TPL_MODE_ROTATE 0  // dimNBlockNum < totalCoreNum: groups rotate across core ranges
#define TPL_MODE_BLOCK  1  // dimNBlockNum >= totalCoreNum: all cores share each group's blocks

#define TPL_NO_GROUP_INDEX 0 // groupIndex输入不存在
#define TPL_GROUP_INDEX 1 // groupIndex输入存在

namespace SwigluMxQuantWithDualAxisOp {
ASCENDC_TPL_ARGS_DECL(
    SwigluMxQuantWithDualAxis,
    ASCENDC_TPL_UINT_DECL(mode, 2, ASCENDC_TPL_UI_LIST, TPL_MODE_ROTATE, TPL_MODE_BLOCK),
    ASCENDC_TPL_UINT_DECL(roundMode, 3, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR),
    ASCENDC_TPL_UINT_DECL(scaleAlg, 1, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1),
    ASCENDC_TPL_UINT_DECL(isGroupIdx, 1, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_GROUP_INDEX));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(mode, ASCENDC_TPL_UI_LIST, TPL_MODE_ROTATE, TPL_MODE_BLOCK),
    ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR),
    ASCENDC_TPL_UINT_SEL(scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1),
    ASCENDC_TPL_UINT_SEL(isGroupIdx, ASCENDC_TPL_UI_LIST, TPL_GROUP_INDEX)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(mode, ASCENDC_TPL_UI_LIST, TPL_MODE_BLOCK),
        ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_ROUND, TPL_FLOOR),
        ASCENDC_TPL_UINT_SEL(scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_0, TPL_SCALE_ALG_1),
        ASCENDC_TPL_UINT_SEL(isGroupIdx, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX)));

} // namespace SwigluMxQuantWithDualAxisOp

#endif // _SWIGLU_MX_QUANT_WITH_DUAL_AXIS_TILING_KEY_H_
