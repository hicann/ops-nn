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
 * \file anti_mx_quant_struct.h
 * \brief Template argument definitions for AntiMxQuant kernel selection.
 *        Only one param to distinguish tail axis vs non-tail axis.
 */

#ifndef _ANTI_MX_QUANT_STRUCT_H_
#define _ANTI_MX_QUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace AntiMxQuantOp {
#define TPL_AXIS_TAIL     0
#define TPL_AXIS_NOT_TAIL 1

ASCENDC_TPL_ARGS_DECL(
    AntiMxQuant,
    ASCENDC_TPL_UINT_DECL(axisMode, 1, ASCENDC_TPL_UI_LIST, TPL_AXIS_TAIL, TPL_AXIS_NOT_TAIL));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(axisMode, ASCENDC_TPL_UI_LIST, TPL_AXIS_TAIL, TPL_AXIS_NOT_TAIL)));

} // namespace AntiMxQuantOp

#endif // _ANTI_MX_QUANT_STRUCT_H_
