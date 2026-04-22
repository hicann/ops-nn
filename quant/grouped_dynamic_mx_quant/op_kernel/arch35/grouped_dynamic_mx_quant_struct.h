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
 * \file grouped_dynamic_mx_quant_struct.h
 * \brief
 */

#ifndef _GROUPED_DYNAMIC_MX_QUANT_STRUCT_H_
#define _GROUPED_DYNAMIC_MX_QUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_MODE_0 0
#define TPL_MODE_1 1

namespace GroupedDynamicMxQuantOp {
ASCENDC_TPL_ARGS_DECL(
    GroupedDynamicMxQuant,
    ASCENDC_TPL_UINT_DECL(mode, 1, ASCENDC_TPL_UI_LIST, TPL_MODE_0, TPL_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(mode, ASCENDC_TPL_UI_LIST, TPL_MODE_0, TPL_MODE_1)));

} // namespace GroupedDynamicMxQuantOp

#endif // _GROUPED_DYNAMIC_MX_QUANT_STRUCT_H_