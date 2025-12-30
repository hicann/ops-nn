/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_update_scatter_struct.h
 * \brief
 */
#ifndef QUANT_UPDATE_SCATTER_STRUCT_H_
#define QUANT_UPDATE_SCATTER_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quant_update_scatter_tpl_def.h"

namespace QuantUpdateScatter {
ASCENDC_TPL_ARGS_DECL(
    QuantUpdateScatter,
    ASCENDC_TPL_UINT_DECL(
        SplitMode, 3, ASCENDC_TPL_UI_LIST, TPL_MODE_LITTLE_ELE_LITTLE_QUANT, TPL_MODE_LARGE_BATCH,
        TPL_MODE_LARGE_ELE_LITTLE_QUANT, TPL_MODE_LARGE_ELE_LARGE_QUANT, TPL_MODE_LARGE_BATCH_LITTLE_QUANT,
        TPL_MODE_LARGE_BATCH_LARGE_QUANT),
    ASCENDC_TPL_DTYPE_DECL(ZeroPointsType, TPL_NONE, TPL_INT32, TPL_BF16),
    ASCENDC_TPL_UINT_DECL(DivMode, 2, ASCENDC_TPL_UI_LIST, TPL_DIV_MODE_DIV, TPL_DIV_MODE_MUL),
    ASCENDC_TPL_UINT_DECL(CastRoundMode, 3, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_RINT, TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_HYBRID));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(
        SplitMode, ASCENDC_TPL_UI_LIST, TPL_MODE_LITTLE_ELE_LITTLE_QUANT, TPL_MODE_LARGE_BATCH,
        TPL_MODE_LARGE_ELE_LITTLE_QUANT, TPL_MODE_LARGE_ELE_LARGE_QUANT, TPL_MODE_LARGE_BATCH_LITTLE_QUANT,
        TPL_MODE_LARGE_BATCH_LARGE_QUANT),
    ASCENDC_TPL_DTYPE_SEL(ZeroPointsType, TPL_NONE, TPL_INT32, TPL_BF16),
    ASCENDC_TPL_UINT_SEL(DivMode, ASCENDC_TPL_UI_LIST, TPL_DIV_MODE_DIV, TPL_DIV_MODE_MUL),
    ASCENDC_TPL_UINT_SEL(CastRoundMode, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_RINT, TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_HYBRID)));

} // namespace QuantUpdateScatter

#endif // QUANT_UPDATE_SCATTER_STRUCT_H_