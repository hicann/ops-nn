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
 * \file quantize_struct.h
 * \brief
 */
#ifndef QUANTIZE_STRUCT_H_
#define QUANTIZE_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quantize_tpl_def.h"

namespace QuantizeOp {
ASCENDC_TPL_ARGS_DECL(
    Quantize,
    ASCENDC_TPL_UINT_DECL(
        perMode, 2, ASCENDC_TPL_UI_LIST, TPL_PER_TENSOR, TPL_PER_CHANNEL, TPL_PER_HEAD, TPL_PER_CHANNEL_NDDMA),
    ASCENDC_TPL_DTYPE_DECL(zeroPointsDtype, TPL_NONE, TPL_INT8, TPL_UINT8, TPL_INT32, TPL_FLOAT, TPL_BF16),
    ASCENDC_TPL_UINT_DECL(divMode, 2, ASCENDC_TPL_UI_LIST, TPL_DIV_MODE_DIV),
    ASCENDC_TPL_UINT_DECL(roundMode, 3, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_NONE),
    ASCENDC_TPL_UINT_DECL(squareMode, 1, ASCENDC_TPL_UI_LIST, TPL_NO_SQRT_MODE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(
        perMode, ASCENDC_TPL_UI_LIST, TPL_PER_TENSOR, TPL_PER_CHANNEL, TPL_PER_HEAD, TPL_PER_CHANNEL_NDDMA),
    ASCENDC_TPL_DTYPE_SEL(zeroPointsDtype, TPL_NONE, TPL_INT8, TPL_UINT8, TPL_INT32, TPL_FLOAT, TPL_BF16),
    ASCENDC_TPL_UINT_SEL(divMode, ASCENDC_TPL_UI_LIST, TPL_DIV_MODE_DIV),
    ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_NONE),
    ASCENDC_TPL_UINT_SEL(squareMode, ASCENDC_TPL_UI_LIST, TPL_NO_SQRT_MODE)));

} // namespace QuantizeOp

#endif // _QUANTIZE_STRUCT_H_