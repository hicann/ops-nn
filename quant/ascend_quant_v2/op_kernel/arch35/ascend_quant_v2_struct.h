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
 * \file ascend_quant_v2_struct.h
 * \brief
 */
#ifndef ASCEND_QUANT_V2_STRUCT_H_
#define ASCEND_QUANT_V2_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef TPL_PER_TENSOR

#define TPL_PER_TENSOR 0
#define TPL_PER_CHANNEL 1
#define TPL_PER_HEAD 2
#define TPL_NO_OFFSET 0
#define TPL_HAS_OFFSET 1
#define TPL_DTYPE_COMMON 0
#define TPL_DIV_MODE_DIV 1
#define TPL_DIV_MODE_MUL 2
#define TPL_ROUND_MODE_NONE 0
#define TPL_ROUND_MODE_ROUND 1
#define TPL_ROUND_MODE_FLOOR 2
#define TPL_ROUND_MODE_CEIL 3
#define TPL_ROUND_MODE_TRUNC 4
#define TPL_ROUND_MODE_RINT 5
#define TPL_ROUND_MODE_HYBRID 6
#define TPL_NO_SQRT_MODE 0
#define TPL_SQRT_MODE 1

#endif

namespace AscendQuantV2Op {
ASCENDC_TPL_ARGS_DECL(
    AscendQuantV2,
    ASCENDC_TPL_UINT_DECL(perMode, 2, ASCENDC_TPL_UI_LIST, TPL_PER_TENSOR, TPL_PER_CHANNEL, TPL_PER_HEAD),
    ASCENDC_TPL_UINT_DECL(zeroPointsDtype, 1, ASCENDC_TPL_UI_LIST, TPL_NO_OFFSET, TPL_HAS_OFFSET),
    ASCENDC_TPL_UINT_DECL(
        roundMode, 3, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_FLOOR, TPL_ROUND_MODE_CEIL,
        TPL_ROUND_MODE_TRUNC, TPL_ROUND_MODE_RINT, TPL_ROUND_MODE_HYBRID),
    ASCENDC_TPL_UINT_DECL(squareMode, 1, ASCENDC_TPL_UI_LIST, TPL_NO_SQRT_MODE, TPL_SQRT_MODE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(perMode, ASCENDC_TPL_UI_LIST, TPL_PER_TENSOR, TPL_PER_CHANNEL, TPL_PER_HEAD),
    ASCENDC_TPL_UINT_SEL(zeroPointsDtype, ASCENDC_TPL_UI_LIST, TPL_NO_OFFSET, TPL_HAS_OFFSET),
    ASCENDC_TPL_UINT_SEL(
        roundMode, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_FLOOR, TPL_ROUND_MODE_CEIL,
        TPL_ROUND_MODE_TRUNC, TPL_ROUND_MODE_RINT, TPL_ROUND_MODE_HYBRID),
    ASCENDC_TPL_UINT_SEL(squareMode, ASCENDC_TPL_UI_LIST, TPL_NO_SQRT_MODE, TPL_SQRT_MODE)));

} // namespace AscendQuantV2Op

#endif // _ASCEND_QUANT_V2_STRUCT_H_