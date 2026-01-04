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
 * \file ascend_quant_struct.h
 * \brief
 */

#ifndef _ASCEND_QUANT_STRUCT_H_
#define _ASCEND_QUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_ROUND_MODE_ROUND 1
#define TPL_ROUND_MODE_FLOOR 2
#define TPL_ROUND_MODE_CEIL 3
#define TPL_ROUND_MODE_TRUNC 4
#define TPL_ROUND_MODE_RINT 5
#define TPL_ROUND_MODE_HYBRID 6

// Extra bit to avoid TilingKey = 0
#define TPL_EXTRA_BIT_ZERO 0
#define TPL_EXTRA_BIT_ONE 1

namespace AscendQuantOp {
ASCENDC_TPL_ARGS_DECL(
    AscendQuant,
    ASCENDC_TPL_UINT_DECL(
        roundMode, 3, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_FLOOR, TPL_ROUND_MODE_CEIL,
        TPL_ROUND_MODE_TRUNC, TPL_ROUND_MODE_RINT, TPL_ROUND_MODE_HYBRID),
    ASCENDC_TPL_UINT_DECL(extraBit, 1, ASCENDC_TPL_UI_LIST, TPL_EXTRA_BIT_ZERO, TPL_EXTRA_BIT_ONE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(
        roundMode, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_FLOOR, TPL_ROUND_MODE_CEIL,
        TPL_ROUND_MODE_TRUNC, TPL_ROUND_MODE_RINT, TPL_ROUND_MODE_HYBRID),
    ASCENDC_TPL_UINT_SEL(extraBit, ASCENDC_TPL_UI_LIST, TPL_EXTRA_BIT_ZERO, TPL_EXTRA_BIT_ONE)));

} // namespace AscendQuantOp

#endif // _ASCEND_QUANT_STRUCT_H_