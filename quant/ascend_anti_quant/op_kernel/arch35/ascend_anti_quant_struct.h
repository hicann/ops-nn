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
 * \file ascend_anti_quant_struct.h
 * \brief Tiling key template arguments for AscendAntiQuant.
 *
 *   Supported (input x, output y) dtype combinations:
 *     int8         -> fp16 / fp32
 *     hifloat8     -> fp16 / fp32
 *     float8_e5m2  -> fp16 / fp32
 *     float8_e4m3  -> fp16 / fp32
 */
#ifndef ASCEND_ANTI_QUANT_STRUCT_H_
#define ASCEND_ANTI_QUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define AAQ_TPL_NO_SQRT_MODE   0
#define AAQ_TPL_SQRT_MODE      1

namespace AscendAntiQuantOp {
ASCENDC_TPL_ARGS_DECL(AscendAntiQuant,
    ASCENDC_TPL_UINT_DECL(sqrtModeKey, 1, ASCENDC_TPL_UI_LIST,
        AAQ_TPL_NO_SQRT_MODE, AAQ_TPL_SQRT_MODE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(sqrtModeKey, ASCENDC_TPL_UI_LIST,
        AAQ_TPL_NO_SQRT_MODE, AAQ_TPL_SQRT_MODE)));

}  // namespace AscendAntiQuantOp
#endif  // ASCEND_ANTI_QUANT_STRUCT_H_
