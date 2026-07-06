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
 * \file dequant_bias_tiling_key.h
 * \brief TilingKey 模板参数定义
 *
 * 模板参数:
 *   hasBias:     0=无bias, 1=有bias
 *   hasActivate: 0=无activate_scale, 1=有activate_scale
 *
 * DTYPE_WEIGHT_SCALE 由框架从 input(1) 自动推导为类型宏
 * 变体数: 2 × 2 = 4 (tiling key) × 2 (dtype) = 8
 */

#ifndef __DEQUANT_BIAS_TILING_KEY_H__
#define __DEQUANT_BIAS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_NO_BIAS 0
#define TPL_HAS_BIAS 1
#define TPL_NO_ACTIVATE 0
#define TPL_HAS_ACTIVATE 1

#define TPL_BIAS_FLOAT 0
#define TPL_BIAS_HALF 1
#define TPL_BIAS_BFLOAT 2
#define TPL_BIAS_INT 3

namespace NsDequantBias {

ASCENDC_TPL_ARGS_DECL(DequantBias, ASCENDC_TPL_UINT_DECL(hasBias, 1, ASCENDC_TPL_UI_LIST, TPL_NO_BIAS, TPL_HAS_BIAS),
                      ASCENDC_TPL_UINT_DECL(hasActivate, 1, ASCENDC_TPL_UI_LIST, TPL_NO_ACTIVATE, TPL_HAS_ACTIVATE),
                      ASCENDC_TPL_UINT_DECL(dtypeBias, 2, ASCENDC_TPL_UI_LIST, TPL_BIAS_FLOAT, TPL_BIAS_HALF,
                                            TPL_BIAS_BFLOAT, TPL_BIAS_INT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(hasBias, ASCENDC_TPL_UI_LIST, TPL_NO_BIAS, TPL_HAS_BIAS),
                                     ASCENDC_TPL_UINT_SEL(hasActivate, ASCENDC_TPL_UI_LIST, TPL_NO_ACTIVATE,
                                                          TPL_HAS_ACTIVATE),
                                     ASCENDC_TPL_UINT_SEL(dtypeBias, ASCENDC_TPL_UI_LIST, TPL_BIAS_FLOAT, TPL_BIAS_HALF,
                                                          TPL_BIAS_BFLOAT, TPL_BIAS_INT)));

} // namespace NsDequantBias

#endif // __DEQUANT_BIAS_TILING_KEY_H__
