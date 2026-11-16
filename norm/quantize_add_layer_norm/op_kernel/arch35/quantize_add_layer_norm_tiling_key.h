/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_TILING_KEY_H_
#define QUANTIZE_ADD_LAYER_NORM_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quantize_add_layer_norm_tiling_data.h"

#define QALN_QUANT_MODE_MUL 0
#define QALN_QUANT_MODE_DIV 1
#define QALN_QUANT_MODE_PER_TENSOR 2

ASCENDC_TPL_ARGS_DECL(QuantizeAddLayerNorm, ASCENDC_TPL_BOOL_DECL(isWelford, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(isBiasBroadcast, 0, 1),
                      ASCENDC_TPL_UINT_DECL(quantMode, 2, ASCENDC_TPL_UI_LIST, QALN_QUANT_MODE_MUL, QALN_QUANT_MODE_DIV,
                                            QALN_QUANT_MODE_PER_TENSOR),
                      ASCENDC_TPL_BOOL_DECL(isMembase, 0, 1), ASCENDC_TPL_BOOL_DECL(additionalOutput, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isWelford, 0, 1), ASCENDC_TPL_BOOL_SEL(isBiasBroadcast, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(quantMode, ASCENDC_TPL_UI_LIST, QALN_QUANT_MODE_MUL,
                                                          QALN_QUANT_MODE_DIV, QALN_QUANT_MODE_PER_TENSOR),
                                     ASCENDC_TPL_BOOL_SEL(isMembase, 0), ASCENDC_TPL_BOOL_SEL(additionalOutput, 0),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(QuantizeAddLayerNormRegbaseTilingData)));

#endif // QUANTIZE_ADD_LAYER_NORM_TILING_KEY_H_
