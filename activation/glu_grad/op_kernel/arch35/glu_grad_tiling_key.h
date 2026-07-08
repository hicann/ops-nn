/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GLU_GRAD_TILING_KEY_H
#define GLU_GRAD_TILING_KEY_H

#include "tiling/template_argument.h"

#define TPL_SMALL_SHAPE 0
#define TPL_BIG_SHAPE 1

ASCENDC_TPL_ARGS_DECL(GLUGrad,
                      ASCENDC_TPL_DATATYPE_DECL(DTYPE_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(1)),
                      ASCENDC_TPL_UINT_DECL(SHAPE_MODE, 8, ASCENDC_TPL_UI_LIST, TPL_SMALL_SHAPE, TPL_BIG_SHAPE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DTYPE_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(SHAPE_MODE, ASCENDC_TPL_UI_LIST, TPL_SMALL_SHAPE, TPL_BIG_SHAPE)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DTYPE_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(SHAPE_MODE, ASCENDC_TPL_UI_LIST, TPL_SMALL_SHAPE, TPL_BIG_SHAPE)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DTYPE_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(SHAPE_MODE, ASCENDC_TPL_UI_LIST, TPL_SMALL_SHAPE, TPL_BIG_SHAPE)), );

#endif
