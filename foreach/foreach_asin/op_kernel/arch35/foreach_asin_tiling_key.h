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
 * \file foreach_asin_tiling_key.h
 * \brief tiling key declare for foreach_asin
 */

#ifndef FOREACH_ASIN_TILING_KEY_H
#define FOREACH_ASIN_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

/* dtype 模板参数取值 */
#define FOREACH_ASIN_TPL_DTYPE_FP32 0
#define FOREACH_ASIN_TPL_DTYPE_FP16 1
#define FOREACH_ASIN_TPL_DTYPE_BF16 2

/* 模板参数声明 */
ASCENDC_TPL_ARGS_DECL(ForeachAsin, ASCENDC_TPL_UINT_DECL(dtype, 2, ASCENDC_TPL_UI_LIST, FOREACH_ASIN_TPL_DTYPE_FP32,
                                                         FOREACH_ASIN_TPL_DTYPE_FP16, FOREACH_ASIN_TPL_DTYPE_BF16));

/* 模板参数组合 */
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(dtype, ASCENDC_TPL_UI_LIST, FOREACH_ASIN_TPL_DTYPE_FP32,
                                                          FOREACH_ASIN_TPL_DTYPE_FP16, FOREACH_ASIN_TPL_DTYPE_BF16)));

#endif // FOREACH_ASIN_TILING_KEY_H
