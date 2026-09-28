/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gemm_syrk_tiling_key.h
 * \brief GemmSyrk kernel template argument table. GemmSyrk has a single kernel
 * path (Blaze FmmWithScaleAdd syrk, MIX 1 AIC : 2 AIV), so the table carries
 * the kernel type selector, the transpose_x layout bit (a stored (..., k, m)
 * computes C = alpha * (a^T @ a) + beta * C via the DNExt GM view) and the
 * tiling struct binding.
 */

#pragma once

#include "ascendc/host_api/tiling/template_argument.h"
#include "gemm_syrk_tiling_data.h"

#define SYRK_KERNEL_BASIC 0
#define SYRK_TRANS_FALSE 0
#define SYRK_TRANS_TRUE 1

// 模板参数
ASCENDC_TPL_ARGS_DECL(GemmSyrk, // 算子OpType
                      ASCENDC_TPL_UINT_DECL(KERNEL_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SYRK_KERNEL_BASIC),
                      ASCENDC_TPL_BOOL_DECL(TRANS, SYRK_TRANS_FALSE, SYRK_TRANS_TRUE));

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(KERNEL_TYPE, ASCENDC_TPL_UI_LIST, SYRK_KERNEL_BASIC),
                                     ASCENDC_TPL_BOOL_SEL(TRANS, SYRK_TRANS_FALSE, SYRK_TRANS_TRUE),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(GemmSyrkTilingData)));
