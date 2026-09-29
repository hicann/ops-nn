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
 * \file quant_matmul_activation_quant_tiling_key.h
 * \brief Compile-time transpose, batch and kernel dispatch keys. Weight layout follows FORMAT_X2.
 */
#pragma once

#if defined(__CCE_AICORE__)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#endif

#include "ascendc/host_api/tiling/template_argument.h"

namespace QuantMatmulActivationQuantArch35TilingKey {

// Batch broadcasting is omitted when both inputs contain a single matrix.
#define TPL_WITH_BATCH 0
#define TPL_WITHOUT_BATCH 1

// Epilogue and A full-load policy.
#define TPL_GELU_NO_FULLLOAD 0
#define TPL_GELU_FULLLOAD 1
#define TPL_SWIGLU_NO_FULLLOAD 2
#define TPL_SWIGLU_FULLLOAD 3

// Select compilable SwiGLU specializations; input validation belongs to host tiling.
#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_Y) && defined(DT_FLOAT8_E4M3FN) && \
    defined(DT_FLOAT8_E5M2) && defined(FORMAT_X2) && defined(FORMAT_ND)
#if defined(FORMAT_FRACTAL_NZ)
#define QMMAQ_SWIGLU_FORMAT_SUPPORTED \
    (FORMAT_X2 == FORMAT_ND || (FORMAT_X2 == FORMAT_FRACTAL_NZ && ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN))
#else
#define QMMAQ_SWIGLU_FORMAT_SUPPORTED (FORMAT_X2 == FORMAT_ND)
#endif
#define QMMAQ_SWIGLU_TEMPLATE_SUPPORTED                                        \
    ((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X1 == DT_FLOAT8_E5M2) && \
     (ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X2 == DT_FLOAT8_E5M2) && \
     (ORIG_DTYPE_Y == DT_FLOAT8_E4M3FN || ORIG_DTYPE_Y == DT_FLOAT8_E5M2) && QMMAQ_SWIGLU_FORMAT_SUPPORTED)
#else
#define QMMAQ_SWIGLU_TEMPLATE_SUPPORTED 0
#endif

ASCENDC_TPL_ARGS_DECL(QuantMatmulActivationQuant,
                      ASCENDC_TPL_UINT_DECL(ATRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(BTRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(BATCHMODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TPL_WITH_BATCH,
                                            TPL_WITHOUT_BATCH),
                      ASCENDC_TPL_UINT_DECL(KERNELTYPE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TPL_GELU_NO_FULLLOAD,
                                            TPL_GELU_FULLLOAD, TPL_SWIGLU_NO_FULLLOAD, TPL_SWIGLU_FULLLOAD));
ASCENDC_TPL_SEL(
#if !defined(__CCE_AICORE__) || QMMAQ_SWIGLU_TEMPLATE_SUPPORTED
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_UINT_SEL(ATRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(BTRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(BATCHMODE, ASCENDC_TPL_UI_LIST, TPL_WITH_BATCH, TPL_WITHOUT_BATCH),
                         ASCENDC_TPL_UINT_SEL(KERNELTYPE, ASCENDC_TPL_UI_LIST, TPL_SWIGLU_NO_FULLLOAD,
                                              TPL_SWIGLU_FULLLOAD)),
#endif
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_UINT_SEL(ATRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(BTRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(BATCHMODE, ASCENDC_TPL_UI_LIST, TPL_WITH_BATCH, TPL_WITHOUT_BATCH),
                         ASCENDC_TPL_UINT_SEL(KERNELTYPE, ASCENDC_TPL_UI_LIST, TPL_GELU_NO_FULLLOAD,
                                              TPL_GELU_FULLLOAD)));

#undef QMMAQ_SWIGLU_FORMAT_SUPPORTED
#undef QMMAQ_SWIGLU_TEMPLATE_SUPPORTED
} // namespace QuantMatmulActivationQuantArch35TilingKey
