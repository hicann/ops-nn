/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_batch_matmul_inplace_add_tiling_key.h
 * \brief
 */
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

namespace QuantBatchMatmulInplaceAddArch35TilingKey {

#if defined(ASC_DEVKIT_MAJOR) && defined(ASC_DEVKIT_MINOR) && ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0 &&       \
    defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_X2_SCALE) && defined(DT_FLOAT8_E4M3FN) && \
    defined(DT_FLOAT8_E5M2) && defined(DT_FLOAT8_E8M0)
#define SUPPORT_MX_WITHOUT_BATCH_TILING_KEY                                    \
    ((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X1 == DT_FLOAT8_E5M2) && \
     (ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X2 == DT_FLOAT8_E5M2) && ORIG_DTYPE_X2_SCALE == DT_FLOAT8_E8M0)
#else
#define SUPPORT_MX_WITHOUT_BATCH_TILING_KEY false
#endif

// Kernel Type - Basic API path
#define TPL_NO_VEC_EPILOGUE_WITH_MMAPI 0
#define TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI 1
#define TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH 2
#define TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH 3

ASCENDC_TPL_ARGS_DECL(
    QuantBatchMatmulInplaceAdd, ASCENDC_TPL_UINT_DECL(ATRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(BTRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(KERNELTYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_NO_VEC_EPILOGUE_WITH_MMAPI,
                          TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI, TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH,
                          TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH));

ASCENDC_TPL_SEL(
#if ((!defined(__CCE_AICORE__)) || (SUPPORT_MX_WITHOUT_BATCH_TILING_KEY))
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
                         ASCENDC_TPL_UINT_SEL(ATRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(BTRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(KERNELTYPE, ASCENDC_TPL_UI_LIST,
                                              TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH,
                                              TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH)),
#endif
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
                         ASCENDC_TPL_UINT_SEL(ATRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(BTRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(KERNELTYPE, ASCENDC_TPL_UI_LIST, TPL_NO_VEC_EPILOGUE_WITH_MMAPI,
                                              TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI)));
} // namespace QuantBatchMatmulInplaceAddArch35TilingKey
