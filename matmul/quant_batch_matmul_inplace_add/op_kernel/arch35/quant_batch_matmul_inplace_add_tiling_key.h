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
#ifndef QBMMIA_ARCH35_TILING_KEY_H
#define QBMMIA_ARCH35_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace QuantBatchMatmulInplaceAddArch35TilingKey {

// Kernel Type
#define TPL_NO_VEC_EPILOGUE_WITH_MMAPI 0
#define TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI 1

ASCENDC_TPL_ARGS_DECL(
    QuantBatchMatmulInplaceAdd, ASCENDC_TPL_UINT_DECL(ATRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(BTRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(
        KERNELTYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_NO_VEC_EPILOGUE_WITH_MMAPI,
        TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY), ASCENDC_TPL_UINT_SEL(ATRANS, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_SEL(BTRANS, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_SEL(
        KERNELTYPE, ASCENDC_TPL_UI_LIST, TPL_NO_VEC_EPILOGUE_WITH_MMAPI,
        TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI)));
} // namespace QuantBatchMatmulInplaceAddArch35TilingKey
#endif