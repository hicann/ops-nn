/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_tiling_key.h
 * \brief
 */
#ifndef INPLACE_INDEX_FILL_TILING_KEY_H_
#define INPLACE_INDEX_FILL_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_MODE_SIMD 0
#define TPL_MODE_SIMT 1
#define TPL_MODE_ADDR_INT32 0
#define TPL_MODE_ADDR_INT64 1

namespace InplaceIndexFill {
ASCENDC_TPL_ARGS_DECL(
    InplaceIndexFill,
    ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 1, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMD, TPL_MODE_SIMT),
    ASCENDC_TPL_UINT_DECL(ADDR_MODE, 1, ASCENDC_TPL_UI_LIST, TPL_MODE_ADDR_INT32, TPL_MODE_ADDR_INT64)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMD),
        ASCENDC_TPL_UINT_SEL(ADDR_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_ADDR_INT64)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_SIMT),
        ASCENDC_TPL_UINT_SEL(ADDR_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_ADDR_INT32, TPL_MODE_ADDR_INT64)
    )
);

} // namespace InplaceIndexFill

#endif // INPLACE_INDEX_FILL_TILING_KEY_H_