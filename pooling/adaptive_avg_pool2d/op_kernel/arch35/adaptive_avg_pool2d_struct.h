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
 * \file adaptive_avg_pool2d_struct.h
 * \brief adaptive_avg_pool2d_struct
 */
#ifndef ADAPTIVE_AVG_POOL2D_STRUCT_H_
#define ADAPTIVE_AVG_POOL2D_STRUCT_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

namespace AdaptiveAvgPool2dOp {

#define TPL_INT32_UINT32 0
#define TPL_INT64_UINT64 1
#define TPL_SMALL_KERNEL 0
#define TPL_BIG_KERNEL 1
#define TPL_SIMT_KERNEL 2

ASCENDC_TPL_ARGS_DECL(AdaptiveAvgPool2d,
    ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 2, ASCENDC_TPL_UI_LIST, TPL_SMALL_KERNEL, TPL_BIG_KERNEL, TPL_SIMT_KERNEL),
    ASCENDC_TPL_UINT_DECL(DTYPE_MODE, 3, ASCENDC_TPL_UI_LIST, TPL_INT32_UINT32, TPL_INT64_UINT64),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_SIMT_KERNEL),
        ASCENDC_TPL_UINT_SEL(DTYPE_MODE, ASCENDC_TPL_UI_LIST, TPL_INT32_UINT32, TPL_INT64_UINT64),
        ASCENDC_TPL_TILING_STRUCT_SEL(AdaptivePool2DSimtTilingData),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_BIG_KERNEL),
        ASCENDC_TPL_UINT_SEL(DTYPE_MODE, ASCENDC_TPL_UI_LIST, TPL_INT32_UINT32),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_SMALL_KERNEL),
        ASCENDC_TPL_UINT_SEL(DTYPE_MODE, ASCENDC_TPL_UI_LIST, TPL_INT32_UINT32, TPL_INT64_UINT64),
    )
);

struct AdaptivePool2DSimtTilingData {
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t hInDim = 0;
    int64_t wInDim = 0;
    int64_t hOutDim = 0;
    int64_t wOutDim = 0;
};

} // namespace AdaptiveAvgPool2dOp
#endif  // ADAPTIVE_AVG_POOL2D_STRUCT_H_
