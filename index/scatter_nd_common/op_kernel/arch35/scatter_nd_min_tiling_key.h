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
 * \file scatter_nd_min_tiling_key.h
 * \brief
 */
#ifndef SCATTER_ND_MIN_TILING_KEY_H_
#define SCATTER_ND_MIN_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_MODE_ADDR_INT32 0
#define TPL_MODE_ADDR_INT64 1

#define CAST_0 0
#define CAST_1 1
#define CAST_2 2
#define CAST_3 3
#define CAST_4 4
#define CAST_5 5

#define TPL_MODE_TEMPLATE_SIMD_SORT 2
#define TPL_MODE_TEMPLATE_SIMT_SORT 5
#define TPL_MODE_TEMPLATE_SIMT 8

namespace ScatterNdCommon {

ASCENDC_TPL_ARGS_DECL(ScatterNdMin,
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 2, ASCENDC_TPL_UI_LIST, TPL_MODE_TEMPLATE_SIMD_SORT,
                                            TPL_MODE_TEMPLATE_SIMT_SORT, TPL_MODE_TEMPLATE_SIMT),
                      ASCENDC_TPL_UINT_DECL(CAST_MODE, 3, ASCENDC_TPL_UI_LIST, CAST_0, CAST_1, CAST_2, CAST_3, CAST_4,
                                            CAST_5),
                      ASCENDC_TPL_UINT_DECL(ADDR_MODE, 1, ASCENDC_TPL_UI_LIST, TPL_MODE_ADDR_INT32,
                                            TPL_MODE_ADDR_INT64));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_TEMPLATE_SIMD_SORT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, CAST_0, CAST_1, CAST_2, CAST_3, CAST_4,
                                              CAST_5),
                         ASCENDC_TPL_UINT_SEL(ADDR_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_ADDR_INT32, TPL_MODE_ADDR_INT64),
                         ASCENDC_TPL_TILING_STRUCT_SEL(ScatterNdCommonSimdSortTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_TEMPLATE_SIMT_SORT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, CAST_0, CAST_1, CAST_2, CAST_3, CAST_4,
                                              CAST_5),
                         ASCENDC_TPL_UINT_SEL(ADDR_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_ADDR_INT32, TPL_MODE_ADDR_INT64),
                         ASCENDC_TPL_TILING_STRUCT_SEL(ScatterNdCommonSimtSortTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_TEMPLATE_SIMT),
                         ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, CAST_0),
                         ASCENDC_TPL_UINT_SEL(ADDR_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_ADDR_INT32, TPL_MODE_ADDR_INT64),
                         ASCENDC_TPL_TILING_STRUCT_SEL(ScatterNdCommonSimtTilingData)));

} // namespace ScatterNdCommon

#endif // SCATTER_ND_MIN_TILING_KEY_H_