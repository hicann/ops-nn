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
 * \file bucketize_v2_tiling_key.h
 * \brief
 */
#ifndef BUCKETIZE_V2_TILING_KEY_H_
#define BUCKETIZE_V2_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "bucketize_v2_struct.h"

#define TPL_MODE_NO_RIGHT 0
#define TPL_MODE_RIGHT 1
#define TPL_MODE_FULL_LOAD  1
#define TPL_MODE_TEMPLATE_CASCADE 2

namespace BucketizeV2 {
ASCENDC_TPL_ARGS_DECL(
    BucketizeV2,
    ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 2, ASCENDC_TPL_UI_LIST, TPL_MODE_TEMPLATE_CASCADE, TPL_MODE_FULL_LOAD),
    ASCENDC_TPL_BOOL_DECL(BOUNDARY_MODE, TPL_MODE_NO_RIGHT, TPL_MODE_RIGHT)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_FULL_LOAD),
        ASCENDC_TPL_BOOL_SEL(BOUNDARY_MODE, TPL_MODE_NO_RIGHT),
        ASCENDC_TPL_TILING_STRUCT_SEL(BucketizeV2FullLoadTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_FULL_LOAD),
        ASCENDC_TPL_BOOL_SEL(BOUNDARY_MODE, TPL_MODE_RIGHT),
        ASCENDC_TPL_TILING_STRUCT_SEL(BucketizeV2FullLoadTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_TEMPLATE_CASCADE),
        ASCENDC_TPL_BOOL_SEL(BOUNDARY_MODE, TPL_MODE_NO_RIGHT),
        ASCENDC_TPL_TILING_STRUCT_SEL(BucketizeV2CascadeTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_MODE_TEMPLATE_CASCADE),
        ASCENDC_TPL_BOOL_SEL(BOUNDARY_MODE, TPL_MODE_RIGHT),
        ASCENDC_TPL_TILING_STRUCT_SEL(BucketizeV2CascadeTilingData)
    )
);

} // namespace BucketizeV2

#endif // BUCKETIZE_V2_TILING_KEY_H_