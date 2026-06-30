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
 * \file inplace_scatter_add_tiling_key.h
 * \brief tiling key declaration for inplace_scatter_add
 */

#ifndef INPLACE_SCATTER_ADD_TILING_KEY_H
#define INPLACE_SCATTER_ADD_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// 两种模式：Default（innerSize > 1，需坐标分解）和 1D（innerSize == 1，直接索引）
#define SCATTER_ADD_MODE_DEFAULT 0
#define SCATTER_ADD_MODE_1D      1

ASCENDC_TPL_ARGS_DECL(
    InplaceScatterAdd,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, SCATTER_ADD_MODE_DEFAULT, SCATTER_ADD_MODE_1D)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SCATTER_ADD_MODE_DEFAULT, SCATTER_ADD_MODE_1D),
        ASCENDC_TPL_TILING_STRUCT_SEL(InplaceScatterAddTilingData)
    )
);

#endif // INPLACE_SCATTER_ADD_TILING_KEY_H
