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
 * \file pool_grad_struct_common.h
 * \brief tiling base data
 */

#ifndef POOL_GRAD_STRUCT_H
#define POOL_GRAD_STRUCT_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

namespace PoolGradNameSpace {
#define TPL_INT32 1
#define TPL_INT64 2

ASCENDC_TPL_ARGS_DECL(
    MaxPoolGrad, ASCENDC_TPL_DTYPE_DECL(INDEX_DTYPE, TPL_INT32, TPL_INT64), ASCENDC_TPL_BOOL_DECL(IS_SIMT, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_CHANNEL_LAST, 0, 1), ASCENDC_TPL_BOOL_DECL(IS_BIGGER_THAN_INT32, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_BIG_KERNEL, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_DTYPE_SEL(INDEX_DTYPE, TPL_INT32),
        ASCENDC_TPL_BOOL_SEL(IS_SIMT, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_CHANNEL_LAST, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_BIGGER_THAN_INT32, 0, 1), ASCENDC_TPL_BOOL_DECL(IS_BIG_KERNEL, 0, 1)),

    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_DTYPE_SEL(INDEX_DTYPE, TPL_INT64),
        ASCENDC_TPL_BOOL_SEL(IS_SIMT, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_CHANNEL_LAST, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_BIGGER_THAN_INT32, 0, 1), ASCENDC_TPL_BOOL_DECL(IS_BIG_KERNEL, 0, 1)));

} // namespace PoolGradNameSpace
#endif // POOL_GRAD_STRUCT_H