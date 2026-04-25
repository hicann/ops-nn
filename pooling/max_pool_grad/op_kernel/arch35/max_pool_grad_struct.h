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
 * \file max_pool_grad_struct.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_STRUCT_H
#define MAX_POOL_GRAD_STRUCT_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

namespace PoolGradNameSpace {

#define TPL_NCHW_SMALL_KERNEL 1
#define TPL_NCHW_BIG_KERNEL 2
#define TPL_SIMT_KERNEL 3

#define TPL_NCHW_FORMAT 0
#define TPL_NHWC_FORMAT 1
#define TPL_INT64 0
#define TPL_INT32 1
#define TPL_NO_CHECK_RANGE 0
#define TPL_CHECK_RANGE 1

ASCENDC_TPL_ARGS_DECL(
    MaxPoolGrad,
    ASCENDC_TPL_UINT_DECL(
        kernelMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_NCHW_SMALL_KERNEL, TPL_NCHW_BIG_KERNEL, TPL_SIMT_KERNEL),
    ASCENDC_TPL_UINT_DECL(format, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_NCHW_FORMAT, TPL_NHWC_FORMAT),
    ASCENDC_TPL_UINT_DECL(indicesDtype, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
    ASCENDC_TPL_UINT_DECL(isCheckRange, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE, TPL_CHECK_RANGE));

ASCENDC_TPL_SEL(
    // SIMT kernel - format can be NCHW or NHWC
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(kernelMode, ASCENDC_TPL_UI_LIST, TPL_SIMT_KERNEL),
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NCHW_FORMAT, TPL_NHWC_FORMAT),
        ASCENDC_TPL_UINT_SEL(indicesDtype, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
        ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE),
        ASCENDC_TPL_TILING_STRUCT_SEL(MaxPoolGradWithArgmaxSimtTilingCommonData)),

    // NCHW BIG kernel - format must be NCHW
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(kernelMode, ASCENDC_TPL_UI_LIST, TPL_NCHW_BIG_KERNEL),
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NCHW_FORMAT),
        ASCENDC_TPL_UINT_SEL(indicesDtype, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
        ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE, TPL_CHECK_RANGE),
        ASCENDC_TPL_TILING_STRUCT_SEL(MaxPoolGradWithArgmaxNCHWTilingCommonData)),

    // NCHW SMALL kernel - format must be NCHW
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(kernelMode, ASCENDC_TPL_UI_LIST, TPL_NCHW_SMALL_KERNEL),
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NCHW_FORMAT),
        ASCENDC_TPL_UINT_SEL(indicesDtype, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
        ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE, TPL_CHECK_RANGE),
        ASCENDC_TPL_TILING_STRUCT_SEL(MaxPoolGradWithArgmaxNCHWTilingCommonData)));

} // namespace PoolGradNameSpace
#endif // MAX_POOL_GRAD_STRUCT_H