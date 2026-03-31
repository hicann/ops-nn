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
 * \file adaptive_avg_pool3d_grad_struct.h
 * \brief adaptive_avg_pool3d_grad tiling struct
 */
#ifndef ADAPTIVE_AVG_POOL3D_GRAD_STRUCT_H_
#define ADAPTIVE_AVG_POOL3D_GRAD_STRUCT_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

namespace AdaptiveAvgPool3dGradOp {

#define TPL_SMALL_KERNEL 1
#define TPL_BIG_KERNEL 2
#define TPL_SIMT_KERNEL 3

#define TPL_INT32 1
#define TPL_INT64 2

ASCENDC_TPL_ARGS_DECL(AdaptiveAvgPool3dGrad,
    ASCENDC_TPL_UINT_DECL(
        TEMPLATE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_SMALL_KERNEL, TPL_BIG_KERNEL, TPL_SIMT_KERNEL),
    ASCENDC_TPL_DTYPE_DECL(INDEX_DTYPE, TPL_INT32, TPL_INT64),
    ASCENDC_TPL_BOOL_DECL(IS_CHANNEL_LAST, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_BIG_KERNEL),
        ASCENDC_TPL_DTYPE_SEL(INDEX_DTYPE, TPL_INT32, TPL_INT64),
        ASCENDC_TPL_BOOL_SEL(IS_CHANNEL_LAST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_SMALL_KERNEL),
        ASCENDC_TPL_DTYPE_SEL(INDEX_DTYPE, TPL_INT32, TPL_INT64),
        ASCENDC_TPL_BOOL_SEL(IS_CHANNEL_LAST, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_SIMT_KERNEL),
        ASCENDC_TPL_DTYPE_SEL(INDEX_DTYPE, TPL_INT32, TPL_INT64),
        ASCENDC_TPL_BOOL_SEL(IS_CHANNEL_LAST, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(AdaptiveAvgPool3dGradTilingDataV35)
    )
);

// 公共基础 tiling data
struct AdaptiveAvgPool3dGradTilingDataV35 {
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t dInDim = 0;
    int64_t hInDim = 0;
    int64_t wInDim = 0;
    int64_t dOutDim = 0;
    int64_t hOutDim = 0;
    int64_t wOutDim = 0;
};

// 小 kernel tiling data
struct AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35 {
    int64_t dInput = 0;
    int64_t hInput = 0;
    int64_t wInput = 0;
    int64_t dOutput = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t highAxisInner = 0;
    int64_t highAxisTail = 0;
    int64_t highAxisOuter = 0;
    int64_t dOutputInner = 0;
    int64_t dOutputTail = 0;
    int64_t dOutputOuter = 0;
    int64_t hOutputInner = 0;
    int64_t hOutputTail = 0;
    int64_t hOutputOuter = 0;
    int64_t wOutputInner = 0;
    int64_t wOutputTail = 0;
    int64_t wOutputOuter = 0;
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    int64_t usedCoreNum = 0;
    int64_t outputBufferSize = 0;
    int64_t gradInputBufferSize = 0;
    int64_t inputQueBufferSize = 0;
    int64_t transQueBufferSize = 0;
    int64_t transOutQueBufferSize = 0;
};

struct AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35 {
    int64_t dInput = 0;
    int64_t hInput = 0;
    int64_t wInput = 0;
    int64_t dOutput = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t highAxisInner = 0;
    int64_t highAxisTail = 0;
    int64_t highAxisOuter = 0;
    int64_t dOutputInner = 0;
    int64_t dOutputTail = 0;
    int64_t dOutputOuter = 0;
    int64_t hOutputInner = 0;
    int64_t hOutputTail = 0;
    int64_t hOutputOuter = 0;
    int64_t wOutputInner = 0;
    int64_t wOutputTail = 0;
    int64_t wOutputOuter = 0;
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    int64_t usedCoreNum = 0;
    int64_t outputBufferSize = 0;
    int64_t gradInputBufferSize = 0;
};

}  // namespace AdaptiveAvgPool3dGradOp

#endif  // ADAPTIVE_AVG_POOL3D_GRAD_STRUCT_H_
