/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file max_pool3d_grad_with_argmax_struct.h
 * \brief max_pool3d_grad_with_argmax_struct
 */
#ifndef MAX_POOL3D_GRAD_WITH_ARGMAX_STRUCT_H_
#define MAX_POOL3D_GRAD_WITH_ARGMAX_STRUCT_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

namespace MaxPool3DGradWithArgmaxOp {

#define TPL_INT32 1
#define TPL_INT64 2

ASCENDC_TPL_ARGS_DECL(MaxPool3DGradWithArgmax,
    ASCENDC_TPL_DTYPE_DECL(INDEX_DTYPE, TPL_INT32, TPL_INT64),
    ASCENDC_TPL_BOOL_DECL(IS_SIMT, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_CHANNEL_LAST, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_CHECK_RANGE, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DTYPE_SEL(INDEX_DTYPE, TPL_INT32),
        ASCENDC_TPL_BOOL_SEL(IS_SIMT, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_CHANNEL_LAST, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_CHECK_RANGE, 0, 1)
    ),

    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DTYPE_SEL(INDEX_DTYPE, TPL_INT64),
        ASCENDC_TPL_BOOL_SEL(IS_SIMT, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_CHANNEL_LAST, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_CHECK_RANGE, 0, 1)
    )
);

struct MaxPool3DGradWithArgmaxTilingDataV35 {
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t dInDim = 0;
    int64_t hInDim = 0;
    int64_t wInDim = 0;
    int64_t dOutDim = 0;
    int64_t hOutDim = 0;
    int64_t wOutDim = 0;
    int64_t kSizeD = 0;
    int64_t kSizeH = 0;
    int64_t kSizeW = 0;
    int64_t strideD = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t padD = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t dilationD = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
};

struct MaxPool3DGradWithArgmaxNCDHWTilingData {
    int64_t dArgmax = 0;
    int64_t hArgmax = 0;
    int64_t wArgmax = 0;
    int64_t dOutput = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t dKernel = 0;
    int64_t hKernel = 0;
    int64_t wKernel = 0;
    int64_t dStride = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    int64_t padD = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t dilationD = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
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
    int64_t gradBufferSize = 0;
    int64_t argmaxBufferSize = 0;
    int64_t dProBatchSize = 0;
    int64_t hProBatchSize = 0;
    int64_t wProBatchSize = 0;
};

} // namespace MaxPool3DGradWithArgmaxOp
#endif  // MAX_POOL3D_GRAD_WITH_ARGMAX_STRUCT_H_