/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_max_pool3d_grad_struct.h
 * \brief adaptive_max_pool3d_grad tiling struct for arch35
 */
#ifndef ADAPTIVE_MAX_POOL3D_GRAD_STRUCT_H_
#define ADAPTIVE_MAX_POOL3D_GRAD_STRUCT_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

namespace AdaptiveMaxPool3dGradOp {

#define TPL_SIMT_KERNEL 2

#define TPL_INT32 1
#define TPL_INT64 2

ASCENDC_TPL_ARGS_DECL(AdaptiveMaxPool3DGrad,
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_SIMT_KERNEL),
                      ASCENDC_TPL_DTYPE_DECL(INDEX_DTYPE, TPL_INT32, TPL_INT64));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, TPL_SIMT_KERNEL),
                                     ASCENDC_TPL_DTYPE_SEL(INDEX_DTYPE, TPL_INT32, TPL_INT64),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(AdaptiveMaxPool3dGradTilingDataV35)));

struct AdaptiveMaxPool3dGradTilingDataV35 {
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t dInDim = 0;
    int64_t hInDim = 0;
    int64_t wInDim = 0;
    int64_t dOutDim = 0;
    int64_t hOutDim = 0;
    int64_t wOutDim = 0;
    int64_t isOverLap = 0;
    int64_t deterministicFlag = 0;
};

} // namespace AdaptiveMaxPool3dGradOp

#endif // ADAPTIVE_MAX_POOL3D_GRAD_STRUCT_H_
