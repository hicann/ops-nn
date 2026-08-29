/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_tiling_key.h
 * \brief Tiling key template args for 3D average pooling backward (arch35).
 */

#ifndef AVG_POOL3_D_GRAD_TILING_KEY_H_
#define AVG_POOL3_D_GRAD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace AvgPool3DGrad {
#define TPL_NCDHW_KERNEL 1
#define TPL_NDHWC_KERNEL 2
#define TPL_SIMT_KERNEL 3
#define TPL_KSIZE_ONE_KERNEL 4
#define TPL_NCDHW_FORMAT 0
#define TPL_NDHWC_FORMAT 1
#define TPL_INT64 0
#define TPL_INT32 1
#define TPL_NO_PAD 0
#define TPL_PAD 1
#define TPL_NO_CHECK_RANGE 0
#define TPL_CHECK_RANGE 1
#define TPL_NO_COUNT_PAD 0
#define TPL_COUNT_PAD 1
#define TPL_NO_DIVISOR 0
#define TPL_HAS_DIVISOR 1

ASCENDC_TPL_ARGS_DECL(
    AvgPool3DGrad,
    ASCENDC_TPL_UINT_DECL(schMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_NCDHW_KERNEL, TPL_NDHWC_KERNEL,
                          TPL_SIMT_KERNEL, TPL_KSIZE_ONE_KERNEL),
    ASCENDC_TPL_UINT_DECL(format, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_NCDHW_FORMAT, TPL_NDHWC_FORMAT),
    ASCENDC_TPL_UINT_DECL(isInt32Meet, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
    ASCENDC_TPL_UINT_DECL(isPad, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_NO_PAD, TPL_PAD),
    ASCENDC_TPL_UINT_DECL(isCheckRange, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE, TPL_CHECK_RANGE),
    ASCENDC_TPL_UINT_DECL(countIncludePad, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_NO_COUNT_PAD, TPL_COUNT_PAD),
    ASCENDC_TPL_UINT_DECL(hasDivisor, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_NO_DIVISOR, TPL_HAS_DIVISOR));

ASCENDC_TPL_SEL(
    // ksize_one kernel: kD=kH=kW=1, not use isPad/isCheckRange/countIncludePad/hasDivisor/isInt32Meet
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_KSIZE_ONE_KERNEL),
                         ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NCDHW_FORMAT, TPL_NDHWC_FORMAT),
                         ASCENDC_TPL_UINT_SEL(isInt32Meet, ASCENDC_TPL_UI_LIST, TPL_INT64),
                         ASCENDC_TPL_UINT_SEL(isPad, ASCENDC_TPL_UI_LIST, TPL_NO_PAD),
                         ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE),
                         ASCENDC_TPL_UINT_SEL(countIncludePad, ASCENDC_TPL_UI_LIST, TPL_NO_COUNT_PAD),
                         ASCENDC_TPL_UINT_SEL(hasDivisor, ASCENDC_TPL_UI_LIST, TPL_NO_DIVISOR, TPL_HAS_DIVISOR),
                         ASCENDC_TPL_TILING_STRUCT_SEL(AvgPool3DGradKsizeOneTilingData)),

    // simt kernel not use isPad/isCheckRange
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_SIMT_KERNEL),
                         ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NCDHW_FORMAT, TPL_NDHWC_FORMAT),
                         ASCENDC_TPL_UINT_SEL(isInt32Meet, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
                         ASCENDC_TPL_UINT_SEL(isPad, ASCENDC_TPL_UI_LIST, TPL_NO_PAD),
                         ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE),
                         ASCENDC_TPL_UINT_SEL(countIncludePad, ASCENDC_TPL_UI_LIST, TPL_NO_COUNT_PAD, TPL_COUNT_PAD),
                         ASCENDC_TPL_UINT_SEL(hasDivisor, ASCENDC_TPL_UI_LIST, TPL_NO_DIVISOR, TPL_HAS_DIVISOR),
                         ASCENDC_TPL_TILING_STRUCT_SEL(AvgPool3DGradSimtTilingData)),

    // ncdhw kernel format must be ncdhw, not use isPad
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_NCDHW_KERNEL),
                         ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NCDHW_FORMAT),
                         ASCENDC_TPL_UINT_SEL(isInt32Meet, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
                         ASCENDC_TPL_UINT_SEL(isPad, ASCENDC_TPL_UI_LIST, TPL_NO_PAD),
                         ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE, TPL_CHECK_RANGE),
                         ASCENDC_TPL_UINT_SEL(countIncludePad, ASCENDC_TPL_UI_LIST, TPL_NO_COUNT_PAD, TPL_COUNT_PAD),
                         ASCENDC_TPL_UINT_SEL(hasDivisor, ASCENDC_TPL_UI_LIST, TPL_NO_DIVISOR, TPL_HAS_DIVISOR),
                         ASCENDC_TPL_TILING_STRUCT_SEL(AvgPool3DGradNCDHWTilingData)),

    // ndhwc kernel format must be ndhwc, not use isPad
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_NDHWC_KERNEL),
                         ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NDHWC_FORMAT),
                         ASCENDC_TPL_UINT_SEL(isInt32Meet, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
                         ASCENDC_TPL_UINT_SEL(isPad, ASCENDC_TPL_UI_LIST, TPL_NO_PAD),
                         ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE, TPL_CHECK_RANGE),
                         ASCENDC_TPL_UINT_SEL(countIncludePad, ASCENDC_TPL_UI_LIST, TPL_NO_COUNT_PAD, TPL_COUNT_PAD),
                         ASCENDC_TPL_UINT_SEL(hasDivisor, ASCENDC_TPL_UI_LIST, TPL_NO_DIVISOR, TPL_HAS_DIVISOR),
                         ASCENDC_TPL_TILING_STRUCT_SEL(AvgPool3DGradNDHWCTilingData)));
} // namespace AvgPool3DGrad
#endif // AVG_POOL3_D_GRAD_TILING_KEY_H_
