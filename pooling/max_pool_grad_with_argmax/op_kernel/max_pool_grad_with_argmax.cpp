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
 * \file max_pool_grad_with_argmax.cpp
 * \brief max_pool_grad_with_argmax implied
 */

#include <cstdint>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "../max_pool_grad_with_argmax_common/max_pool_grad_with_argmax_nhwc_bigc_kernel_common.h"
#include "../max_pool_grad_with_argmax_common/max_pool_grad_with_argmax_nhwc_merge_hwc_kernel_common.h"
#include "../max_pool_grad_with_argmax_common/max_pool_grad_with_argmax_nhwc_merge_hwc_int64_kernel_common.h"
#include "../max_pool_grad_with_argmax_common/max_pool_grad_with_argmax_nhwc_merge_wc_kernel_common.h"
#include "../max_pool_grad_with_argmax_common/max_pool_grad_with_argmax_struct_common.h"
#include "../max_pool_grad_with_argmax_common/max_pool_grad_with_argmax_kernel_simt.h"

#define NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC 500
#define CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC 501
#define NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC_INT64 510
#define CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC_INT64 511
#define NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC 600
#define CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC 601
#define NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC_INT64 610
#define CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC_INT64 611

#define MPGA_SIMT_NCHW                   801
#define MPGA_SIMT_NHWC                   802
#define MPGA_SIMT_NCHW_INT64              803
#define MPGA_SIMT_NHWC_INT64               804

#define NO_CHECK_RANGE_TILING_KEY_NHWC_BIGC 700
#define CHECK_RANGE_TILING_KEY_NHWC_BIGC 701
#define NO_CHECK_RANGE_TILING_KEY_NHWC_BIGC_INT64 710
#define CHECK_RANGE_TILING_KEY_NHWC_BIGC_INT64 711

extern "C" __global__ __aicore__ void max_pool_grad_with_argmax(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
    if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCMergeHWCBase<DTYPE_X, DTYPE_ARGMAX, 0, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 501", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCMergeHWCBase<DTYPE_X, DTYPE_ARGMAX, 1, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 510", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCMergeHWCInt64Base<DTYPE_X, DTYPE_ARGMAX, 0, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NHWC_MERGE_HWC_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 511", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCMergeHWCInt64Base<DTYPE_X, DTYPE_ARGMAX, 1, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 600", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCMergeWCBase<DTYPE_X, DTYPE_ARGMAX, int32_t, 0, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 601", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCMergeWCBase<DTYPE_X, DTYPE_ARGMAX, int32_t, 1, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    }  else if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 610", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCMergeWCBase<DTYPE_X, DTYPE_ARGMAX, int64_t, 0, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 611", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCMergeWCBase<DTYPE_X, DTYPE_ARGMAX, int64_t, 1, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC_BIGC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 700", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCBigcBase<DTYPE_X, DTYPE_ARGMAX, int32_t, 0, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NHWC_BIGC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 701", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCBigcBase<DTYPE_X, DTYPE_ARGMAX, int32_t, 1, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    }    else if (TILING_KEY_IS(NO_CHECK_RANGE_TILING_KEY_NHWC_BIGC_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 710", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCBigcBase<DTYPE_X, DTYPE_ARGMAX, int64_t, 0, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(CHECK_RANGE_TILING_KEY_NHWC_BIGC_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 711", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxKernelNHWCBigcBase<DTYPE_X, DTYPE_ARGMAX, int64_t, 1, VER_NORMAL> op;
        op.Init(x, grad, argmax, y, pipeBase, tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(MPGA_SIMT_NCHW)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 801", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NCHW_FORMAT, int32_t> op(&pipeBase, &tilingDataIn);
        op.Init(x, grad, argmax, y);
        op.Process();
    } else if (TILING_KEY_IS(MPGA_SIMT_NHWC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 802", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NHWC_FORMAT, int32_t> op(&pipeBase, &tilingDataIn);
        op.Init(x, grad, argmax, y);
        op.Process();
    } else if (TILING_KEY_IS(MPGA_SIMT_NCHW_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 803", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NCHW_FORMAT, int64_t> op(&pipeBase, &tilingDataIn);
        op.Init(x, grad, argmax, y);
        op.Process();
    } else if (TILING_KEY_IS(MPGA_SIMT_NHWC_INT64)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 804", MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolGradWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NHWC_FORMAT, int64_t> op(&pipeBase, &tilingDataIn);
        op.Init(x, grad, argmax, y);
        op.Process();
    }
}