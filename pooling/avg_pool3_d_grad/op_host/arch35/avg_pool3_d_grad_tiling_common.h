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
 * \file avg_pool3_d_grad_tiling_common.h
 * \brief 3D average pooling backward shared constants and input info (arch35).
 *        Modeled on avg_pool_v2_grad_tiling_common.h.
 */

#ifndef OP_IMPL_AVG_POOL3_D_GRAD_TILING_COMMON_H_
#define OP_IMPL_AVG_POOL3_D_GRAD_TILING_COMMON_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "pooling/pool_3d_common/op_host/arch35/pool_3d_tiling_common.h"

namespace optiling {

// Constants reused from pool_3d_tiling_common.h:
//   DHW_DIMS=3, PAD_DIMS=6, NCDHW_DIMS=5, ONE_DIMS=1
//   D_DIM=0, H_DIM=1, W_DIM=2
//   FRONT_PAD_INDEX=0, BACKEND_PAD_INDEX=1, TOP_PAD_INDEX=2,
//   BOTTOM_PAD_INDEX=3, LEFT_PAD_INDEX=4, RIGHT_PAD_INDEX=5
//   DivRtn(x, y)

static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();
// Backward semantics: gradShape is the pooled (forward output) shape,
// outShape equals inputShape (the original input restored by orig_input_shape).
struct AvgPool3DGradInputInfo {
    int64_t batches;
    int64_t channels;
    std::array<int64_t, DHW_DIMS> inputShape;
    std::array<int64_t, DHW_DIMS> gradShape;
    std::array<int64_t, DHW_DIMS> outShape;
    std::array<int64_t, DHW_DIMS> kernelSize;
    std::array<int64_t, DHW_DIMS> stride;
    std::array<int64_t, PAD_DIMS> pad; // front/back/top/bottom/left/right
    bool ceilMode = false;
    bool countIncludePad = true;
    int64_t divisorOverride = 0;
    ge::Format inputFormat;
    int64_t dtypeSize = 0;
    int64_t isInt32Meet = 1;
    int64_t hasDivisor = 0;
};

} // namespace optiling

#endif // OP_IMPL_AVG_POOL3_D_GRAD_TILING_COMMON_H_
