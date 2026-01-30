/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool_v2_grad_tiling_common.h
 * \brief
 */

#ifndef OP_IMPL_AVG_POOL_V2_GRAD_TILING_COMMON_H_
#define OP_IMPL_AVG_POOL_V2_GRAD_TILING_COMMON_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling
{
const int32_t HW_DIMS = 2;
const int32_t PAD_DIMS = 4;

const int32_t ONE_DIMS = 1;
const int32_t CHW_DIMS = 3;
const int32_t NCHW_DIMS = 4;

const uint32_t H_DIM = 0;
const uint32_t W_DIM = 1;

const uint32_t TOP_PAD_INDEX = 0;
const uint32_t BOTTOM_PAD_INDEX = 1;
const uint32_t LEFT_PAD_INDEX = 2;
const uint32_t RIGHT_PAD_INDEX = 3;
static const gert::Shape g_vec_1_shape = {1};

struct AvgPoolV2GradInputInfo {
    int64_t batches;
    int64_t channels;
    std::array<int64_t, HW_DIMS> inputShape;
    std::array<int64_t, HW_DIMS> gradShape;
    std::array<int64_t, HW_DIMS> outShape;
    std::array<int64_t, HW_DIMS> kernelSize;
    std::array<int64_t, HW_DIMS> stride;
    std::array<int64_t, PAD_DIMS> pad;
    bool ceilMode = false; 
    bool countIncludePad = true;
    bool globalPooling = false;
    int64_t divisorOverride = 0;
    ge::Format inputFormat;
    int64_t dtypeSize = 0;
    int64_t isInt32Meet = 1;
    int64_t hasDivisor = 0;
};

static const gert::Shape& EnsureNotScalar(const gert::Shape& inShape) {
 if (inShape.IsScalar()) {
   return g_vec_1_shape;
 }
 return inShape;
}
} // namespace optiling

#endif