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
 * \file avg_pool_v2_grad_tiling_common.h
 * \brief
 */

#ifndef OP_IMPL_AVG_POOL_V2_GRAD_TILING_COMMON_H_
#define OP_IMPL_AVG_POOL_V2_GRAD_TILING_COMMON_H_

#include <algorithm>
#include <array>
#include <string>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "log/log.h"

namespace optiling {
const int32_t HW_DIMS = 2;
const int32_t PAD_DIMS = 4;

const int32_t ZERO_DIMS = 0;
const int32_t ONE_DIMS = 1;
const int32_t CHW_DIMS = 3;
const int32_t NCHW_DIMS = 4;

const int32_t AVG_POOL_GRAD_DIM_ZERO = 0;
const int32_t AVG_POOL_GRAD_DIM_ONE = 1;
const int32_t AVG_POOL_GRAD_DIM_TWO = 2;
const int32_t AVG_POOL_GRAD_DIM_THREE = 3;

const uint32_t H_DIM = 0;
const uint32_t W_DIM = 1;

const uint32_t TOP_PAD_INDEX = 0;
const uint32_t BOTTOM_PAD_INDEX = 1;
const uint32_t LEFT_PAD_INDEX = 2;
const uint32_t RIGHT_PAD_INDEX = 3;
static const gert::Shape g_vec_1_shape = {1};
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();

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

struct AvgPoolV2GradCommon {
    int64_t nDim;
    int64_t cDim;
    int64_t hDim;
    int64_t wDim;
    std::string padModeStr;
};

static inline void CalcAvgPoolGradSamePad(AvgPoolV2GradInputInfo& inputData)
{
    int64_t hPadNeed = std::max(int64_t{0}, (inputData.gradShape[H_DIM] - 1) * inputData.stride[H_DIM] +
                                                inputData.kernelSize[H_DIM] - inputData.inputShape[H_DIM]);
    int64_t topPad = hPadNeed / 2;
    int64_t bottomPad = hPadNeed - topPad;

    int64_t wPadNeed = std::max(int64_t{0}, (inputData.gradShape[W_DIM] - 1) * inputData.stride[W_DIM] +
                                                inputData.kernelSize[W_DIM] - inputData.inputShape[W_DIM]);
    int64_t leftPad = wPadNeed / 2;
    int64_t rightPad = wPadNeed - leftPad;

    inputData.pad = {topPad, bottomPad, leftPad, rightPad};
}

static inline ge::graphStatus CalcAvgPoolGradShapeInfo(gert::TilingContext* context, AvgPoolV2GradInputInfo& inputData,
                                                       AvgPoolV2GradCommon& commInfo, const int32_t* shapeValue)
{
    auto inputShape0 = context->GetInputShape(0);
    auto shapeDim = inputShape0->GetStorageShape().GetDim(0);
    if (inputData.inputFormat == ge::Format::FORMAT_NCHW) {
        if (shapeDim == CHW_DIMS) {
            commInfo.cDim = AVG_POOL_GRAD_DIM_ZERO;
            commInfo.hDim = AVG_POOL_GRAD_DIM_ONE;
            commInfo.wDim = AVG_POOL_GRAD_DIM_TWO;
            inputData.batches = shapeValue[commInfo.cDim];
        } else {
            commInfo.nDim = AVG_POOL_GRAD_DIM_ZERO;
            commInfo.cDim = AVG_POOL_GRAD_DIM_ONE;
            commInfo.hDim = AVG_POOL_GRAD_DIM_TWO;
            commInfo.wDim = AVG_POOL_GRAD_DIM_THREE;
            inputData.batches = shapeValue[commInfo.nDim] * shapeValue[commInfo.cDim];
        }
        inputData.channels = 1;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
        if (shapeDim == CHW_DIMS) {
            commInfo.cDim = AVG_POOL_GRAD_DIM_TWO;
            commInfo.hDim = AVG_POOL_GRAD_DIM_ZERO;
            commInfo.wDim = AVG_POOL_GRAD_DIM_ONE;
            inputData.batches = 1;
        } else {
            commInfo.nDim = AVG_POOL_GRAD_DIM_ZERO;
            commInfo.cDim = AVG_POOL_GRAD_DIM_THREE;
            commInfo.hDim = AVG_POOL_GRAD_DIM_ONE;
            commInfo.wDim = AVG_POOL_GRAD_DIM_TWO;
            inputData.batches = shapeValue[commInfo.nDim];
        }
        inputData.channels = shapeValue[commInfo.cDim];
    } else {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "data_format",
                                  Ops::Base::ToString(inputData.inputFormat).c_str(), "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }
    inputData.inputShape = {shapeValue[commInfo.hDim], shapeValue[commInfo.wDim]};
    return ge::GRAPH_SUCCESS;
}

static inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}
} // namespace optiling

#endif
