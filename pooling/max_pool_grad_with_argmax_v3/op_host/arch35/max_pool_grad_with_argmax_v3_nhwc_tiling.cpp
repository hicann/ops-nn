/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_grad_with_argmax_v3_nhwc_tiling.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"
#include "max_pool_grad_with_argmax_v3_nhwc_tiling.h"

namespace optiling {
static constexpr int64_t THRESHOLD = 2;
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NHWC = 500;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NHWC = 501;
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC = 600;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC = 601;
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NHWC_BIGC = 700;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NHWC_BIGC = 701;

bool MaxPoolGradWithArgmaxV3NHWCTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NHWC) {
        return false;
    }

    NHWCBase->InitializationVars(context_, &hardwareData);
    return true;
}

uint64_t MaxPoolGradWithArgmaxV3NHWCTiling::GetTilingKey() const
{
    uint16_t computeSizeArgmax = NHWCBase->GetBaseData().vRegSize / NHWCBase->GetBaseData().indexBytes;

    uint64_t tilingKey = CHECK_RANGE_TILING_KEY_NHWC;
    if(computeSizeArgmax / inputData.cGrad < THRESHOLD) {
        tilingKey = NO_CHECK_RANGE_TILING_KEY_NHWC_BIGC;
    } else if(computeSizeArgmax / (inputData.cGrad * inputData.wGrad) < THRESHOLD) {
        tilingKey = NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC;
    }

    if (NHWCBase->GetSplitData().isCheckRange == 1) {
        tilingKey += 1;
    }

    if (inputData.isInt32Meet == 0) {
        tilingKey += T3_INT64;
    }

    return tilingKey;
}

ge::graphStatus MaxPoolGradWithArgmaxV3NHWCTiling::DoOpTiling()
{
    return NHWCBase->DoOpTiling(context_, GetTilingKey());
}

ge::graphStatus MaxPoolGradWithArgmaxV3NHWCTiling::PostTiling()
{
    return NHWCBase->PostTiling(context_);
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3NHWCTiling, 3);

} // namespace optiling
