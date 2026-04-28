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
 * \file max_pool_grad_with_argmax_v3_nchw_tiling.cpp
 * \brief MaxPoolGradWithArgmaxV3 NCHW格式Tiling实现，使用公共Tiling类
 */
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool_grad_with_argmax_v3_nchw_tiling.h"

namespace optiling {
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NCHW = 100;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NCHW = 101;
static constexpr int64_t T3_INT64 = 10;

bool MaxPoolGradWithArgmaxV3NCHWTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }

    nchwTilingCommon.InitializationVars(context_, &hardwareData);
    return nchwTilingCommon.CheckUBSize();
}

uint64_t MaxPoolGradWithArgmaxV3NCHWTiling::GetTilingKey() const
{
    uint64_t tilingKey = NO_CHECK_RANGE_TILING_KEY_NCHW;
    auto splitData = nchwTilingCommon.GetSplitData();
    if (splitData.isCheckRange == 1) {
        tilingKey = CHECK_RANGE_TILING_KEY_NCHW;
    }
    if (inputData.isInt32Meet == 0) {
        tilingKey += T3_INT64;
    }

    return tilingKey;
}

ge::graphStatus MaxPoolGradWithArgmaxV3NCHWTiling::DoOpTiling()
{
    return nchwTilingCommon.DoOpTiling(context_, GetTilingKey());
}

ge::graphStatus MaxPoolGradWithArgmaxV3NCHWTiling::PostTiling()
{
    return nchwTilingCommon.PostTiling(context_);
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3NCHWTiling, 2);

} // namespace optiling