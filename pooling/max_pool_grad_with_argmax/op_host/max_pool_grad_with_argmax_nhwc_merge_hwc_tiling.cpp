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
 * \file max_pool_grad_with_argmax_nhwc_merge_hwc_tiling.cpp
 * \brief
 */
#include "max_pool_grad_with_argmax_nhwc_merge_hwc_tiling.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling
{
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NHWC = 500;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NHWC = 501;
static constexpr int64_t T3_INT64 = 10;
static constexpr int64_t THRESHOLD_HWC = 2;

bool MaxPoolGradWithArgmaxMergeHWCTiling::IsCapable()
{
    NHWCBase->InitializationVars(context_, &hardwareData);
    uint16_t computeSizeArgmax = NHWCBase->GetBaseData().vRegSize / NHWCBase->GetBaseData().indexBytes;
    uint16_t concurrencyCount = computeSizeArgmax / (inputData.cGrad * inputData.wGrad);
    if (inputData.inputFormat != ge::Format::FORMAT_NHWC || concurrencyCount < THRESHOLD_HWC) {
        return false;
    }
    return NHWCBase->CheckUBSize();
}

uint64_t MaxPoolGradWithArgmaxMergeHWCTiling::GetTilingKey() const
{
    uint64_t tilingKey = NO_CHECK_RANGE_TILING_KEY_NHWC;
    if (NHWCBase->GetSplitData().isCheckRange == 1) {
        tilingKey = CHECK_RANGE_TILING_KEY_NHWC;
    }
    if (inputData.isInt32Meet == 0) {
        tilingKey += T3_INT64;
    }
    return tilingKey;
}

ge::graphStatus MaxPoolGradWithArgmaxMergeHWCTiling::DoOpTiling()
{
    return NHWCBase->DoOpTiling(context_, GetTilingKey());
}

ge::graphStatus MaxPoolGradWithArgmaxMergeHWCTiling::PostTiling()
{
    return NHWCBase->PostTiling(context_);
}

REGISTER_TILING_TEMPLATE("MaxPoolGradWithArgmax", MaxPoolGradWithArgmaxMergeHWCTiling, 9);

}  // namespace optiling
