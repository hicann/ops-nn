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
 * \file max_pool_grad_with_argmax_nhwc_merge_wc_tiling.cpp
 * \brief
 */
#include "max_pool_grad_with_argmax_nhwc_merge_wc_tiling.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling
{
static constexpr int64_t THRESHOLD_WC = 2;
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC = 600;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC = 601;

bool MaxPoolGradWithArgmaxNHWCMergeWcTiling::IsCapable()
{
    if (inputData.inputFormat == ge::Format::FORMAT_NCHW)  {
        return false;
    }
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxNHWCMergeWcTiling::IsCapable()");
    NHWCBase->InitializationVars(context_, &hardwareData);
    uint16_t computeSizeArgmax = NHWCBase->GetBaseData().vRegSize / NHWCBase->GetBaseData().indexBytes;
    uint16_t concurrencyCount = computeSizeArgmax / (inputData.cGrad * inputData.wGrad);
    if(concurrencyCount < THRESHOLD_WC) {
        return NHWCBase->CheckUBSize();
    }
    return false;
}

uint64_t MaxPoolGradWithArgmaxNHWCMergeWcTiling::GetTilingKey() const
{
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxNHWCMergeWcTiling::GetTilingKey()");
    uint64_t tilingKey = NO_CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC;
    if (NHWCBase->GetSplitData().isCheckRange == 1) {
        tilingKey = CHECK_RANGE_TILING_KEY_NHWC_MERGE_WC;
    }
    if (inputData.isInt32Meet == 0) {
        tilingKey += T3_INT64;
    }

    return tilingKey;
}

ge::graphStatus MaxPoolGradWithArgmaxNHWCMergeWcTiling::DoOpTiling()
{
    return NHWCBase->DoOpTiling(context_, GetTilingKey());
}

ge::graphStatus MaxPoolGradWithArgmaxNHWCMergeWcTiling::PostTiling()
{
    return NHWCBase->PostTiling(context_);
}

REGISTER_TILING_TEMPLATE("MaxPoolGradWithArgmax", MaxPoolGradWithArgmaxNHWCMergeWcTiling, 8);

}  // namespace optiling