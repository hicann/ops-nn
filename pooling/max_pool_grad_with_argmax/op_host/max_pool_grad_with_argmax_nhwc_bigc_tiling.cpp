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
* \file max_pool_grad_with_argmax_nhwc_bigc_tiling.cpp
* \brief
*/
#include "max_pool_grad_with_argmax_nhwc_bigc_tiling.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling
{
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NHWC_BIGC = 700;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NHWC_BIGC = 701;
static constexpr int64_t TWO_BIGC = 2;

bool MaxPoolGradWithArgmaxNHWCBigcTiling::IsCapable()
{
    if (inputData.inputFormat == ge::Format::FORMAT_NCHW)  {
        return false;
    }
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxNHWCBigcTiling::IsCapable()");
    NHWCBase->InitializationVars(context_, &hardwareData);
    uint16_t computeSizeArgmax = NHWCBase->GetBaseData().vRegSize / NHWCBase->GetBaseData().indexBytes;
    uint16_t concurrencyCount = computeSizeArgmax / inputData.cGrad;
    if(concurrencyCount < TWO_BIGC) {
        return NHWCBase->CheckUBSize();
    }
    return false;
}

uint64_t MaxPoolGradWithArgmaxNHWCBigcTiling::GetTilingKey() const
{
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxNHWCBigcTiling::GetTilingKey()");
    uint64_t tilingKey = NO_CHECK_RANGE_TILING_KEY_NHWC_BIGC;
    if (NHWCBase->GetSplitData().isCheckRange == 1) {
        tilingKey = CHECK_RANGE_TILING_KEY_NHWC_BIGC;
    }
    if (inputData.isInt32Meet == 0) {
        tilingKey += T3_INT64;
    }
    return tilingKey;
}

ge::graphStatus MaxPoolGradWithArgmaxNHWCBigcTiling::DoOpTiling()
{
    return NHWCBase->DoOpTiling(context_, GetTilingKey());
}

ge::graphStatus MaxPoolGradWithArgmaxNHWCBigcTiling::PostTiling()
{
    return NHWCBase->PostTiling(context_);
}

REGISTER_TILING_TEMPLATE("MaxPoolGradWithArgmax", MaxPoolGradWithArgmaxNHWCBigcTiling, 7);

}  // namespace optiling