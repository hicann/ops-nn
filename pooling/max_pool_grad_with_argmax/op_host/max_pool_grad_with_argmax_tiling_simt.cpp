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
 * \file max_pool_grad_with_argmax_tiling_simt.cpp
 * \brief
 */

#include <cctype>
#include <algorithm>

#include "tiling_base/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "max_pool_grad_with_argmax_tiling_simt.h"  
using namespace AscendC;
using namespace ge;

namespace optiling {

int64_t outputDataCount = 0;  // total elements in grad_input
constexpr int64_t MAX_INT32 = 2147483647;
constexpr uint64_t SIMT_NCHW_TILING_KEY_INT32 = 801;
constexpr uint64_t SIMT_NHWC_TILING_KEY_INT32 = 802;
constexpr uint64_t SIMT_NCHW_TILING_KEY_INT64 = 803;
constexpr uint64_t SIMT_NHWC_TILING_KEY_INT64 = 804;

bool MaxPoolGradWithArgmaxTilingSIMT::IsCapable()
{
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingSIMT::IsCapable()");
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context_)){
        return true;
    }
    return false;
}


uint64_t MaxPoolGradWithArgmaxTilingSIMT::GetTilingKey() const
{  
    outputDataCount = inputData.nX * inputData.cX * inputData.wX * inputData.hX;
    if (inputData.inputFormat == ge::Format::FORMAT_NCHW && outputDataCount <= MAX_INT32) {
        return SIMT_NCHW_TILING_KEY_INT32;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NHWC && outputDataCount <= MAX_INT32) {
        return SIMT_NHWC_TILING_KEY_INT32;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NHWC && outputDataCount > MAX_INT32) {
        return SIMT_NHWC_TILING_KEY_INT64;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NCHW && outputDataCount > MAX_INT32) {
        return SIMT_NCHW_TILING_KEY_INT64;
    }
    return SIMT_NCHW_TILING_KEY_INT32;
}

ge::graphStatus MaxPoolGradWithArgmaxTilingSIMT::DoOpTiling()
{   
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingSIMT::DoOpTiling()");
    return SimtBase->DoOpTiling(context_, GetTilingKey());
}

ge::graphStatus MaxPoolGradWithArgmaxTilingSIMT::PostTiling()
{   
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxTilingSIMT::PostTiling()");
    return SimtBase->PostTiling(context_, hardwareData);
}

REGISTER_TILING_TEMPLATE("MaxPoolGradWithArgmax", MaxPoolGradWithArgmaxTilingSIMT, 10);

}  // namespace optiling