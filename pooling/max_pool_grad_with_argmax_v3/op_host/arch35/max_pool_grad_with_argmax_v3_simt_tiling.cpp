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
 * \file max_pool_grad_with_argmax_v3_ksize_one_tiling.cpp
 * \brief
 */

#include "max_pool_grad_with_argmax_v3_simt_tiling.h"
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling
{
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 1024;
static constexpr int64_t EXTRA_BUFFER_SIZE = 256;
static constexpr int64_t SIMT_NCHW_INT32_TILING_KEY = 900;
static constexpr int64_t SIMT_NCHW_INT64_TILING_KEY = 901;
static constexpr int64_t T3_INT64 = 10;
static constexpr int64_t CACHE_LINE_SIZE = 128;
static constexpr int64_t MIN_DATA_SIZE   = 1024;
static constexpr int64_t LOCAL_MEMORY_SIZE = 16384;
static constexpr int64_t SIMT_OUT_THRESHOLD = 256;

bool MaxPoolGradWithArgmaxV3SimtTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }

    if ((inputData.hKernel * inputData.wKernel < SIMT_OUT_THRESHOLD) &&
        (inputData.hGrad * inputData.wGrad < SIMT_OUT_THRESHOLD)) {
        return true;
    }

    return true;
}

uint64_t MaxPoolGradWithArgmaxV3SimtTiling::GetTilingKey() const
{
    int64_t totalData = inputData.nX * inputData.cX * inputData.hX * inputData.wX;
    uint64_t tilingKey = SIMT_NCHW_INT32_TILING_KEY;
    if (totalData > INT32_MAX) {
        tilingKey = SIMT_NCHW_INT64_TILING_KEY;
    }
    return tilingKey;
}

ge::graphStatus MaxPoolGradWithArgmaxV3SimtTiling::DoOpTiling()
{
    return SimtBase->DoOpTiling(context_);
}

ge::graphStatus MaxPoolGradWithArgmaxV3SimtTiling::PostTiling()
{
    return SimtBase->PostTiling(context_, hardwareData);
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3SimtTiling, 1);
}  // namespace optiling
