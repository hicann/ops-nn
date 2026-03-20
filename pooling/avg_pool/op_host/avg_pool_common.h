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
 * \file avg_pool_common.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_AVGPOOL_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_AVGPOOL_COMMON_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"

namespace optiling
{
const int32_t HW_DIMS = 2;
const int32_t PAD_DIMS = 4;

const int32_t ONE_DIMS = 1;
const int32_t NCHW_DIMS = 4;

const uint32_t H_DIM = 0;
const uint32_t W_DIM = 1;

const uint32_t TOP_PAD_INDEX = 0;
const uint32_t BOTTOM_PAD_INDEX = 1;
const uint32_t LEFT_PAD_INDEX = 2;
const uint32_t RIGHT_PAD_INDEX = 3;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const int64_t OP_TYPE_AVG_POOL_V1 = 0;
const int64_t OP_TYPE_AVG_POOL_V2 = 1;

struct AvgPoolInputInfo {
    int64_t batches;
    int64_t channels;
    std::array<int64_t, HW_DIMS> inputShape;
    std::array<int64_t, HW_DIMS> outShape;
    std::array<int64_t, HW_DIMS> kernelSize;
    std::array<int64_t, HW_DIMS> stride;
    std::array<int64_t, PAD_DIMS> pad;
    bool ceilMode = false; 
    bool countIncludePad = false;
    bool globalPooling = false;
    int64_t divisorOverride = 0;
    ge::Format inputFormat;
    int64_t dtypeSize = 0;
};

int64_t AvgPoolDivRtn(const int64_t x, const int64_t y);

}  // namespace optiling

#endif