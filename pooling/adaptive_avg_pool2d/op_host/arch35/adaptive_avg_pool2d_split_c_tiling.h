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
 * \file adaptive_avg_pool2d_split_c_tiling.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL2D_SPLIT_C_TILING_H
#define ADAPTIVE_AVG_POOL2D_SPLIT_C_TILING_H

#include "adaptive_avg_pool2d_base_tiling.h"
#include "../../op_kernel/arch35/adaptive_avg_pool2d_struct.h"

namespace optiling {

struct SplitCComputeInfo : public CommonComputeInfo {
    uint64_t wInFactor{0};
};

DECLARE_SPLIT_TILING_CLASS(AdaptiveAvgPool2dSplitCTiling, SplitCComputeInfo);

} // namespace optiling
#endif // ADAPTIVE_AVG_POOL2D_SPLIT_C_TILING_H
