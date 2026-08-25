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
 * \file apply_came_part1_tiling.h
 * \brief
 */
#ifndef OPS_NN_OPTIM_APPLY_CAME_PART1_TILING_ARCH35_H_
#define OPS_NN_OPTIM_APPLY_CAME_PART1_TILING_ARCH35_H_

#include <cstdint>
#include <vector>
#include "register/op_impl_registry.h"

namespace optiling {

#include "optim/apply_came_part1/op_kernel/arch35/apply_came_part1_tiling_data.h"

struct ApplyCamePart1CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

ge::graphStatus TilingApplyCamePart1(gert::TilingContext* context);

} // namespace optiling
#endif // OPS_NN_OPTIM_APPLY_CAME_PART1_TILING_ARCH35_H_
