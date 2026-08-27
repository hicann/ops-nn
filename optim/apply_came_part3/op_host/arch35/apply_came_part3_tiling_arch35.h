/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_APPLY_CAME_PART3_TILING_ARCH35_H_
#define OPS_NN_APPLY_CAME_PART3_TILING_ARCH35_H_

#include <cstdint>
#include "register/op_impl_registry.h"

namespace optiling {
#include "optim/apply_came_part3/op_kernel/arch35/apply_came_part3_tiling_data.h"

struct ApplyCamePart3CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatform = 0;
};

ge::graphStatus TilingApplyCamePart3(gert::TilingContext* context);
} // namespace optiling

#endif // OPS_NN_APPLY_CAME_PART3_TILING_ARCH35_H_
