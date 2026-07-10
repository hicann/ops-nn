/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include "exe_graph/runtime/tiling_context.h"
#include "../../op_kernel/arch35/global_lp_pool_tiling_data.h"

namespace gert {
namespace tiling {

// Tiling function - called during graph compilation to compute split parameters
class GlobalLpPoolTilingFunc {
public:
    static ge::graphStatus ComputeTiling(gert::TilingContext* ctx);
};

} // namespace tiling
} // namespace gert
