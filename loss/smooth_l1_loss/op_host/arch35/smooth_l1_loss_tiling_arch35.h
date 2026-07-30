/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_TILING_H_

#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/smooth_l1_loss_tilingdata.h"

namespace optiling {

struct SmoothL1LossCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_TILING_H_
