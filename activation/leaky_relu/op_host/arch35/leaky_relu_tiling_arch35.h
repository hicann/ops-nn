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
 * \file leaky_relu_tiling_arch35.h
 * \brief
 */
#ifndef RUNTIME_V2_OP_IMPL_LEAKY_RELU_TILING_H_
#define RUNTIME_V2_OP_IMPL_LEAKY_RELU_TILING_H_

#include "../../op_kernel/arch35/leaky_relu_tilingdata.h"

namespace optiling {
struct LeakrReluCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};
}  // namespace optiling

#endif // RUNTIME_V2_OP_IMPL_LEAKY_RELU_TILING_H_