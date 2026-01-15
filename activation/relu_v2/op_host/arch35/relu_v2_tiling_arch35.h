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
 * \file relu_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_RELU_V2_TILING_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_RELU_V2_TILING_H_

#include "register/tilingdata_base.h"
#include "op_common/atvoss/elewise/elewise_tiling.h"
#include "../../op_kernel/arch35/relu_v2_tiling_struct.h"

namespace optiling {

struct ReluV2CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_RELU_V2_TILING_H_