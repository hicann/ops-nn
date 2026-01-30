/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_matmul_asw_basic_tiling.cpp
 * \brief
 */
#include "fused_matmul_basic_streamk_tiling.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace fused_matmul {
using matmul_v3_advanced::strategy::BASIC_STREAM_K;	
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulStreamKTiling, ASCEND950, BASIC_STREAM_K);

bool FusedMatMulStreamKTiling::IsCapable()
{
    if (args_.batchInfo->batchC > 1) {
        return false;
    }
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    // Only relu and "" support streamK
    if (opType != "relu" && opType != "") {
        return false;
    }
    return MatMulV3BasicStreamKTiling::IsCapable();
}

} // namespace fused_matmul
} // namespace optiling