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
#include "fused_matmul_asw_basic_tiling.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace fused_matmul {
using matmul_v3_advanced::strategy::BASIC_ASWT;	
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulAswBasicApiTiling, DAV_3510, BASIC_ASWT);

bool FusedMatMulAswBasicApiTiling::IsCapable()
{
    if (args_.batchInfo->batchC > 1) {
        return false;
    }
    if (args_.bFormat != ge::FORMAT_ND || args_.aFormat != ge::FORMAT_ND) {
        OP_LOGD(args_.opName, "ND is the only supported format for basic api");
        return false;
    }

    OP_LOGI(args_.opName, "FusedMatMul tiling enable state basic api");
    return true;
}

uint64_t FusedMatMulAswBasicApiTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

ge::graphStatus FusedMatMulAswBasicApiTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3BasicTilingData>(tiling);
}
} // namespace fused_matmul
} // namespace optiling