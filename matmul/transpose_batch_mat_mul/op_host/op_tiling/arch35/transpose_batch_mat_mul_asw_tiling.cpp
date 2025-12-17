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
 * \file transpose_batch_mat_mul_asw_tiling.cc
 * \brief
 */

#include "transpose_batch_mat_mul_asw_tiling.h"
#include "transpose_batch_mat_mul_tiling_strategy.h"
#include "transpose_batch_mat_mul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace optiling {
namespace transpose_batch_mat_mul_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(TransposeBatchMatMul, TransposeBatchMatMulAswTiling, ASCEND910_95, BASE);

ge::graphStatus TransposeBatchMatMulAswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    return ge::GRAPH_SUCCESS;
}

uint64_t TransposeBatchMatMulAswTiling::GetTilingKey() const
{
    return MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .GetTilingKey();
}

uint64_t TransposeBatchMatMulAswTiling::GetBlockDim() const
{
    return compileInfo_.aicNum;
}
}
}