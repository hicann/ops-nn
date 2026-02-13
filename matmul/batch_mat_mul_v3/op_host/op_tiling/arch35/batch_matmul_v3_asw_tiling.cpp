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
 * \file batch_matmul_v3_asw_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_asw_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3AswTiling, DAV_3510, BASE);
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3AswTiling, DAV_RESV, BASE); //supportMmadS8S4平台
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, BatchMatMulV3AswTiling, DAV_RESV, BASE); //supportMmadS8S4平台

ge::graphStatus BatchMatMulV3AswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchMatMulV3AswTiling::GetTilingKey() const
{
    return BatchMatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::BASIC)
        .GetTilingKey();
}

uint64_t BatchMatMulV3AswTiling::GetNumBlocks() const
{
    return compileInfo_.aicNum;
}

ge::graphStatus BatchMatMulV3AswTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3TilingData>(tiling);
}
} // namespace batch_matmul_v3_advanced
} // namespace optiling