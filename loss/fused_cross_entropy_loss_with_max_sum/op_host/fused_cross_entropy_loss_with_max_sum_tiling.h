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
 * \file fused_cross_entropy_loss_with_max_sum.h
 * \brief
 */
#ifndef FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_H
#define FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_H

#include "register/tilingdata_base.h"

namespace optiling {
struct FusedCrossEntropyLossWithMaxSumCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t sysWorkspaceSize = 0;
    int64_t ubSizePlatForm = 0;
};

BEGIN_TILING_DATA_DEF(FusedCrossEntropyLossWithMaxSumTilingData)
TILING_DATA_FIELD_DEF(uint32_t, formerbtCountLen);
TILING_DATA_FIELD_DEF(uint32_t, latterbtCountLen);
TILING_DATA_FIELD_DEF(uint32_t, formerbtCountTime);
TILING_DATA_FIELD_DEF(uint32_t, latterbtCountTime);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreReservedbtNum);
TILING_DATA_FIELD_DEF(uint32_t, latterCoreReservedbtNum);
TILING_DATA_FIELD_DEF(uint32_t, singleCalculationQuantity);
TILING_DATA_FIELD_DEF(uint32_t, singleCalculationReservedQuantity);
TILING_DATA_FIELD_DEF(uint32_t, elementsNumber);
TILING_DATA_FIELD_DEF(uint32_t, vLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedCrossEntropyLossWithMaxSum, FusedCrossEntropyLossWithMaxSumTilingData)

ge::graphStatus TilingFusedCrossEntropyLossWithMaxSum(gert::TilingContext* context);
ge::graphStatus TilingPrepareForFusedCrossEntropyLossWithMaxSum(gert::TilingParseContext* context);

} // namespace optiling

#endif // FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_H
