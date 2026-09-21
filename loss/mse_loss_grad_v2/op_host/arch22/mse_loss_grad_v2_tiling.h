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
 * \file mse_loss_grad_v2_tiling.h
 * \brief
 */
#ifndef MSE_LOSS_GRAD_V2_TILING_H
#define MSE_LOSS_GRAD_V2_TILING_H

#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossGradTilingData)
TILING_DATA_FIELD_DEF(float, cof);
TILING_DATA_FIELD_DEF(uint64_t, totalLength); // 总计算数据量
TILING_DATA_FIELD_DEF(uint64_t, tileNum);     // 每个核上总计算数据分块个数
TILING_DATA_FIELD_DEF(uint64_t, padLength);   // 尾块的个数
TILING_DATA_FIELD_DEF(uint64_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, usedDb); //  8 Bytes align with cof
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossGradV2, MseLossGradTilingData)

void GetTilingKey(const uint32_t dtypeKey, uint32_t& tilingKey);
} // namespace optiling
#endif
