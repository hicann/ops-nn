/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file thnn_fused_lstm_cell_tiling.h
 * \brief
 */
#ifndef _THNN_FUSED_LSTM_CELL_TILING_H_
#define _THNN_FUSED_LSTM_CELL_TILING_H_
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ThnnFusedLstmCellTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, B);
    TILING_DATA_FIELD_DEF(uint64_t, H);
    TILING_DATA_FIELD_DEF(uint64_t, BH);
    TILING_DATA_FIELD_DEF(uint64_t, col);  // 任务块的列数
    TILING_DATA_FIELD_DEF(uint64_t, taskTotal);  // 总任务块数
    TILING_DATA_FIELD_DEF(uint64_t, taskSingle);  // 单个核心需要处理的任务块数

    TILING_DATA_FIELD_DEF(uint32_t, taskSize);  // 按照fp32算，单个任务块能容纳（处理）的元素数量
    TILING_DATA_FIELD_DEF(uint32_t, tailSize);  // 单个门行尾任务块的元素数量
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ThnnFusedLstmCell, ThnnFusedLstmCellTilingData);

struct ThnnFusedLstmCellCompileInfo {};

}  // namespace optiling
#endif  // _THNN_FUSED_LSTM_CELL_TILING_H_