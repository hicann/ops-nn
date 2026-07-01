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
 * \file test_thnn_fused_lstm_cell_tiling_def.h
 * \brief
 */

#ifndef TEST_THNN_FUSED_LSTM_CELL_TILING_DEF_H_
#define TEST_THNN_FUSED_LSTM_CELL_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

struct ThnnFusedLstmCellTilingDataTest {
    uint64_t B = 0;
    uint64_t H = 0;
    uint64_t BH = 0;
    uint64_t col = 0;
    uint64_t taskTotal = 0;
    uint64_t taskSingle = 0;
    uint64_t taskSize = 0;
    uint64_t tailSize = 0;
};

inline void InitThnnFusedLstmCellTilingDataTest(uint8_t* tiling, ThnnFusedLstmCellTilingDataTest* data)
{
    memcpy(data, tiling, sizeof(ThnnFusedLstmCellTilingDataTest));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                    \
    ThnnFusedLstmCellTilingDataTest tiling_data;                  \
    InitThnnFusedLstmCellTilingDataTest(tiling_arg, &tiling_data)
#endif // FUSED_LINEAR_ONLINE_MAX_SUM_TILING_H_TEST_KERNEL