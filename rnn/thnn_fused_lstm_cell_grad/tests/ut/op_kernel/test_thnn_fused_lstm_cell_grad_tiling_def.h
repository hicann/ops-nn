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
 * \file test_thnn_fused_lstm_cell_grad_tiling_def.h
 * \brief
 */
#ifndef TEST_THNN_FUSED_LSTM_CELL_GRAD_TILING_DEF_H_
#define TEST_THNN_FUSED_LSTM_CELL_GRAD_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

struct ThnnFusedLstmCellGradTilingDataTest {
    int64_t ubSize = 0;
    // lstm input tiling
    int64_t batch = 0;
    int64_t hiddenSize = 0;

    // vector block params
    int64_t singleCoreM = 0;
    int64_t singleCoreMTail = 0;
    int64_t singleCoreN = 0;
    int64_t singleCoreNTail = 0;
    int64_t baseN = 0;
    int64_t baseM = 0;
    int64_t mCnt = 0;
    int64_t nCnt = 0;

    // reduce block params
    int64_t singleCoreReduceN = 0;
    int64_t singleCoreReduceNTail = 0;
    int64_t baseReduceN = 0;
    int64_t nReduceCnt = 0;
    int64_t maxReduceNumOnce = 0;
    int64_t reduceBlockSize = 0;

    // lstm attr
    int64_t isBias = 0;
};

inline void InitThnnFusedLstmCellGradTilingDataTest(uint8_t* tiling, ThnnFusedLstmCellGradTilingDataTest* data)
{
    memcpy(data, tiling, sizeof(ThnnFusedLstmCellGradTilingDataTest));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                    \
    ThnnFusedLstmCellGradTilingDataTest tiling_data;                  \
    InitThnnFusedLstmCellGradTilingDataTest(tiling_arg, &tiling_data)
#endif // FUSED_LINEAR_ONLINE_MAX_SUM_TILING_H_TEST_KERNEL