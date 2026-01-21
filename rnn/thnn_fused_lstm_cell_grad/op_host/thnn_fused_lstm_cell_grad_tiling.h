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
 * \file thnn_fused_lstm_cell_grad_tiling.h
 * \brief
 */
#ifndef _THNN_FUSED_LSTM_CELL_GRAD_TILING_H_
#define _THNN_FUSED_LSTM_CELL_GRAD_TILING_H_
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(ThnnFusedLstmCellGradTilingData)
        TILING_DATA_FIELD_DEF(int64_t, ubSize);
        // rnn input tiling
        TILING_DATA_FIELD_DEF(int64_t, batch);
        TILING_DATA_FIELD_DEF(int64_t, hiddenSize);

        // vector block params
        TILING_DATA_FIELD_DEF(int64_t, singleCoreM);
        TILING_DATA_FIELD_DEF(int64_t, singleCoreMTail);
        TILING_DATA_FIELD_DEF(int64_t, singleCoreN);
        TILING_DATA_FIELD_DEF(int64_t, singleCoreNTail);
        TILING_DATA_FIELD_DEF(int64_t, baseN);
        TILING_DATA_FIELD_DEF(int64_t, baseM);
        TILING_DATA_FIELD_DEF(int64_t, mCnt);
        TILING_DATA_FIELD_DEF(int64_t, nCnt);

        // reduce block params
        TILING_DATA_FIELD_DEF(int64_t, singleCoreReduceN);
        TILING_DATA_FIELD_DEF(int64_t, singleCoreReduceNTail);
        TILING_DATA_FIELD_DEF(int64_t, baseReduceN);
        TILING_DATA_FIELD_DEF(int64_t, nReduceCnt);
        TILING_DATA_FIELD_DEF(int64_t, maxReduceNumOnce);
        TILING_DATA_FIELD_DEF(int64_t, reduceBlockSize);

        // rnn attr
        TILING_DATA_FIELD_DEF(int64_t, isBias);
    END_TILING_DATA_DEF;
    REGISTER_TILING_DATA_CLASS(ThnnFusedLstmCellGrad, ThnnFusedLstmCellGradTilingData)

    struct ThnnFusedLstmCellGradTilingParams {
        // include 2 matmul tiling
      int64_t tilingKey{0};
      int64_t usedCoreNum{0};

      // rnn input tiling
      int64_t batch{0};
      int64_t hiddenSize{0};

      // tmp
      int64_t ubSize{0};
      int64_t sysAivCoreNum{0};

      // vector block params
      int64_t singleCoreM{0};
      int64_t singleCoreMTail{0};
      int64_t singleCoreN{0};
      int64_t singleCoreNTail{0};
      int64_t baseN{0};
      int64_t baseM{0};
      int64_t mCnt{0};
      int64_t nCnt{0};

      // reduce block params
      int64_t singleCoreReduceN{0};
      int64_t singleCoreReduceNTail{0};
      int64_t baseReduceN{0};
      int64_t nReduceCnt{0};
      int64_t maxReduceNumOnce{0};
      int64_t reduceBlockSize{0};
      
      // attr
      int64_t isBias{0};
    };

    struct ThnnFusedLstmCellGradCompileInfo {
        uint32_t aicCoreNum{0};
        int64_t sysWorkspaceSize{0};
        int64_t ubSizePlatForm{0};
    };
}
#endif