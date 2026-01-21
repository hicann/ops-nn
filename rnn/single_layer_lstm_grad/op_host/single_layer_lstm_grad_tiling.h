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
 * \file single_layer_lstm_grad_tiling.h
 * \brief
 */
#ifndef _SINGLE_LAYER_LSTM_GRAD_TILING_H_
#define _SINGLE_LAYER_LSTM_GRAD_TILING_H_
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
    enum class GateOrder : int64_t {
        IJFO = 0,
        IFJO
    };

    struct CutBatchTilingParam {
        int64_t taskNum{0};
        int64_t copyMLines{0};
        int64_t copyMLinesTail{0};
        int64_t nLoop{0};
        int64_t copyNLength{0};
        int64_t copyNLengthTail{0};
        int64_t splitTaskPerCore{0};
        int64_t splitPreCore{0};
    };

    BEGIN_TILING_DATA_DEF(CutBatchTiling)
        TILING_DATA_FIELD_DEF(int64_t, taskNum);
        TILING_DATA_FIELD_DEF(int64_t, copyMLines);
        TILING_DATA_FIELD_DEF(int64_t, copyMLinesTail);
        TILING_DATA_FIELD_DEF(int64_t, nLoop);
        TILING_DATA_FIELD_DEF(int64_t, copyNLength);
        TILING_DATA_FIELD_DEF(int64_t, copyNLengthTail);
        TILING_DATA_FIELD_DEF(int64_t, splitTaskPerCore);
        TILING_DATA_FIELD_DEF(int64_t, splitPreCore);
    END_TILING_DATA_DEF;
    REGISTER_TILING_DATA_CLASS(CutBatchTilingOp, CutBatchTiling)

    BEGIN_TILING_DATA_DEF(SingleLayerLstmGradTilingData)
        TILING_DATA_FIELD_DEF(int64_t, ubSize);
        // rnn input tiling
        TILING_DATA_FIELD_DEF(int64_t, timeStep);
        TILING_DATA_FIELD_DEF(int64_t, batch);
        TILING_DATA_FIELD_DEF(int64_t, inputSize);
        TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
        TILING_DATA_FIELD_DEF(int64_t, isBias);
        TILING_DATA_FIELD_DEF(int64_t, isSeqLength);

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
        TILING_DATA_FIELD_DEF(int64_t, gateOrder);
        TILING_DATA_FIELD_DEF(int64_t, direction);
        TILING_DATA_FIELD_DEF(float, cellClip);
        TILING_DATA_FIELD_DEF(float, forgetBias);

        // split and concat params
        TILING_DATA_FIELD_DEF(int64_t, inputSizeAligned);
        TILING_DATA_FIELD_DEF(int64_t, hiddenSizeAligned);
        TILING_DATA_FIELD_DEF(int64_t, oneLineAligned);
        TILING_DATA_FIELD_DEF_STRUCT(CutBatchTiling, dxhInputTiling);
        TILING_DATA_FIELD_DEF_STRUCT(CutBatchTiling, dxhHiddenTiling);
        TILING_DATA_FIELD_DEF_STRUCT(CutBatchTiling, xhInputTiling);
        TILING_DATA_FIELD_DEF_STRUCT(CutBatchTiling, xhHiddenTiling);

        // matmul params
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, dwMMParam);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, dgateMMParam);
    END_TILING_DATA_DEF;
    REGISTER_TILING_DATA_CLASS(SingleLayerLstmGrad, SingleLayerLstmGradTilingData)

    struct SingleLayerLstmGradTilingParams {
        // include 2 matmul tiling
      int64_t tilingKey{0};
      int64_t usedCoreNum{0};

      // rnn input tiling
      int64_t timeStep{0};
      int64_t batch{0};
      int64_t inputSize{0};
      int64_t hiddenSize{0};
      int64_t isBias{0};
      int64_t isInithc{0};
      int64_t isSeqLength{0};

      // rnn attr
      int64_t gateOrder{1};
      int64_t direction{0};
      float cellClip{0.0f};
      float forgetBias{0.0f};

      // tmp
      int64_t ubSize{0};
      int64_t sysAicCoreNum{0};
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
      
      // aligned params
      int64_t inputSizeAligned{0};
      int64_t hiddenSizeAligned{0};
      int64_t oneLineAligned{0};
    };

    struct SingleLayerLstmGradCompileInfo {
        uint32_t aicCoreNum{0};
        uint32_t aivCoreNum{0};
        int64_t sysWorkspaceSize{0};
        int64_t ubSizePlatForm{0};
    };
}
#endif