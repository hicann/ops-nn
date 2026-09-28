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
 * \file single_layer_lstm_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../single_layer_lstm_grad.h"
#include "../matmul_config.h"
#include "single_layer_lstm_grad_regbase_tiling_data.h"
#include "single_layer_lstm_grad_regbase_small.h"
#include "single_layer_lstm_grad_wide_adapter.h"
using namespace AscendC;

// Hundreds select XH_HUGE; thousands select DXH_HUGE.
#define LSTM_KEY_CONCAT_SMALL_FLAGS false, true
#define LSTM_KEY_SPLIT_SMALL_FLAGS true, false

#define GENERAL_OP_IMPL(templateClass, ...)                                                                        \
    do {                                                                                                           \
        templateClass<__VA_ARGS__> op;                                                                             \
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.dwMM, dwMMTiling, op.dgateMM, dgateMMTiling);            \
        op.Init(x, w, b, y, init_h, init_c, h, c, dy, dh, dc, i, j, f, o, tanhct, seq_length, dw, db, dx, dh_prev, \
                dc_prev, &tiling_data, workspace, &pipe);                                                          \
        op.Process();                                                                                              \
    } while (0)

extern "C" __global__ __aicore__ void single_layer_lstm_grad(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, GM_ADDR init_h,
                                                             GM_ADDR init_c, GM_ADDR h, GM_ADDR c, GM_ADDR dy,
                                                             GM_ADDR dh, GM_ADDR dc, GM_ADDR i, GM_ADDR j, GM_ADDR f,
                                                             GM_ADDR o, GM_ADDR tanhct, GM_ADDR seq_length, GM_ADDR dw,
                                                             GM_ADDR db, GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dc_prev,
                                                             GM_ADDR workspace, GM_ADDR rnnGradTiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(20000, KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(20000)) {
        // Path S owns an accumulator workspace for narrow weights. The public
        // framework pointer includes a reserved prefix; never use it as data.
        if (g_coreType == AscendC::AIC) {
            return;
        }
        GET_TILING_DATA_WITH_STRUCT(LstmGradRegbaseSmallTilingData, tilingDataSmall, rnnGradTiling);
        SetSysWorkspace(workspace);
        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        TPipe pipeSmall;
        // Floating IO share DTYPE_W; accumulation and UB staging use FP32.
        LstmGradRegbase::LstmGradRegbaseSmall<DTYPE_W> op;
        op.Init(x, w, b, init_h, init_c, h, c, dy, dh, dc, i, j, f, o, tanhct, dw, db, dx, dh_prev, dc_prev,
                userWorkspace, &tilingDataSmall, &pipeSmall);
        op.Process();
        return;
    }
    GET_TILING_DATA(tiling_data, rnnGradTiling);
    const SingleLayerLstmGradTilingData* __restrict tilingData = &tiling_data;
    const TCubeTiling* __restrict dwMMTiling = &(tilingData->dwMMParam);
    const TCubeTiling* __restrict dgateMMTiling = &(tilingData->dgateMMParam);
    TPipe pipe;
    GM_ADDR outputDx = dx;
    GM_ADDR outputDw = dw;
    GM_ADDR outputDb = db;
    GM_ADDR outputDh = dh_prev;
    GM_ADDR outputDc = dc_prev;
    LstmGradWide::Adapter<DTYPE_W> replay;
    if constexpr (!std::is_same<DTYPE_W, float>::value) {
        SetSysWorkspace(workspace);
        replay.Init(GetUserWorkspace(workspace), tilingData->timeStep, tilingData->batch, tilingData->inputSize,
                    tilingData->hiddenSize, tilingData->privateBiasComponents, tilingData->direction,
                    tilingData->gateOrder, &pipe);
        replay.Prepare(x, w, b, init_h, init_c, dy, dh, dc);
        x = replay.At(replay.layout.x);
        w = replay.At(replay.layout.w);
        b = replay.At(replay.layout.bias);
        init_h = replay.At(replay.layout.initH);
        init_c = replay.At(replay.layout.initC);
        dy = replay.At(replay.layout.dy);
        dh = replay.At(replay.layout.dh);
        dc = replay.At(replay.layout.dc);
        i = replay.Plane(0);
        j = replay.Plane(1);
        f = replay.Plane(2);
        o = replay.Plane(3);
        tanhct = replay.Plane(4);
        c = replay.Plane(5);
        h = replay.Plane(6);
        y = h;
        dx = replay.At(replay.layout.dx);
        dw = replay.At(replay.layout.dw);
        db = replay.At(replay.layout.db);
        dh_prev = replay.At(replay.layout.dhPrev);
        dc_prev = replay.At(replay.layout.dcPrev);
        workspace = replay.At(replay.layout.legacy);
    }
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_CFG, MM_CFG, false, false);
    } else if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_HUGE_CFG, MM_CFG, false, false);
    } else if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_CFG, MM_HUGE_CFG, false, false);
    } else if (TILING_KEY_IS(11)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_HUGE_CFG, MM_HUGE_CFG, false, false);
    } else if (TILING_KEY_IS(100)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_CFG, MM_CFG, LSTM_KEY_CONCAT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(101)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_HUGE_CFG, MM_CFG, LSTM_KEY_CONCAT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(110)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_CFG, MM_HUGE_CFG, LSTM_KEY_CONCAT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(111)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_HUGE_CFG, MM_HUGE_CFG, LSTM_KEY_CONCAT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1000)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_CFG, MM_CFG, LSTM_KEY_SPLIT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1001)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_HUGE_CFG, MM_CFG, LSTM_KEY_SPLIT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1010)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_CFG, MM_HUGE_CFG, LSTM_KEY_SPLIT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1011)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_HUGE_CFG, MM_HUGE_CFG, LSTM_KEY_SPLIT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1100)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_CFG, MM_CFG, true, true);
    } else if (TILING_KEY_IS(1101)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_HUGE_CFG, MM_CFG, true, true);
    } else if (TILING_KEY_IS(1110)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_CFG, MM_HUGE_CFG, true, true);
    } else if (TILING_KEY_IS(1111)) {
        GENERAL_OP_IMPL(RNNGrad, float, MM_HUGE_CFG, MM_HUGE_CFG, true, true);
    }
    if constexpr (!std::is_same<DTYPE_W, float>::value) {
        replay.Finish(outputDx, outputDw, outputDb, outputDh, outputDc);
    }

#ifdef __CCE_KT_TEST__
    EmptyTestFunc();
#endif // __CCE_KT_TEST__
}
