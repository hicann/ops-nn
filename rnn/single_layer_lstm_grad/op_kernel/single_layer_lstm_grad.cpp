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
#include "single_layer_lstm_grad.h"
#include "matmul_config.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/single_layer_lstm_grad_regbase_tiling_data.h"
#include "arch35/single_layer_lstm_grad_regbase_small.h"
#endif
using namespace AscendC;

// key 的百位来自 concat 的小 UB 判定(对应 XH_HUGE),千位来自 dxh 拆分的小 UB 判定(对应 DXH_HUGE)。
// legacy soc 上这两位与模板实参的对应关系是反的,属历史遗留,仅 fp16 且 inputSize+hiddenSize 约
// 33k~49k 区间可达;此处保持 legacy 现状不变,arch35 采用与 tiling 意图一致的对应。
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define LSTM_KEY_CONCAT_SMALL_FLAGS false, true
#define LSTM_KEY_SPLIT_SMALL_FLAGS true, false
#else
#define LSTM_KEY_CONCAT_SMALL_FLAGS true, false
#define LSTM_KEY_SPLIT_SMALL_FLAGS false, true
#endif

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
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(20000, KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(20000)) {
        // Path S: small-shape AIV-only regbase kernel, zero sync, zero workspace
        if (g_coreType == AscendC::AIC) {
            return;
        }
        GET_TILING_DATA_WITH_STRUCT(LstmGradRegbaseSmallTilingData, tilingDataSmall, rnnGradTiling);
        TPipe pipeSmall;
        LstmGradRegbase::LstmGradRegbaseSmall<DTYPE_X> op;
        op.Init(x, w, init_h, init_c, h, c, dy, dh, dc, i, j, f, o, tanhct, dw, db, dx, dh_prev, dc_prev,
                &tilingDataSmall, &pipeSmall);
        op.Process();
        return;
    }
#endif
    GET_TILING_DATA(tiling_data, rnnGradTiling);
    const SingleLayerLstmGradTilingData* __restrict tilingData = &tiling_data;
    const TCubeTiling* __restrict dwMMTiling = &(tilingData->dwMMParam);
    const TCubeTiling* __restrict dgateMMTiling = &(tilingData->dgateMMParam);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_CFG, false, false);
    } else if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_CFG, false, false);
    } else if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_HUGE_CFG, false, false);
    } else if (TILING_KEY_IS(11)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_HUGE_CFG, false, false);
    } else if (TILING_KEY_IS(100)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_CFG, LSTM_KEY_CONCAT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(101)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_CFG, LSTM_KEY_CONCAT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(110)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_HUGE_CFG, LSTM_KEY_CONCAT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(111)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_HUGE_CFG, LSTM_KEY_CONCAT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1000)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_CFG, LSTM_KEY_SPLIT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1001)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_CFG, LSTM_KEY_SPLIT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1010)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_HUGE_CFG, LSTM_KEY_SPLIT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1011)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_HUGE_CFG, LSTM_KEY_SPLIT_SMALL_FLAGS);
    } else if (TILING_KEY_IS(1100)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_CFG, true, true);
    } else if (TILING_KEY_IS(1101)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_CFG, true, true);
    } else if (TILING_KEY_IS(1110)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_HUGE_CFG, true, true);
    } else if (TILING_KEY_IS(1111)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_HUGE_CFG, true, true);
    }

#ifdef __CCE_KT_TEST__
    EmptyTestFunc();
#endif // __CCE_KT_TEST__
}
