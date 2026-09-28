/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_grad_tiling_data.h
 * \brief DynamicAUGRUGrad tiling data struct（ascend950/arch35）
 */

#ifndef __DYNAMIC_AUGRU_GRAD_TILING_DATA_H__
#define __DYNAMIC_AUGRU_GRAD_TILING_DATA_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// gate_order取值：zrh(z,r,n)为默认门序，rzh(r,z,n)为备选
constexpr int64_t DYNAMIC_AUGRU_GRAD_GATE_ZRH = 0;
constexpr int64_t DYNAMIC_AUGRU_GRAD_GATE_RZH = 1;

// matmul M/N/K维最小对齐粒度：HPad/tbPad按此对齐（host与kernel两侧共用）
constexpr int64_t MM_DIM_ALIGN = 16;
// 3 * HPad is divisible by this length, so recurrent projections have no K tail.
constexpr int64_t MM_RECURRENT_CHUNK = 3 * MM_DIM_ALIGN;

// Weight-gradient reduction masks. A short K does not by itself guarantee
// relative accuracy near cancellation; enabled blocks use compensated merging.
constexpr int64_t MM_CHUNK_DGATE = 1;     // bit0: dgateMM(K=3*HPad)
constexpr int64_t MM_CHUNK_DW_INPUT = 2;  // bit1: dwInputMM(K=TB)
constexpr int64_t MM_CHUNK_DW_HIDDEN = 4; // bit2: dwHiddenMM(K=TB)
constexpr int64_t MM_CHUNK_DX = 8;        // bit3: dxMM(K=3*HPad)

struct DynamicAUGRUGradTilingData {
    // H按MM_DIM_ALIGN对齐后的值（HPad = Ceil(H,MM_DIM_ALIGN)*MM_DIM_ALIGN），workspace中
    // dGi/dGh/hPrev的行距分别为3*HPad/3*HPad/HPad，pad区恒0对matmul无贡献
    int64_t hPad = 0;
    // 每时间步 dh_prev = dgate_h[t] @ w_hidden^T：M=B, N=H, K=3H（B转置）
    TCubeTiling dgateMMParam;
    // dw_input = x^T @ dgate_x：M=I, N=3H, K=TB（A转置）
    TCubeTiling dwInputMMParam;
    // dw_hidden = h_prev^T @ dgate_h：M=H, N=3H, K=TB（A转置）
    TCubeTiling dwHiddenMMParam;
    // dx = dgate_x @ w_input^T：M=TB, N=I, K=3H（B转置）
    TCubeTiling dxMMParam;

    // K轴分块累加：mmChunkK为分块长度（0=全部关闭，启用时对应MM的
    // TCubeTiling按K=mmChunkK生成，kernel按块循环并累加部分和）；mmChunkMask按MM使能
    int64_t mmChunkK = 0;
    int64_t mmChunkMask = 0;

    int64_t timeStep = 0;    // T
    int64_t batchSize = 0;   // B
    int64_t hiddenSize = 0;  // H
    int64_t inputSize = 0;   // I
    int64_t isSeqLength = 0; // 1表示传入seq_length，kernel内在线生成掩码
    int64_t gateOrder = 0;   // 0=zrh 1=rzh

    int64_t bTile = 0;    // 向量阶段单tile行数（batch方向）
    int64_t hTile = 0;    // 向量阶段单tile列数（hidden方向，不超过H）
    int64_t ubLength = 0; // 单个fp32 UB缓冲的元素容量

    // 单tile流水开关：fp32且单tile时置1，kernel侧BPTT循环改为
    // dgateMM异步IterateAll -> 预取下一时间步操作数 -> WaitIterateAll
    int64_t enablePipeline = 0;

    // db归约内联开关：BPTT循环内把dGi/dGh各槽列和累加进每核UB累加器，
    // 收尾仅对各核partial[blockDim,3H]小表做列归约（替代重读全量dGi/dGh）。
    // 启用条件：扣全部固定占用（含seq_length UB分片）后余量装得下两组[3HPad]累加器及其补偿
    int64_t enableDbInline = 0;

    // db_input/db_hidden按列归约的切分：单核负责的列块宽度（列块由kernel按
    // blockIdx跨核循环覆盖全部列）
    int64_t singleCoreReduceN = 0;
};

#endif // __DYNAMIC_AUGRU_GRAD_TILING_DATA_H__
