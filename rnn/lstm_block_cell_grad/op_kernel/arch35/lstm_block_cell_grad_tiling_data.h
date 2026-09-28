/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Shared TilingData summary struct for the LSTMBlockCellGrad operator on
 * arch35 (13 fields = 11 x int64_t + 2 x int32_t, in declaration order).
 *
 * ALL 4 tilingKey branches (2 dtype x 2 use_peephole, keys {0, 1, 4, 5}) share
 * this single struct — no branch-private fields: dtype / use_peephole
 * differences are carried by the tilingKey template parameters
 * (lstm_block_cell_grad_struct.h), never by TilingData.  Included by BOTH host
 * tiling code and the device kernel entry (GET_TILING_DATA_WITH_STRUCT), so
 * both sides see the same layout.
 *
 * Memory layout note: the struct must be trivially copyable (plain int64_t /
 * int32_t fields, no pointers, no virtual methods) because it is serialised
 * into a raw byte buffer that is passed from host to device via global memory
 * and deserialised on the device side (memcpy of
 * sizeof(LSTMBlockCellGradTilingData) = 96 bytes, 8-byte aligned).
 */

#pragma once
#include <cstdint>

struct LSTMBlockCellGradTilingData {
    // —— 全局 shape（int64 支持动态 shape 与大 shape；B/C 允许 0——空张量正向，
    //    由 kernel 首步短路承载）——
    int64_t batchSize;  // B：batch 维长度（≥0 任意整数、无 2 幂对齐假设；==0 空张量正向：
                        //   (B,·) 输出 0 元素免写 + 3×(H,) 零填充；batch=1 时归约轴长 1）
    int64_t cellSize;   // C：cell 维长度（≥0 任意整数；==0 空张量正向：全输出 0 维早退；
                        //   窥孔权重向量宽 / dicfo 单列块宽 / workspace partial 行宽）
    int64_t numInputs;  // N：x 列宽（仅 shape 校验用：w 行数 = N + C；kernel 计算链不消费）
    int64_t dicfoWidth; // C4 = 4*C：dicfo / b 列宽（icfo=[i,c,f,o] 四列块，恒为 4 的倍数）

    // —— 多核切分：大小核均衡（切分轴随分支：peep=false 切 batch 行、
    //    peep=true 切 cell 列，字段为双语义，
    //    具体语义由 tilingKey 承载）——
    int32_t usedCoreNum; // 实际参与核数 = min(AICore 数, max(peep=true ? C : B, 1))；= SetBlockDim 值；
                         //   单元素轴时为 1；空张量（B==0/H==0）host 短路填 1（严禁 0）
    int32_t bigCoreCnt; // 前 bigCoreCnt 个核各处理 bigCoreCols 个单元（行/列），其余核处理 smallCoreCols 个
    int64_t bigCoreCols;   // 大核单元数 = CeilDiv(splitAxis, usedCoreNum)；peep=true: cell 列数、
                           //   peep=false: batch 行数；splitAxis = peep=true ? C : B
    int64_t smallCoreCols; // 小核单元数 = FloorDiv(splitAxis, usedCoreNum)；bigCoreCnt = splitAxis mod usedCoreNum
                           //   （可为 0）；peep=true: cell 列数、peep=false: batch 行数

    // —— UB 切分：bTile × cTile 二维 tile（大 shape 分批多轮搬运的核心参数）——
    int64_t bTile;      // 单轮 batch 行数（valid，1 ≤ bTile ≤ B；由 UB 预算按 dtype 分支求解）
    int64_t cTile;      // 单轮 cell 列数（valid，1 ≤ cTile ≤ C；主块轮列宽）
    int64_t cTileAlign; // cTile 的 UB padded 对齐值 = CeilAlign(cTile, 256B/sizeof(T))（FP32:64 / FP16:128
                        //   元素）——UB 行步长 / VF repeat 粒度，保证 256B 对齐
    int64_t cTileNum;   // cell 方向轮数 = CeilDiv(C, cTile)（全局一致：cell 不切核）
    int64_t cTileLast;  // cell 末轮 valid 列数 = C − (cTileNum−1)×cTile（非 2 幂 C 的尾块，< cTile 时
                        //   按 valid 长度搬运 / 输出，padded 区 garbage 隔离（恒 valid 长度写回、
                        //   禁止额外 cell 方向清零），不向接口暴露对齐）
    // 注：batch 方向轮数不设字段——peep=false 各核行数不同（big/small），peep=true 全 batch 一致
    //   （列切分）；kernel 现算 CeilDiv(本核行数, bTile)
};
