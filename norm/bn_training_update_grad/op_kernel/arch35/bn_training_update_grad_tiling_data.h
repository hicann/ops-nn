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
 * \file bn_training_update_grad_tiling_data.h
 * \brief tiling data struct shared by host tiling and arch35 kernel（ND + NHWC 双路径字段）
 *
 * rstd[c]        = 1 / sqrt(batch_variance[c] + epsilon)
 * diff_scale[c]  = sum_{n,r} grads[n,c,r] * (x[n,c,r] - batch_mean[c]) * rstd[c]
 * diff_offset[c] = sum_{n,r} grads[n,c,r]
 *
 * 单条 tilingKey=0，kernel 按 isNhwc 运行时分发（dtype 由编译期三二进制覆盖）：
 *
 * ND 路径（isNhwc=0，原逻辑逐位不变）：ND/NCHW [N,C,R...] plane 连续，channel 主切分
 * （每 channel 的完整归约由唯一归属核完成，零核间通信、无 workspace）。
 * 切分模型：C 按 channelCores 均分（前 cFormerCoreNum 核每核 cFormerLen 个 channel，
 * 其余 cLatterLen 个）；blockIdx = cRangeIdx。核内 channel 范围再按 cLenCap 分 chunk、
 * R 按 sliceR 分片、N 按 rowsPerTile 分 tile。
 *
 * NHWC 路径（isNhwc=1，含 ND C==1 大规模 / ND R==1 巨 C 两条 reroute——布局与 NHWC 同构）：
 * [rows, C] 行主序（C=最后一维，rows=numel/C）。cLenCap 复用为 c 窗口宽 W（恒 64 对齐，
 * 保证 UB 行基址 32B 对齐、行尾 64 元向量无掩码读不越 pitch）；sliceR 复用为每 tile 行数
 * tileRows；rowsPerTile 恒 1（占位不消费）。两种切分（nhwcSplitMode）：
 *   1 = channelSplit：C 按 channelCores=min(C,coreNum) 均分（cFormer* 语义为通道段），
 *       每核独占通道段扫全部行，零通信零 workspace；段长 > Wmax 时 c 窗口化（C 无上限）。
 *   2 = rowSplit：rows 按 coreNum 均分（cFormer* 语义为行段），每核独占行段×全部 C，
 *       0 号核覆盖写输出 + SyncAll + 其余核浮点原子加直写（零 workspace；原子和顺序
 *       不定，误差 O(eps)，部分和数量 = 核数 ≤64）。
 */

#ifndef BN_TRAINING_UPDATE_GRAD_TILING_DATA_H
#define BN_TRAINING_UPDATE_GRAD_TILING_DATA_H

#include <cstdint>

struct BNTrainingUpdateGradTilingData {
    int64_t numN; // ND: N（dim0）；NHWC: rows = numel/C（kernel 不直接消费，保留供 host 日志与调试核对）
    int64_t numC;      // ND: C（dim1）；NHWC: C（最后一维）
    int64_t innerSize; // ND: R = prod(d2:)；NHWC: 恒 1
    int64_t channelCores; // ND: channel 切分核数；NHWC: blockDim（kernel 不直接消费：blockDim 已定，保留供核对）
    int64_t
        cFormerCoreNum; // ND: 前 cFormerCoreNum 核每核 cFormerLen 个 channel；NHWC: 切分维（mode1=通道段/mode2=行段）
    int64_t cFormerLen;
    int64_t cLatterLen;
    int64_t cLenCap;     // ND: channel chunk 长度上限；NHWC: c 窗口宽 W（64 对齐）
    int64_t sliceR;      // ND: R 维分片长度；NHWC: 每 tile 行数 tileRows
    int64_t rowsPerTile; // ND: 每 tile 的 n 行数上限；NHWC: 恒 1（占位，不消费）
    int64_t isNhwc; // 0=ND 原路径（kernel 不读其余 NHWC 字段）；1=NHWC 统一计算体（含 ND reroute）
    int64_t nhwcSplitMode; // NHWC 切分模式：1=channelSplit（零通信）；2=rowSplit（0核覆盖写+SyncAll+其余核原子加直写）
    float epsilon;  // 加到 batch_variance 上再开方取倒数
    float reserved; // 结构体 8B 对齐填充
};

#endif // BN_TRAINING_UPDATE_GRAD_TILING_DATA_H
