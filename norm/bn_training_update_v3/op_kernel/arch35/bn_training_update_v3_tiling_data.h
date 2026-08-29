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
 * \file bn_training_update_v3_tiling_data.h
 * \brief tiling data struct shared by host tiling and arch35 kernel（ND/NHWC 双路径字段）
 *
 * numRecip      = 1 / (N * R)                       （host fp64 计算后舍入 fp32）
 * batchVarScaler = num / (num - 1)（num==1 时 0.0，host fp64 计算后舍入 fp32）
 * save_mean[c]      = sum[c] * numRecip
 * save_variance[c]  = square_sum[c] * numRecip - save_mean[c]^2
 * batch_mean[c]     = save_mean[c]；reserve_1[c] = save_mean[c]
 * batch_variance[c] = save_variance[c] * batchVarScaler（无偏）；reserve_2[c] = save_variance[c]（有偏）
 * multiplier[c]  = scale[c] / sqrt(save_variance[c] + epsilon)
 * addend[c]      = offset[c] - multiplier[c] * save_mean[c]
 * y[n,c,r]       = multiplier[c] * x[n,c,r] + addend[c]
 *
 * 计算路径（tilingKey 恒 0，kernel 按 isNhwc 运行时分发）：
 * - ND/NCHW（isNhwc=0）：[N,C,R...] plane 连续，plane p 属于 channel p%C，
 *   per-channel 仿射系数广播。
 * - NHWC（isNhwc=1）：x 任意 rank≥2、C=最后一维、rows=numel/C=num（numRecip 语义同 N*R）。
 *   内存完全连续；向量访存 32B 对齐约束 ⇒ 系数切片仅能取 64 对齐整块，三路径：
 *     Flat（nhwcPath=1）：C%64==0 且 C≤12288。静态 pattern=coeff 本身
 *       （pattern[j]=coeff[j%C] 周期恰为 C），flat 逐 64 向量按 v mod (C/64)
 *       取 pattern 向量；chunk staging 落址恒 64 对齐。
 *     Stream（nhwcPath=2）：C%64==0 且 C>12288，pattern 放不下 UB，退化为驻留
 *       12288 元 chunk 环滑动重载。
 *     Rows（nhwcPath=3）：C%64≠0 且整行预算内（fp32 C≲9100/fp16≲13700）。行距 pitch（64 元素
 *       对齐）的 UB tile，逐行 1D DataCopyPad；行内按 64-chunk 取 coeff 无旋转连续段（天然对齐），
 *       全量系数一次驻留。
 *     RowsWindowed（nhwcPath=4）：C%64≠0 且整行预算外（odd-C 无上限）。c 窗口外层 × 行内层：
 *       系数窗 W 元从任意通道偏移按 64 对齐直算重建（无拷贝拼接，规避 VEC 340），每窗流式处理
 *       本核全部行的对应段；每核系数计算总量仍 C/64（每通道恰好一次），UB 占用与 C 无关。
 *
 * 统一切分模型：units 个独立 plane 按 unitCores 均分（former/latter），plane 不足时
 * plane 内 inner 维再切 innerCores 份；blockIdx = unitIdx * innerCores + innerIdx。无 workspace。
 * NHWC 下 plane 语义随路径变化（Flat/Stream=一个 64 元向量块、Rows=一行），机制照用。
 */

#ifndef BN_TRAINING_UPDATE_V3_TILING_DATA_H
#define BN_TRAINING_UPDATE_V3_TILING_DATA_H

#include <cstdint>

struct BNTrainingUpdateV3TilingData {
    int64_t numN;      // N（kernel 不直接消费：units/numC 已含；保留供 host 日志与调试核对）
    int64_t numC;      // C（ND：dim1；NHWC：最后一维）
    int64_t innerSize; // R = prod(d2:)（ND）；NHWC-Flat/Stream=64（向量宽），Rows=1
    int64_t units;     // plane 数（ND：N*C；NHWC：rows 或向量块总数）
    int64_t unitCores; // plane 维切分核数（kernel 由 blockIdx/formerCoreNum 反推，保留供 host 日志）
    int64_t formerCoreNum; // 前 formerCoreNum 核每核 formerUnits 个 plane，其余 latterUnits 个
    int64_t formerUnits;
    int64_t latterUnits;
    int64_t innerCores;   // plane 内 inner 维切分份数（plane < 核数且 inner 够大时 >1）
    int64_t innerPerCore; // 每份的 inner 长度，末尾截断
    int64_t ubTileSize;   // UB tile 元素数（VL=64 对齐；Rows 路径复用为 tileRows）
    int64_t isNhwc;       // 0=ND/NCHW plane 路径；1=NHWC 路径（kernel Process 入口分发）
    int64_t nhwcPath; // NHWC 内部分派：1=Flat（静态 pattern）2=Stream（窗口驻留）3=Rows（整行快路径）
                      // 4=RowsWindowed（odd-C 无上限的 c 窗口流式）
    float epsilon;  // 加到 save_variance 上再开方
    float numRecip; // 1/(N*R)，fp64 计算后舍入 fp32（对齐 A2 TBE tvm.const 语义）
    float batchVarScaler; // num/(num-1)（num==1 时 0.0），fp64 计算后舍入 fp32（对齐 A2 TBE python float 语义）；
                          // batch_variance = save_variance * batchVarScaler（无偏估计）
    float reserved; // 结构体 8B 对齐填充
};

#endif // BN_TRAINING_UPDATE_V3_TILING_DATA_H
