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
 * \file mse_loss_grad_v2_tiling_data.h
 * \brief TilingData shared by host tiling and device kernel on arch35: broadcast
 *        boilerplate fields plus the operator-specific folded coefficient `cof`.
 *        RANK=4/8 instantiate the same struct, only the array dims differ.
 *
 *        Field units: split.aI/aITail, maxBroShape, shapes/strides are element
 *        counts; multicore fields are core/tile counts; perBufBytes is bytes;
 *        rank is a dimension count; cof is an fp32 coefficient.
 */

#ifndef MSE_LOSS_GRAD_V2_TILING_DATA_H_
#define MSE_LOSS_GRAD_V2_TILING_DATA_H_
#pragma once
#include <cstdint>

// === 算子特定常量 ===
constexpr int64_t MAX_INPUT_SLOTS = 3;  // 输入张量数（predict/label/dout）
constexpr int64_t MAX_OUTPUT_SLOTS = 1; // 输出张量数（y）
constexpr int64_t
    PHYS_NODES = 4; // 物理存活节点 P = TBuf 槽位数：
                    //   3 输入 TBuf + 1 输出 TBuf 峰值同驻（VF 寄存器链中间值不落 UB）；
                    //   峰值出现在三输入 CopyIn 完成后至结果写出、输入 buffer 未跨 tile 复用释放时（保守无串行化复用）

struct SplitResult {
    int64_t axis;   // UB 切分轴
    int64_t aI;     // 内轴 tile 大小（元素数）
    int64_t aO;     // 外轴 tile 数
    int64_t aITail; // 末块大小（元素数）
};

struct MultiCoreResult {
    int64_t numCores;   // 参与计算的核数
    int64_t totalTiles; // tile 总数
    int64_t tilesMain;  // 每核主 tile 数
    int64_t coresTail;  // 多处理一个 tile 的核数
};

// 使用场景：RANK ∈ {4, 8} 两档（effective rank ≤ 4 用 MseLossGradV2TilingData<4>，5–8 用 <8>）
template <int64_t RANK>
struct MseLossGradV2TilingData {
    // —— 公共字段（Broadcast 范式模板原样）——
    SplitResult split;         // UB 切分结果（来源：FindSplitAxis）
    MultiCoreResult multicore; // 多核切分结果（来源：MultiCoreSplit）
    int64_t rank;              // 实际有效 rank（PadAndSqueeze 去 1 补 1 后，1~8；Kernel 运行期读取）
    int64_t perBufBytes;       // 单 buffer 字节数 = (UB/PHYS_NODES) & ~31（32B 向下对齐）
    int64_t maxBroShape[RANK]; // broadcast 上界 shape（UB/多核切分的统一坐标系）
    int64_t numInputs;         // 输入张量数 = 3
    int64_t numOutputs;        // 输出张量数 = 1
    int64_t inputShapes[MAX_INPUT_SLOTS][RANK];  // 各输入补 1 后的 shape
    int64_t inputStrides[MAX_INPUT_SLOTS][RANK]; // 各输入 GM stride（broadcast 轴 = 0，随路广播）
    int64_t outputShapes[MAX_OUTPUT_SLOTS][RANK]; // 输出 shape（= broadcast 结果 = predict 形状，InferShapeDtype.md）
    int64_t outputStrides[MAX_OUTPUT_SLOTS][RANK]; // 输出 GM stride
    // —— 扩充字段（本算子特有）——
    float cof; // host 预折叠梯度系数：reduction="mean" → 2/numel(predict)，
               //   "none"/"sum" → 2.0（fp32 折叠；kernel 内无浮点除法）
};
#endif // MSE_LOSS_GRAD_V2_TILING_DATA_H_
