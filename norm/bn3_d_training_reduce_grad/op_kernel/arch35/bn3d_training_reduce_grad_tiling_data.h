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
 * \file bn3d_training_reduce_grad_tiling_data.h
 * \brief BN3DTrainingReduceGrad 的 TilingData 结构定义
 */

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// 公共常量
// ---------------------------------------------------------------------------
constexpr int64_t
    MAX_INPUT_SLOTS = 7; // 输入张量数：grads / x / diff_scale / diff_offset / scale / batch_mean / batch_variance
constexpr int64_t MAX_OUTPUT_SLOTS = 1; // 输出张量数：y
constexpr int64_t PHYS_NODES = 8; // 物理存活节点 P（TBuf 槽位数）：批量 CopyIn 一次性搬入 7 输入
                                  // + 1 个工作槽，使单 tile 内 MTE2→V 仅需 1 次同步

// ---------------------------------------------------------------------------
// SplitResult — UB 切分结果
// ---------------------------------------------------------------------------
struct SplitResult {
    int64_t axis;   // UB 切分轴（effective 坐标系）
    int64_t aI;     // 内轴 tile 大小（元素数）
    int64_t aO;     // 外轴 tile 数
    int64_t aITail; // 末块大小（元素数）
};

// ---------------------------------------------------------------------------
// MultiCoreResult — 多核切分结果
// ---------------------------------------------------------------------------
struct MultiCoreResult {
    int64_t numCores;   // 参与计算的核数
    int64_t totalTiles; // tile 总数
    int64_t tilesMain;  // 每核主 tile 数
    int64_t coresTail;  // 多处理一个 tile 的核数
};

// ---------------------------------------------------------------------------
// BN3DTrainingReduceGradTilingData<RANK> — 主 TilingData
// RANK 为编译期模板参数：两实例结构相同、
// 仅 maxBroShape / inputShapes / inputStrides / outputShapes / outputStrides
// 数组维度按 RANK 展开。perBufElems 不落字段
// ---------------------------------------------------------------------------
template <int64_t RANK>
struct BN3DTrainingReduceGradTilingData {
    // —— 公共字段（Broadcast 范式标准模板） ——
    SplitResult split;         // UB 切分结果
    MultiCoreResult multicore; // 多核切分结果
    int64_t rank;              // 实际有效 rank（补 1 去 1 后）
    int64_t perBufBytes;       // 单 buffer 字节数 = (UB/P) & ~31（来源：P 结论表）
    int64_t maxBroShape[RANK]; // 广播后各维大小（maximumBroShape 坐标系）
    // 注: numInputs/numOutputs 曾作为字段保留, kernel 从不读取(G8 死字段检视),
    //     已移除。本算子槽位数固定 = MAX_INPUT_SLOTS/MAX_OUTPUT_SLOTS。
    int64_t inputShapes[MAX_INPUT_SLOTS][RANK];    // 各输入补 1 后的 shape
    int64_t inputStrides[MAX_INPUT_SLOTS][RANK];   // 各输入 GM stride（broadcast 轴 = 0）
    int64_t outputShapes[MAX_OUTPUT_SLOTS][RANK];  // 各输出 shape
    int64_t outputStrides[MAX_OUTPUT_SLOTS][RANK]; // 各输出 GM stride
    // —— 扩充字段（本算子特有） ——
    float epsilon; // attr epsilon（NumericalStable：attribute→TilingData）；加在 batch_variance 上的小正数
    int64_t num;   // num = N·D·H·W（除 C 外各维乘积，host 侧 int64 计算）
};
