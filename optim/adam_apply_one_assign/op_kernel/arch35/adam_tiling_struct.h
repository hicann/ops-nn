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
 * \file adam_tiling_struct.h
 * \brief 
 */
#pragma once
#include <cstdint>

// === 算子特定常量 — 新算子必须修改 ===
// 这些值随算子而变，抄代码时必须替换为你的算子的实际值
constexpr int64_t kMaxInputSlots  = 10;  // 本算子最大输入数
constexpr int64_t kMaxOutputSlots = 3;   // 本算子最大输出数
constexpr int64_t kPhysNodes      = 5;   // 物理存活节点 P（= TBuf 槽位数）

struct SplitResult {
    int64_t axis;
    int64_t a_i;
    int64_t a_o;
    int64_t a_i_tail;
};

struct MultiCoreResult {
    int64_t num_cores;
    int64_t total_tiles;
    int64_t tiles_main;
    int64_t cores_tail;
};

template<int64_t kRank>
struct AdamTilingData {
    SplitResult     split;
    MultiCoreResult multicore;
    int64_t         rank;            // 实际 rank (1~8)，Kernel 运行期读取
    int64_t         per_buf_bytes;   // UB/P 向下对齐 32B，Kernel 用此初始化 TBuf
    int64_t         max_bro_shape[kRank];
    int64_t         num_inputs;
    int64_t         num_outputs;
    int64_t         input_shapes [kMaxInputSlots][kRank];
    int64_t         input_strides[kMaxInputSlots][kRank];
    int64_t         output_shapes[kMaxOutputSlots][kRank];
    int64_t         output_strides[kMaxOutputSlots][kRank];
};
