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
 * \file arg_max_grad_tiling_data.h
 * \brief
 */
#ifndef ARG_MAX_GRAD_TILING_DATA_H
#define ARG_MAX_GRAD_TILING_DATA_H

#include <cstdint>

struct ArgMaxGradArch35TilingData {
    int64_t outer = 0;        // ∏ dims[0..dimension-1]
    int64_t dimSize = 0;      // D = dims[dimension], 被选择的轴长
    int64_t inner = 0;        // ∏ dims[dimension+1..rank-1]
    int64_t totalElems = 0;   // outer * D * inner, 输出总元素数
    int64_t elemsPerCore = 0; // 每核负责的元素数(按 32B 对齐, 保证跨核不共享搬运块)
    int64_t colsPerChunk = 0; // 单次驻留 UB 的元素数(按 VRegSize/sizeof(var dtype) 对齐)
    // 各 UB buffer 的字节数由 host 依 UB 容量与向量寄存器宽度一次算准, 内核直接透传给
    // InitBuffer, 不得再做二次对齐/补齐 —— 内核侧二次对齐会让实际占用偏离 host 的预算,
    // 且各 buffer 起点会偏离向量寄存器整宽边界(实测 errcode 340: VEC 访问 UB 地址未对齐)。
    int64_t tBufBytes = 0;    // T 域单块字节数(var / out / updates 各一块)
    int64_t i32BufBytes = 0;  // int32 域单块字节数(assist / indices 各一块)
    int64_t maskBufBytes = 0; // 掩码缓冲字节数
    int64_t selBufBytes = 0;  // int8 借道 half 的三块暂存合计字节数; 非 int8 为 0
};

#endif // ARG_MAX_GRAD_TILING_DATA_H
