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
 * \file arg_max_grad_d_tiling_data.h
 * \brief
 */
#ifndef ARG_MAX_GRAD_D_TILING_DATA_H
#define ARG_MAX_GRAD_D_TILING_DATA_H

#include <cstdint>

struct ArgMaxGradDArch35TilingData {
    int64_t outer = 0;        // ∏ dims[0..dimension-1]
    int64_t dimSize = 0;      // D = dims[dimension], 被选择的轴长
    int64_t inner = 0;        // ∏ dims[dimension+1..rank-1]
    int64_t totalElems = 0;   // outer * D * inner, 输出总元素数
    int64_t elemsPerCore = 0; // 每核负责的元素数(按 32B 对齐, 保证跨核不共享搬运块)
    int64_t colsPerChunk = 0; // 单次驻留 UB 的元素数(VL 对齐)
};

#endif // ARG_MAX_GRAD_D_TILING_DATA_H
