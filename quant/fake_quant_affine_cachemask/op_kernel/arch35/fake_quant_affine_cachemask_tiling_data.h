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
 * \file fake_quant_affine_cachemask_tiling_data.h
 * \brief TilingData 结构体（PT / PC / PC_NDDMA / PH 共享 15 字段，套用 static-quant reference）
 */
#ifndef FAKE_QUANT_AFFINE_CACHEMASK_TILING_DATA_H
#define FAKE_QUANT_AFFINE_CACHEMASK_TILING_DATA_H

#include <cstdint>

struct FakeQuantAffineCachemaskTilingDataArch35 {
    // === 总览 ===
    int64_t numCore         = 0;   // 实际开核数（kernel `if (blockIdx >= numCore) return;`）
    int64_t mode            = 0;   // 0=PT, 1=PC, 2=PC_NDDMA, 3=PH（占位）

    // === 合轴后 shape ===
    int64_t dim0            = 1;   // PT=1; PC=∏前 n-1 轴; PH=∏前 axis 轴
    int64_t dim1            = 0;   // PT=∏xShape; PC=x[-1]; PH=x[axis]
    int64_t dim2            = 1;   // PT=1; PC=1; PH=∏后 axis 轴

    // === 切核 ===
    int64_t blockAxis       = 0;   // PT=1; PC ∈ {0,1}; PH ∈ {0,1,2}
    int64_t blockUnion      = 1;   // PT/PC=1; PH=1 / actCore/dim0 / actCore/(dim0×dim1)
    int64_t blockFactor     = 0;   // 每核处理量（首核 & 中间核）
    int64_t blockTailFactor = 0;   // 最后一核处理量（保证非 0）

    // === UB 切分 ===
    int64_t ubAxis          = 0;   // PT=1; PC ∈ {0,1}; PH ∈ {0,1,2}
    int64_t baseN           = 1;   // PT=1; PC ∈ {1, maxN}; PH ∈ {1, dim1, maxN}
    int64_t baseLen         = 0;   // UB tile 列数

    // === 业务参数 ===
    int64_t hasZeroPoint    = 1;   // 0/1
    int64_t axis            = 0;   // 用户原始 axis（归一化后）
    int64_t quantMin        = -128;
    int64_t quantMax        = 127;
};

#endif // FAKE_QUANT_AFFINE_CACHEMASK_TILING_DATA_H
