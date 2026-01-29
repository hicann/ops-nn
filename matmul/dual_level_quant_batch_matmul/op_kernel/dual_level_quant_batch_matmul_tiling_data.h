/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_v2_tiling_data.h
 * \brief
 */

#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_KERNEL_TILING_DATA_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_KERNEL_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"


// tiling data注意8B对齐，尽量手动添加对齐的保留字段
#pragma pack(push, 8)
struct DualLevelQuantBatchMatmulBasicTilingData {
    uint32_t mL1Size = 256; // 不是基本块，也不是单核处理的块大小，而是每次载L1时的基本块大小
    uint32_t nL1Size = 256;
    uint32_t kL1Size = 512;
};
#pragma pack(pop)

#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_KERNEL_TILING_DATA_H