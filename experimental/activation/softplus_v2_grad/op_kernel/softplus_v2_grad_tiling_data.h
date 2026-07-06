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
 * \file softplus_backward_tiling_data.h
 * \brief tiling data struct
 */

#ifndef SOFTPLUS_BACKWARD_TILING_KERNEL_H
#define SOFTPLUS_BACKWARD_TILING_KERNEL_H
#include <cstdint>

// Kernel端纯结构体：与Host端TilingData字段一一对应，但不依赖任何Host头文件
struct SoftplusV2GradTilingData {
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint64_t finalBigTileNum;
    uint64_t finalSmallTileNum;
    uint64_t tileDataNum;
    uint64_t smallTailDataNum;
    uint64_t bigTailDataNum;
    uint64_t tailBlockNum;
    uint64_t bigprocessDataNum_computes;
    uint64_t smallprocessDataNum_computes;
    uint64_t tailbigprocessDataNum_computes;
    uint64_t tailsmallprocessDataNum_computes;
    uint64_t typeLength;
    float beta;
    float threshold;
};

#endif // SOFTPLUS_BACKWARD_TILING_KERNEL_H