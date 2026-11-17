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
 * \file conv2d_v2_tiling_data.h
 * \brief
 */

#ifndef CONV2D_V2_TILING_DATA_H
#define CONV2D_V2_TILING_DATA_H

namespace Ops {
namespace NN {
namespace Conv2dV2 {

#pragma pack(push, 8)

struct Conv2DMNFullLoadKPartialTiling {
    uint64_t cout = 0;
    uint64_t hout = 0;
    uint64_t wout = 0;
    uint64_t cin = 0;
    uint64_t hin = 0;
    uint64_t win = 0;
    uint64_t kh = 0;
    uint64_t kw = 0;
    uint32_t batchDim = 0;
    uint32_t nDim = 0;
    uint32_t hoDim = 0;
    uint32_t batch = 0;
    uint32_t singleCoreBatch = 0;
    uint32_t singleCoreCi = 0;
    uint32_t singleCoreCo = 0;
    uint32_t singleCoreHo = 0;
    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t dilationH = 0;
    uint32_t dilationW = 0;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t kAL1 = 0;
    uint32_t kBL1 = 0;
    uint32_t kL0 = 0;
    uint32_t nBL1 = 0;
    uint32_t cinAInCore = 0;
    uint32_t cinATailInCore = 0;
    uint32_t cinBInCore = 0;
    uint32_t cinBTailInCore = 0;
    uint8_t hf32Enable = 0;
    uint8_t hf32TransMode = 0;
    uint8_t hasBias = 0;
    int8_t offsetx = 0;
    uint32_t bUbNStep = 0;
    uint32_t bUbKStep = 0;
};

#pragma pack(pop)
} // namespace Conv2dV2
} // namespace NN
} // namespace Ops

#endif // CONV2D_V2_TILING_DATA_H
