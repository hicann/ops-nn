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
 * \file adaptive_max_pool3d_grad_common.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_GRAD_COMMON_H
#define ADAPTIVE_MAX_POOL3D_GRAD_COMMON_H

#include "../pool_3d_common/arch32/max_pool3d_grad_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AdaptiveMaxPool3DGradComm {
using namespace AscendC;
using namespace MaxPool3DGradCommon;
struct BlockParams : public BlockParamsCommon {
    float coeffD = 0.0;
    float coeffH = 0.0;
    float hwDims = 0.0;
    uint64_t maxKwAlign8 = 0;
    uint64_t maxKwAlign16 = 0;
    uint64_t maxKwAlignDtype = 0;
    uint64_t startD = 0;
    uint64_t startH = 0;
    uint64_t startW = 0;
    uint64_t deltaD = 0;
    uint64_t deltaH = 0;
    uint64_t deltaW = 0;
    uint64_t kernelShape = 0;
};

struct TilingParams : public TilingParamsCommon {
    uint64_t baseNcTail = 0;
    uint64_t maxKd = 0;
    uint64_t maxKh = 0;
    uint64_t maxKw = 0;
    uint64_t maxKdhwLen = 0;
    uint64_t addMode = 0;
};

__aicore__ inline uint64_t FloorDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (uint64_t)(x / y);
}

} // namespace AdaptiveMaxPool3DGradComm

#endif // ADAPTIVE_MAX_POOL3D_GRAD_COMMON_H