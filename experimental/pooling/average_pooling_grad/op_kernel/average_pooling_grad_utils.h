/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AVERAGE_POOLING_GRAD_UTILS_H
#define AVERAGE_POOLING_GRAD_UTILS_H

#include "kernel_operator.h"

namespace NsAveragePoolingGrad {

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;

struct TileTask {
    uint32_t nc;
    uint32_t batch;
    uint32_t channel;
    uint32_t hStart;
    uint32_t hEnd;
    uint32_t wStart;
    uint32_t wEnd;
};

struct OutputCacheRange {
    int32_t ohStart;
    int32_t ohEnd;
    int32_t owStart;
    int32_t owEnd;
    uint32_t cacheH;
    uint32_t cacheW;
    uint32_t cacheStride;
};

__aicore__ inline int32_t FloorDiv(int32_t a, int32_t b)
{
    if (b == 0) {
        return 0;
    }
    if (a >= 0) {
        return a / b;
    }
    return -((-a + b - 1) / b);
}

__aicore__ inline int32_t ScalarMax(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

__aicore__ inline int32_t ScalarMin(int32_t a, int32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int32_t PStart(int32_t pos, uint32_t pad, uint32_t kernel, uint32_t stride)
{
    return FloorDiv(pos + static_cast<int32_t>(pad) - static_cast<int32_t>(kernel),
        static_cast<int32_t>(stride)) + 1;
}

__aicore__ inline int32_t PEnd(int32_t pos, uint32_t pad, uint32_t stride)
{
    return FloorDiv(pos + static_cast<int32_t>(pad), static_cast<int32_t>(stride)) + 1;
}

__aicore__ inline int32_t ClampStart(int32_t start)
{
    return ScalarMax(start, 0);
}

__aicore__ inline int32_t ClampEnd(int32_t end, int32_t limit)
{
    return ScalarMin(end, limit);
}

__aicore__ inline uint32_t AlignUpElements(uint32_t elements)
{
    constexpr uint32_t align = ALIGN_BYTES / sizeof(float);
    return ((elements + align - 1) / align) * align;
}

} // namespace NsAveragePoolingGrad

#endif // AVERAGE_POOLING_GRAD_UTILS_H
