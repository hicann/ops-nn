/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_GEOMETRY_H
#define UNION_SPARSE_GEN_GEOMETRY_H
#include <cstdint>
#include <limits>

namespace UnionSparseGenGeometry {
struct Result {
    int64_t groups;
    int64_t capacity;
};

// Shape inference deliberately has no tensor/context argument: only host metadata
// participates. Length values are consumed by the device kernel, never by inference.
inline bool Resolve(int64_t totalQ, int64_t topK, int64_t batch, int64_t width, int64_t groups, int64_t capacity,
                    Result& result, int64_t residual = 0)
{
    const int64_t maximum = std::numeric_limits<int64_t>::max();
    if (totalQ < 0 || topK <= 0 || batch < 0 || width != 8 || (residual != 0 && residual != 1)) {
        return false;
    }
    if (groups <= 0) {
        if (batch > (maximum - totalQ) / (width - 1)) {
            return false;
        }
        groups = (totalQ + (width - 1) * batch) / width;
    }
    if (capacity <= 0) {
        if (topK > maximum / width - residual) {
            return false;
        }
        capacity = (topK + residual) * width;
    }
    result = {groups, capacity};
    return true;
}
} // namespace UnionSparseGenGeometry
#endif
