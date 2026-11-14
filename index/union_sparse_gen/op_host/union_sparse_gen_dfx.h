/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_DFX_H
#define UNION_SPARSE_GEN_DFX_H

#include <cstddef>
#include <string>

namespace UnionSparseGenDfx {
constexpr const char* kInputNames[] = {"sparseBlockIdx", "sparseBlockCount", "cuSqlenQ",
                                       "cuSqlenKv",      "seqUsedQ",         "seqUsedKv"};

template <typename Shape>
inline std::string ShapeText(const Shape& shape)
{
    std::string text = "[";
    for (std::size_t i = 0; i < shape.GetDimNum(); ++i) {
        text += (i == 0 ? "" : ",") + std::to_string(shape.GetDim(i));
    }
    return text + "]";
}

} // namespace UnionSparseGenDfx
#endif
