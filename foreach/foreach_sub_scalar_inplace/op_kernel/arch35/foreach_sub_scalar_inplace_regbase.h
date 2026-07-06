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
 * \file foreach_sub_scalar_inplace_regbase.h
 * \brief
 */
#ifndef FOREACH_SUB_SCALAR_INPLACE_REGBASE_H
#define FOREACH_SUB_SCALAR_INPLACE_REGBASE_H

#include "../../foreach_utils/arch35/foreach_unary_scalar_cast_inplace.h"

namespace ForeachSubScalarInplace {
using namespace AscendC;
// In-place foreach sub-scalar: x = x - scalar, implemented as Adds(x, -scalar). half/bfloat16 compute
// in float via Cast, float/int32 subtract directly. Scalar read / cast buffer / dispatch shared by base.
template <typename T, typename ScalarT, typename Tiling>
class ForeachSubScalarInplaceRegbase
    : public ForeachUnaryScalarCastInplaceRegbase<T, ScalarT, Tiling,
                                                  ForeachSubScalarInplaceRegbase<T, ScalarT, Tiling>> {
public:
    template <typename U>
    __aicore__ inline void ApplyScalarOp(LocalTensor<U> dst, LocalTensor<U> src, U scalar, int64_t dataCount)
    {
        Adds(dst, src, -scalar, dataCount);
    }
};
} // namespace ForeachSubScalarInplace

#endif // FOREACH_SUB_SCALAR_INPLACE_REGBASE_H
