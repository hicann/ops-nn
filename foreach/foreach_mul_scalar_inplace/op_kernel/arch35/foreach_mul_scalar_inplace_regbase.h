/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file foreach_mul_scalar_inplace_regbase.h
 * \brief
 */
#ifndef FOREACH_MUL_SCALAR_INPLACE_REGBASE_H
#define FOREACH_MUL_SCALAR_INPLACE_REGBASE_H

#include "../../foreach_utils/arch35/foreach_unary_scalar_cast_inplace.h"

namespace ForeachMulScalarInplace {
using namespace AscendC;
// In-place foreach mul-scalar: x = x * scalar. half/bfloat16 compute in float via Cast,
// float/int32 multiply directly. Scalar read / cast buffer / Compute dispatch are shared by the base.
template <typename T, typename ScalarT, typename Tiling>
class ForeachMulScalarInplaceRegbase
    : public ForeachUnaryScalarCastInplaceRegbase<T, ScalarT, Tiling,
          ForeachMulScalarInplaceRegbase<T, ScalarT, Tiling>>
{
public:
    template <typename U>
    __aicore__ inline void ApplyScalarOp(LocalTensor<U> dst, LocalTensor<U> src, U scalar, int64_t dataCount)
    {
        Muls(dst, src, scalar, dataCount);
    }
};
} // namespace ForeachMulScalarInplace

#endif // FOREACH_MUL_SCALAR_INPLACE_REGBASE_H
