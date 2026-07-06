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
 * \file foreach_a_cos_inplace_regbase.h
 * \brief
 */
#ifndef FOREACH_A_COS_INPLACE_REGBASE_H
#define FOREACH_A_COS_INPLACE_REGBASE_H

#include "lib/math/kernel_operator_acos_intf.h"
#include "../../foreach_utils/arch35/foreach_unary_cast_inplace.h"

namespace ForeachACosInplace {
using namespace AscendC;
// In-place foreach acos: x = acos(x), valid input range [-1, 1]. half/float compute directly;
// bfloat16 computes in float (Acos supports only half/float). Init/Compute are shared by the base.
template <typename T, typename Tiling>
class ForeachACosInplaceRegbase
    : public ForeachUnaryCastInplaceRegbase<T, Tiling, ForeachACosInplaceRegbase<T, Tiling>> {
public:
    template <typename U>
    __aicore__ inline void ApplyOp(LocalTensor<U> dst, LocalTensor<U> src, int64_t dataCount)
    {
        Acos(dst, src, static_cast<uint32_t>(dataCount));
    }
};
} // namespace ForeachACosInplace

#endif // FOREACH_A_COS_INPLACE_REGBASE_H
