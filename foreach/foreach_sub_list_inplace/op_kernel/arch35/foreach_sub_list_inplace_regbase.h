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
 * \file foreach_sub_list_inplace_regbase.h
 * \brief
 */
#ifndef FOREACH_SUB_LIST_INPLACE_REGBASE_H
#define FOREACH_SUB_LIST_INPLACE_REGBASE_H

#include "../../foreach_utils/arch35/foreach_binary_alpha_cast_inplace.h"

namespace ForeachSubListInplace {
using namespace AscendC;
// In-place foreach sub-list: y(=x1) = x1 - alpha * x2. half/bfloat16 compute in float
// (bf16 scalar cast is unsupported on dav-3510); float/int compute directly. Init/Compute shared by base.
template <typename T, typename ScalarT, typename Tiling>
class ForeachSubListInplaceRegbase
    : public ForeachBinaryAlphaCastInplaceRegbase<T, ScalarT, Tiling,
                                                  ForeachSubListInplaceRegbase<T, ScalarT, Tiling>> {
public:
    template <typename U>
    __aicore__ inline void ApplyBinaryOp(LocalTensor<U> dst, LocalTensor<U> a, LocalTensor<U> b, int64_t dataCount)
    {
        Sub(dst, a, b, dataCount);
    }
};
} // namespace ForeachSubListInplace

#endif // FOREACH_SUB_LIST_INPLACE_REGBASE_H
