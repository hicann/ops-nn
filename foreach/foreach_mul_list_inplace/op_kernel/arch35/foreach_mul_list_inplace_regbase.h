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
 * \file foreach_mul_list_inplace_regbase.h
 * \brief
 */
#ifndef FOREACH_MUL_LIST_INPLACE_REGBASE_H
#define FOREACH_MUL_LIST_INPLACE_REGBASE_H

#include "../../foreach_utils/arch35/foreach_regbase_binary.h"

namespace ForeachMulListInplace {
using namespace AscendC;
// In-place foreach mul-list: y(=x1) = x1 * x2. Reuses the shared ForeachRegbaseBinary base
// (two input tensor lists); the apt entry feeds x1 as both the first input and the output.
template <typename T, typename Tiling>
class ForeachMulListInplaceRegbase : public ForeachRegbaseBinary<T, Tiling, ForeachMulListInplaceRegbase<T, Tiling>> {
public:
    using Base = ForeachRegbaseBinary<T, Tiling, ForeachMulListInplaceRegbase<T, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachMulListInplaceRegbase() : Base(*this){};
    __aicore__ inline void Init(GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR outputs, GM_ADDR workspace,
                                const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(tensor1, tensor2, outputs, workspace, tilingData, tPipe);
    }

    __aicore__ inline void Compute(LocalTensor<T> in1Local, LocalTensor<T> in2Local, LocalTensor<T> outLocal,
                                   int64_t dataCount)
    {
        Mul(outLocal, in1Local, in2Local, dataCount);
    }
};
} // namespace ForeachMulListInplace

#endif // FOREACH_MUL_LIST_INPLACE_REGBASE_H
