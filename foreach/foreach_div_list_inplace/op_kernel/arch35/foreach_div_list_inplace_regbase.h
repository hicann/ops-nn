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
 * \file foreach_div_list_inplace_regbase.h
 * \brief
 */
#ifndef FOREACH_DIV_LIST_INPLACE_REGBASE_H
#define FOREACH_DIV_LIST_INPLACE_REGBASE_H

#include "../../foreach_utils/arch35/foreach_regbase_binary.h"

namespace ForeachDivListInplace {
using namespace AscendC;
// In-place foreach div-list: y(=x1) = x1 * x2. Reuses the shared ForeachRegbaseBinary base
// (two input tensor lists); the apt entry feeds x1 as both the first input and the output.
template <typename T, typename Tiling>
class ForeachDivListInplaceRegbase : public ForeachRegbaseBinary<T, Tiling, ForeachDivListInplaceRegbase<T, Tiling>> {
public:
    using Base = ForeachRegbaseBinary<T, Tiling, ForeachDivListInplaceRegbase<T, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachDivListInplaceRegbase() : Base(*this){};
    __aicore__ inline void Init(GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR outputs, GM_ADDR workspace,
                                const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(tensor1, tensor2, outputs, workspace, tilingData, tPipe);
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            tPipe->InitBuffer(castBuf1_, tilingData->inputsTensorUbSize * sizeof(float));
            tPipe->InitBuffer(castBuf2_, tilingData->inputsTensorUbSize * sizeof(float));
            cast1_ = castBuf1_.template Get<float>();
            cast2_ = castBuf2_.template Get<float>();
        }
    }

    // dav-3510 vdiv does not support half/bfloat16, so those compute in float.
    __aicore__ inline void Compute(LocalTensor<T> in1Local, LocalTensor<T> in2Local, LocalTensor<T> outLocal,
                                   int64_t dataCount)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            Cast(cast1_, in1Local, RoundMode::CAST_NONE, dataCount);
            Cast(cast2_, in2Local, RoundMode::CAST_NONE, dataCount);
            Div(cast1_, cast1_, cast2_, dataCount);
            Cast(outLocal, cast1_, RoundMode::CAST_RINT, dataCount);
        } else {
            Div(outLocal, in1Local, in2Local, dataCount);
        }
    }

private:
    TBuf<TPosition::VECCALC> castBuf1_;
    TBuf<TPosition::VECCALC> castBuf2_;
    LocalTensor<float> cast1_;
    LocalTensor<float> cast2_;
};
} // namespace ForeachDivListInplace

#endif // FOREACH_DIV_LIST_INPLACE_REGBASE_H
