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
 * \file foreach_unary_cast_inplace.h
 * \brief Shared intermediate base for in-place unary foreach ops whose bfloat16 path computes in
 *        float (e.g. acos, log). A leaf only supplies ApplyOp(); Init/Compute are shared here.
 */
#ifndef FOREACH_UNARY_CAST_INPLACE_H
#define FOREACH_UNARY_CAST_INPLACE_H

#include "foreach_regbase_unary.h"

using namespace AscendC;

// CRTP intermediate: Derived supplies ApplyOp(dst, src, dataCount) (the unary math op).
// half/float compute directly; bfloat16 casts to float, applies, then casts back.
template <typename T, typename Tiling, typename Derived>
class ForeachUnaryCastInplaceRegbase : public ForeachRegbaseUnary<T, Tiling, Derived>
{
public:
    using Base = ForeachRegbaseUnary<T, Tiling, Derived>;
    using Base::Process;
    __aicore__ inline ForeachUnaryCastInplaceRegbase() : Base(static_cast<Derived&>(*this)){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(inputs, outputs, workspace, tilingData, tPipe);
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            tPipe->InitBuffer(castBuf_, tilingData->inputsTensorUbSize * sizeof(float));
            castLocal_ = castBuf_.template Get<float>();
        }
    }

    __aicore__ inline void Compute(LocalTensor<T> inLocal, LocalTensor<T> outLocal, int64_t dataCount)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            Cast(castLocal_, inLocal, RoundMode::CAST_NONE, dataCount);
            static_cast<Derived*>(this)->ApplyOp(castLocal_, castLocal_, dataCount);
            Cast(outLocal, castLocal_, RoundMode::CAST_RINT, dataCount);
        } else {
            static_cast<Derived*>(this)->ApplyOp(outLocal, inLocal, dataCount);
        }
    }

protected:
    TBuf<TPosition::VECCALC> castBuf_;
    LocalTensor<float> castLocal_;
};

#endif // FOREACH_UNARY_CAST_INPLACE_H
