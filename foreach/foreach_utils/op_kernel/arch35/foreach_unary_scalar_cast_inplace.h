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
 * \file foreach_unary_scalar_cast_inplace.h
 * \brief Shared intermediate base for in-place foreach ops of the form x = op(x, scalar) whose
 *        half/bfloat16 path computes in float (e.g. mul_scalar, sub_scalar). A leaf only supplies
 *        ApplyScalarOp(); scalar read, cast buffer and Compute dispatch are shared here.
 */
#ifndef FOREACH_UNARY_SCALAR_CAST_INPLACE_H
#define FOREACH_UNARY_SCALAR_CAST_INPLACE_H

#include "foreach_regbase_unary.h"

using namespace AscendC;

// CRTP intermediate: Derived supplies ApplyScalarOp(dst, src, scalar, dataCount).
// float/int compute directly; half/bfloat16 cast to float, apply, then cast back.
template <typename T, typename ScalarT, typename Tiling, typename Derived>
class ForeachUnaryScalarCastInplaceRegbase : public ForeachRegbaseUnary<T, Tiling, Derived>
{
public:
    using Base = ForeachRegbaseUnary<T, Tiling, Derived>;
    using Base::Process;
    __aicore__ inline ForeachUnaryScalarCastInplaceRegbase() : Base(static_cast<Derived&>(*this)){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(inputs, outputs, workspace, tilingData, tPipe);
        inScalarGM_.SetGlobalBuffer((__gm__ ScalarT*)scalar, 1);
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            scalarVal_ = inScalarGM_.GetValue(0);
            tPipe->InitBuffer(castBuf_, tilingData->inputsTensorUbSize * sizeof(float));
            castLocal_ = castBuf_.template Get<float>();
        } else {
            scalarVal_ = T(inScalarGM_.GetValue(0));
        }
    }

    __aicore__ inline void Compute(LocalTensor<T> inLocal, LocalTensor<T> outLocal, int64_t dataCount)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            Cast(castLocal_, inLocal, RoundMode::CAST_NONE, dataCount);
            static_cast<Derived*>(this)->ApplyScalarOp(castLocal_, castLocal_, float(scalarVal_), dataCount);
            Cast(outLocal, castLocal_, RoundMode::CAST_RINT, dataCount);
        } else {
            static_cast<Derived*>(this)->ApplyScalarOp(outLocal, inLocal, static_cast<T>(scalarVal_), dataCount);
        }
    }

protected:
    GlobalTensor<ScalarT> inScalarGM_;
    TBuf<TPosition::VECCALC> castBuf_;
    LocalTensor<float> castLocal_;
    ScalarT scalarVal_ = 0;
};

#endif // FOREACH_UNARY_SCALAR_CAST_INPLACE_H
