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
 * \file foreach_binary_alpha_cast_inplace.h
 * \brief Shared intermediate base for in-place foreach ops of the form x1 = op(x1, alpha * x2) whose
 *        half/bfloat16 path computes in float (e.g. add_list, sub_list). A leaf only supplies
 *        ApplyBinaryOp(); alpha read, cast buffers and Compute dispatch are shared here.
 */
#ifndef FOREACH_BINARY_ALPHA_CAST_INPLACE_H
#define FOREACH_BINARY_ALPHA_CAST_INPLACE_H

#include "foreach_regbase_binary.h"

using namespace AscendC;

// CRTP intermediate: Derived supplies ApplyBinaryOp(dst, a, b, dataCount) (Add/Sub of x1 and alpha*x2).
// float/int compute directly; half/bfloat16 cast to float (bf16 scalar cast is unsupported on dav-3510).
template <typename T, typename ScalarT, typename Tiling, typename Derived>
class ForeachBinaryAlphaCastInplaceRegbase : public ForeachRegbaseBinary<T, Tiling, Derived> {
public:
    using Base = ForeachRegbaseBinary<T, Tiling, Derived>;
    using Base::Process;
    __aicore__ inline ForeachBinaryAlphaCastInplaceRegbase() : Base(static_cast<Derived&>(*this)){};
    __aicore__ inline void Init(GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR alpha, GM_ADDR outputs, GM_ADDR workspace,
                                const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(tensor1, tensor2, outputs, workspace, tilingData, tPipe);
        inScalarGM_.SetGlobalBuffer((__gm__ ScalarT*)alpha, 1);
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            alphaFloat_ = float(inScalarGM_.GetValue(0));
            tPipe->InitBuffer(castBuf1_, tilingData->inputsTensorUbSize * sizeof(float));
            tPipe->InitBuffer(castBuf2_, tilingData->inputsTensorUbSize * sizeof(float));
            cast1_ = castBuf1_.template Get<float>();
            cast2_ = castBuf2_.template Get<float>();
        } else {
            alphaVal_ = T(inScalarGM_.GetValue(0));
        }
    }

    __aicore__ inline void Compute(LocalTensor<T> in1Local, LocalTensor<T> in2Local, LocalTensor<T> outLocal,
                                   int64_t dataCount)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            Cast(cast1_, in1Local, RoundMode::CAST_NONE, dataCount);
            Cast(cast2_, in2Local, RoundMode::CAST_NONE, dataCount);
            Muls(cast2_, cast2_, alphaFloat_, dataCount);
            PipeBarrier<PIPE_V>();
            static_cast<Derived*>(this)->ApplyBinaryOp(cast1_, cast1_, cast2_, dataCount);
            Cast(outLocal, cast1_, RoundMode::CAST_RINT, dataCount);
        } else {
            Muls(in2Local, in2Local, alphaVal_, dataCount);
            PipeBarrier<PIPE_V>();
            static_cast<Derived*>(this)->ApplyBinaryOp(outLocal, in1Local, in2Local, dataCount);
        }
    }

protected:
    GlobalTensor<ScalarT> inScalarGM_;
    TBuf<TPosition::VECCALC> castBuf1_;
    TBuf<TPosition::VECCALC> castBuf2_;
    LocalTensor<float> cast1_;
    LocalTensor<float> cast2_;
    float alphaFloat_ = 0;
    T alphaVal_ = 0;
};

#endif // FOREACH_BINARY_ALPHA_CAST_INPLACE_H
