/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_mul_scalar_regbase.h
 * \brief
 */
#ifndef FOREACH_MUL_SCALAR_REGBASE_H
#define FOREACH_MUL_SCALAR_REGBASE_H

#include "../../foreach_utils/arch35/foreach_regbase_unary.h"

namespace ForeachMulScalar {
using namespace AscendC;
template <typename T, typename ScalarT, typename Tiling>
class ForeachMulScalarRegbase : public ForeachRegbaseUnary<T, Tiling, ForeachMulScalarRegbase<T, ScalarT, Tiling>>
{
public:
    using Base = ForeachRegbaseUnary<T, Tiling, ForeachMulScalarRegbase<T, ScalarT, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachMulScalarRegbase() : Base(*this){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(inputs, outputs, workspace, tilingData, tPipe);
        inScalarGM_.SetGlobalBuffer((__gm__ ScalarT*)scalar, 1);
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            scalarVal_ = inScalarGM_.GetValue(0);
            tPipe->InitBuffer(castBuf_, tilingData->inputsTensorUbSize * sizeof(float));
            castLocal_ = castBuf_.Get<float>();
        } else {
            scalarVal_ = T(inScalarGM_.GetValue(0));
        }
    }

    __aicore__ inline void Compute(LocalTensor<T> inLocal, LocalTensor<T> outLocal, int64_t dataCount)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            Cast(this->castLocal_, inLocal, RoundMode::CAST_NONE, dataCount);
            Muls(this->castLocal_, this->castLocal_, float(this->scalarVal_), dataCount);
            Cast(outLocal, this->castLocal_, RoundMode::CAST_RINT, dataCount);
        } else {
            Muls(outLocal, inLocal, this->scalarVal_, dataCount);
        }
    }

private:
    GlobalTensor<ScalarT> inScalarGM_;
    TBuf<TPosition::VECCALC> castBuf_;
    LocalTensor<float> castLocal_;
    ScalarT scalarVal_ = 0;
};
} // namespace ForeachMulScalar

#endif // FOREACH_Mul_SCALAR_REGBASE_H