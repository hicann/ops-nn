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
 * \file foreach_add_scalar_regbase.h
 * \brief
 */
#ifndef FOREACH_ADD_SCALAR_REGBASE_H
#define FOREACH_ADD_SCALAR_REGBASE_H

#include "../foreach_utils/arch35/foreach_regbase_unary.h"

namespace ForeachAddScalar {
using namespace AscendC;
template <typename T, typename ScalarT, typename Tiling>
class ForeachAddScalarRegbase : public ForeachRegbaseUnary<T, Tiling, ForeachAddScalarRegbase<T, ScalarT, Tiling>> {
public:
    using Base = ForeachRegbaseUnary<T, Tiling, ForeachAddScalarRegbase<T, ScalarT, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachAddScalarRegbase() : Base(*this){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(inputs, outputs, workspace, tilingData, tPipe);
        inScalarGM.SetGlobalBuffer((__gm__ ScalarT*)scalar, 1);
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            scalarVal = inScalarGM.GetValue(0);
            tPipe->InitBuffer(castBuf, tilingData->inputsTensorUbSize * sizeof(float));
            castLocal = castBuf.Get<float>();
        } else {
            scalarVal = T(inScalarGM.GetValue(0));
        }
    }

    __aicore__ inline void Compute(LocalTensor<T> inLocal, LocalTensor<T> outLocal, int64_t dataCount)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            Cast(this->castLocal, inLocal, RoundMode::CAST_NONE, dataCount);
            Adds(this->castLocal, this->castLocal, float(this->scalarVal), dataCount);
            Cast(outLocal, this->castLocal, RoundMode::CAST_RINT, dataCount);
        } else {
            Adds(outLocal, inLocal, this->scalarVal, dataCount);
        }
    }

private:
    GlobalTensor<ScalarT> inScalarGM;
    TBuf<TPosition::VECCALC> castBuf;
    LocalTensor<float> castLocal;
    ScalarT scalarVal = 0;
};
} // namespace ForeachAddScalar

#endif // FOREACH_ADD_SCALAR_REGBASE_H