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
 * \file foreach_sqrt_regbase.h
 * \brief
 */
#ifndef FOREACH_SQRT_REGBASE_H
#define FOREACH_SQRT_REGBASE_H

#include "../../foreach_utils/arch35/foreach_regbase_unary.h"

namespace ForeachSqrt {
using namespace AscendC;
template <typename T, typename Tiling>
class ForeachSqrtRegbase : public ForeachRegbaseUnary<T, Tiling, ForeachSqrtRegbase<T, Tiling>>
{
public:
    using Base = ForeachRegbaseUnary<T, Tiling, ForeachSqrtRegbase<T, Tiling>>;
    using Base::Process;
    __aicore__ inline ForeachSqrtRegbase() : Base(*this){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData, TPipe* tPipe)
    {
        Base::Init(inputs, outputs, workspace, tilingData, tPipe);
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            tPipe->InitBuffer(castBuf_, tilingData->inputsTensorUbSize * sizeof(float));
            castLocal_ = castBuf_.Get<float>();
        }
    }

    __aicore__ inline void Compute(LocalTensor<T> inLocal, LocalTensor<T> outLocal, int64_t dataCount)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            Cast(this->castLocal_, inLocal, RoundMode::CAST_NONE, dataCount);
            Sqrt(this->castLocal_, this->castLocal_, dataCount);
            Cast(outLocal, this->castLocal_, RoundMode::CAST_RINT, dataCount);
        } else {
            Sqrt(outLocal, inLocal, dataCount);
        }
    }

private:
    TBuf<TPosition::VECCALC> castBuf_;
    LocalTensor<float> castLocal_;
};
} // namespace ForeachSqrt

#endif // FOREACH_SQRT_REGBASE_H
