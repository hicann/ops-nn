/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file hard_sigmoid_dag.h
 * \brief HardSigmoid DAG definitions (atvoss Elewise mode)
 *
 * Formula: HardSigmoid(x) = max(0, min(1, alpha * x + beta))
 *   where alpha = 1/6, beta = 0.5
 *
 * Three DAG variants:
 *   1. HardSigmoidCompute<T>   - fp32 direct compute (no Cast)
 *   2. HardSigmoidWithCast<T>  - fp16/bf16 (Cast up to fp32, compute, Cast down)
 *   3. HardSigmoidInt32         - int32 (Cast RINT to fp32, compute, Cast RINT back)
 */

#ifndef HARD_SIGMOID_DAG_H
#define HARD_SIGMOID_DAG_H

// Host compile mock for __aicore__
#ifndef __CCE_AICORE__
#ifndef __aicore__
#define __aicore__
#endif
#endif

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

namespace NsHardSigmoid {

// ============================================================================
// Variant 1: HardSigmoidCompute<T> - fp32 direct compute (no Cast)
// ============================================================================
template <typename T>
struct HardSigmoidCompute {
    using VarAlpha  = Placeholder::Var<T, 0>;
    using VarBeta   = Placeholder::Var<T, 1>;
    using ConstOne  = MAKE_CONST(T, 1.0f);
    using ConstZero = MAKE_CONST(T, 0.0f);

    using OpInput   = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpMuls    = Bind<Vec::Muls<T>, OpInput, VarAlpha>;
    using OpAdds    = Bind<Vec::Adds<T>, OpMuls, VarBeta>;
    using OpMins    = Bind<Vec::Mins<T>, OpAdds, ConstOne>;         // min(., 1)
    using OpMaxs    = Bind<Vec::Maxs<T>, OpMins, ConstZero>;        // max(., 0)
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpMaxs>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};

// ============================================================================
// Variant 2: HardSigmoidWithCast<T> - fp16/bf16 Cast to fp32 compute
// ============================================================================
template <typename T>
struct HardSigmoidWithCast {
    using VarAlpha  = Placeholder::Var<float, 0>;
    using VarBeta   = Placeholder::Var<float, 1>;
    using ConstOne  = MAKE_CONST(float, 1.0f);
    using ConstZero = MAKE_CONST(float, 0.0f);

    using OpInput   = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastUp    = Bind<Vec::Cast<float, T, 0>, OpInput>;           // T -> float (CAST_NONE)
    using OpMuls    = Bind<Vec::Muls<float>, CastUp, VarAlpha>;
    using OpAdds    = Bind<Vec::Adds<float>, OpMuls, VarBeta>;
    using OpMins    = Bind<Vec::Mins<float>, OpAdds, ConstOne>;
    using OpMaxs    = Bind<Vec::Maxs<float>, OpMins, ConstZero>;
    using CastDown  = Bind<Vec::Cast<T, float, 1>, OpMaxs>;            // float -> T (CAST_RINT)
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastDown>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};

// ============================================================================
// Variant 3: HardSigmoidInt32 - int32 special path (CAST_RINT both ways)
// ============================================================================
struct HardSigmoidInt32 {
    using VarAlpha  = Placeholder::Var<float, 0>;
    using VarBeta   = Placeholder::Var<float, 1>;
    using ConstOne  = MAKE_CONST(float, 1.0f);
    using ConstZero = MAKE_CONST(float, 0.0f);

    using OpInput   = Bind<Vec::CopyIn<int32_t>, Placeholder::In0<int32_t>>;
    using CastUp    = Bind<Vec::Cast<float, int32_t, 1>, OpInput>;      // int32 -> float (CAST_RINT)
    using OpMuls    = Bind<Vec::Muls<float>, CastUp, VarAlpha>;
    using OpAdds    = Bind<Vec::Adds<float>, OpMuls, VarBeta>;
    using OpMins    = Bind<Vec::Mins<float>, OpAdds, ConstOne>;
    using OpMaxs    = Bind<Vec::Maxs<float>, OpMins, ConstZero>;
    using CastDown  = Bind<Vec::Cast<int32_t, float, 5>, OpMaxs>;       // float -> int32 (CAST_TRUNC, matches golden int32 truncate)
    using OpCopyOut = Bind<Vec::CopyOut<int32_t>, Placeholder::Out0<int32_t>, CastDown>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};

} // namespace NsHardSigmoid

#endif  // HARD_SIGMOID_DAG_H
