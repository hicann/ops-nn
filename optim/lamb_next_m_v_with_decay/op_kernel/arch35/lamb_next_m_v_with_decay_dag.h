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
 * \file    lamb_next_m_v_with_decay_dag.h
 * \brief lamb_next_m_v_with_decay_dag head file
 */

#ifndef LAMB_NEXT_M_V_DAG_H
#define LAMB_NEXT_M_V_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

// The atvoss Placeholder ships In0..In11 (12 inputs). LambNextMVWithDecay has 13 inputs; extend with In12
// following the framework's exact pattern (additive: the struct + its IsInHolder trait, same namespace).
namespace Ops {
namespace Base {
namespace Placeholder {
template <class T, class Attr = InAttr<>>
struct In12 : public Holder<T, Attr, 12> {};
template <class U, class Attr>
struct IsInHolder<In12<U, Attr>> {
    constexpr static bool Value = true;
};
} // namespace Placeholder
} // namespace Base
} // namespace Ops

namespace LambNextMVWithDecayOp {
using namespace AscendC;
using namespace Ops::Base;
// In : 0 input_mul3(g^2) 1 input_mul2(v) 2 input_realdiv1(1-b2^t) 3 input_mul1(g) 4 input_mul0(m)
//      5 input_realdiv0(1-b1^t) 6 input_mul4(param) 7 mul0_x(b1) 8 mul1_sub(1-b1) 9 mul2_x(b2)
//      10 mul3_sub1(1-b2) 11 mul4_x(wd) 12 add2_y(eps)   [0,1,3,4,6 full; rest scalar]
// Out: 0 y1(update) 1 y2(next_m) 2 y3(next_v) 3 y4(m_unbiased/(sqrt(v_unbiased)+eps))
// Compute in U (=float): half casts to float (div/sqrt need float precision). rsqrt -> Div(_,Sqrt).
template <typename T, typename U>
struct LambNextMVWithDecayCompute {
    // full tensors
    using InG2 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InV = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InG = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;
    using InM = Bind<Vec::CopyInBrc<T>, Placeholder::In4<T>>;
    using InParam = Bind<Vec::CopyInBrc<T>, Placeholder::In6<T>>;
    // scalar coefficients
    using InRd1 = Bind<Vec::Duplicate<T>, Placeholder::In2<T, Placeholder::ScalarAttr<true>>>;
    using InRd0 = Bind<Vec::Duplicate<T>, Placeholder::In5<T, Placeholder::ScalarAttr<true>>>;
    using InB1 = Bind<Vec::Duplicate<T>, Placeholder::In7<T, Placeholder::ScalarAttr<true>>>;
    using InOmB1 = Bind<Vec::Duplicate<T>, Placeholder::In8<T, Placeholder::ScalarAttr<true>>>;
    using InB2 = Bind<Vec::Duplicate<T>, Placeholder::In9<T, Placeholder::ScalarAttr<true>>>;
    using InOmB2 = Bind<Vec::Duplicate<T>, Placeholder::In10<T, Placeholder::ScalarAttr<true>>>;
    using InWd = Bind<Vec::Duplicate<T>, Placeholder::In11<T, Placeholder::ScalarAttr<true>>>;
    using InEps = Bind<Vec::Duplicate<T>, Placeholder::In12<T, Placeholder::ScalarAttr<true>>>;

    using G2 = Bind<Vec::Cast<U, T, 0>, InG2>;
    using V = Bind<Vec::Cast<U, T, 0>, InV>;
    using G = Bind<Vec::Cast<U, T, 0>, InG>;
    using M = Bind<Vec::Cast<U, T, 0>, InM>;
    using Param = Bind<Vec::Cast<U, T, 0>, InParam>;
    using Rd1 = Bind<Vec::Cast<U, T, 0>, InRd1>;
    using Rd0 = Bind<Vec::Cast<U, T, 0>, InRd0>;
    using B1 = Bind<Vec::Cast<U, T, 0>, InB1>;
    using OmB1 = Bind<Vec::Cast<U, T, 0>, InOmB1>;
    using B2 = Bind<Vec::Cast<U, T, 0>, InB2>;
    using OmB2 = Bind<Vec::Cast<U, T, 0>, InOmB2>;
    using Wd = Bind<Vec::Cast<U, T, 0>, InWd>;
    using Eps = Bind<Vec::Cast<U, T, 0>, InEps>;

    // next_v = v*b2 + g^2*(1-b2)
    using NextV = Bind<Vec::Add<U>, Bind<Vec::Mul<U>, V, B2>, Bind<Vec::Mul<U>, G2, OmB2>>;
    using VUnbias = Bind<Vec::Div<U>, NextV, Rd1>;
    using SqrtVeps = Bind<Vec::Sqrt<U>, Bind<Vec::Add<U>, VUnbias, Eps>>;       // sqrt(v_unbiased+eps)
    using SqrtVAddEps = Bind<Vec::Add<U>, Bind<Vec::Sqrt<U>, VUnbias>, Eps>;    // sqrt(v_unbiased)+eps
    // next_m = m*b1 + g*(1-b1)
    using NextM = Bind<Vec::Add<U>, Bind<Vec::Mul<U>, M, B1>, Bind<Vec::Mul<U>, G, OmB1>>;
    using MUnbias = Bind<Vec::Div<U>, NextM, Rd0>;
    // y1 = param*wd + m_unbiased/sqrt(v_unbiased+eps)
    using Y1 = Bind<Vec::Add<U>, Bind<Vec::Mul<U>, Param, Wd>, Bind<Vec::Div<U>, MUnbias, SqrtVeps>>;
    // with_decay: y4 = param*wd + m_unbiased/(sqrt(v_unbiased)+eps)  (NextMV has no param*wd term in y4)
    using Y4 = Bind<Vec::Add<U>, Bind<Vec::Mul<U>, Param, Wd>, Bind<Vec::Div<U>, MUnbias, SqrtVAddEps>>;

    using Y1Cast = Bind<Vec::Cast<T, U, 1>, Y1>;
    using Y2Cast = Bind<Vec::Cast<T, U, 1>, NextM>;
    using Y3Cast = Bind<Vec::Cast<T, U, 1>, NextV>;
    using Y4Cast = Bind<Vec::Cast<T, U, 1>, Y4>;

    using OpOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y1Cast>;
    using OpOut1 = Bind<Vec::CopyOut<T>, Placeholder::Out1<T>, Y2Cast>;
    using OpOut2 = Bind<Vec::CopyOut<T>, Placeholder::Out2<T>, Y3Cast>;
    using OpOut3 = Bind<Vec::CopyOut<T>, Placeholder::Out3<T>, Y4Cast>;

    using Outputs = Elems<OpOut0, OpOut1, OpOut2, OpOut3>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_1>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LambNextMVWithDecayOp
#endif // LAMB_NEXT_M_V_DAG_H
