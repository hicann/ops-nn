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
 * \file    lamb_next_right_dag.h
 * \brief lamb_next_right_dag head file
 */

#ifndef LAMB_NEXT_RIGHT_DAG_H
#define LAMB_NEXT_RIGHT_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LambNextRightOp {
using namespace AscendC;
using namespace Ops::Base;
// In : 0 input_square(g, full) 1 input_mul2(v, full) 2 mul2_x(b2) 3 mul3_x(1-b2)
//      4 truediv1_recip(1/(1-b2^t)) 5 add2_y(eps)   [2..5 scalar]
// Out: 0 y1(next_v = v*b2 + g^2*(1-b2)) 1 y2(sqrt(next_v/(1-b2^t)) + eps)
// Compute in U (=float): half casts to float (sqrt needs float precision).
template <typename T, typename U>
struct LambNextRightCompute {
    using InG = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InV = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InB2 = Bind<Vec::Duplicate<T>, Placeholder::In2<T, Placeholder::ScalarAttr<true>>>;
    using InOmB2 = Bind<Vec::Duplicate<T>, Placeholder::In3<T, Placeholder::ScalarAttr<true>>>;
    using InRecip = Bind<Vec::Duplicate<T>, Placeholder::In4<T, Placeholder::ScalarAttr<true>>>;
    using InEps = Bind<Vec::Duplicate<T>, Placeholder::In5<T, Placeholder::ScalarAttr<true>>>;

    using G = Bind<Vec::Cast<U, T, 0>, InG>;
    using V = Bind<Vec::Cast<U, T, 0>, InV>;
    using B2 = Bind<Vec::Cast<U, T, 0>, InB2>;
    using OmB2 = Bind<Vec::Cast<U, T, 0>, InOmB2>;
    using Recip = Bind<Vec::Cast<U, T, 0>, InRecip>;
    using Eps = Bind<Vec::Cast<U, T, 0>, InEps>;

    // next_v = v*b2 + g^2*(1-b2)
    using NextV = Bind<Vec::Add<U>, Bind<Vec::Mul<U>, V, B2>, Bind<Vec::Mul<U>, Bind<Vec::Mul<U>, G, G>, OmB2>>;
    // y2 = sqrt(next_v * recip) + eps
    using Y2 = Bind<Vec::Add<U>, Bind<Vec::Sqrt<U>, Bind<Vec::Mul<U>, NextV, Recip>>, Eps>;

    using Y1Cast = Bind<Vec::Cast<T, U, 1>, NextV>;
    using Y2Cast = Bind<Vec::Cast<T, U, 1>, Y2>;

    using OpOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y1Cast>;
    using OpOut1 = Bind<Vec::CopyOut<T>, Placeholder::Out1<T>, Y2Cast>;

    using Outputs = Elems<OpOut0, OpOut1>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LambNextRightOp
#endif // LAMB_NEXT_RIGHT_DAG_H
