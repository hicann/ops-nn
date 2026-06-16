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
 * \file    lamb_apply_optimizer_assign_dag.h
 * \brief lamb_apply_optimizer_assign_dag head file
 */

#ifndef LAMB_APPLY_OPTIMIZER_ASSIGN_DAG_H
#define LAMB_APPLY_OPTIMIZER_ASSIGN_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LambApplyOptimizerAssignOp {
using namespace AscendC;
using namespace Ops::Base;
// In : 0 grad, 1 inputv(v), 2 inputm(m), 3 input3(param), 4 mul0_x(b1), 5 mul1_x(1-b1),
//      6 mul2_x(b2), 7 mul3_x(1-b2), 8 add2_y(eps), 9 steps, 10 do_use_weight, 11 weight_decay_rate
// Out: 0 update, 1 inputv(next_v, in-place), 2 inputm(next_m, in-place)
// Compute in U (=float): half casts to float (div/sqrt/log/exp need float precision); fp32 native.
template <typename T, typename U>
struct LambApplyOptimizerAssignCompute {
    using CNegOne = MAKE_CONST(U, -1.0);
    using COne = MAKE_CONST(U, 1.0);

    using InGrad = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InV = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InM = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using InParam = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;
    // 8 scalar [1] inputs: ScalarAttr + Duplicate (broadcast scalar to a tensor without a
    // double-buffered mte2 buffer) so downstream tensor ops match and we stay in buffer budget.
    using InB1 = Bind<Vec::Duplicate<T>, Placeholder::In4<T, Placeholder::ScalarAttr<true>>>;
    using InOmB1 = Bind<Vec::Duplicate<T>, Placeholder::In5<T, Placeholder::ScalarAttr<true>>>;
    using InB2 = Bind<Vec::Duplicate<T>, Placeholder::In6<T, Placeholder::ScalarAttr<true>>>;
    using InOmB2 = Bind<Vec::Duplicate<T>, Placeholder::In7<T, Placeholder::ScalarAttr<true>>>;
    using InEps = Bind<Vec::Duplicate<T>, Placeholder::In8<T, Placeholder::ScalarAttr<true>>>;
    using InSteps = Bind<Vec::Duplicate<T>, Placeholder::In9<T, Placeholder::ScalarAttr<true>>>;
    using InDoUse = Bind<Vec::Duplicate<T>, Placeholder::In10<T, Placeholder::ScalarAttr<true>>>;
    using InWd = Bind<Vec::Duplicate<T>, Placeholder::In11<T, Placeholder::ScalarAttr<true>>>;

    using Grad = Bind<Vec::Cast<U, T, 0>, InGrad>;
    using V = Bind<Vec::Cast<U, T, 0>, InV>;
    using M = Bind<Vec::Cast<U, T, 0>, InM>;
    using Param = Bind<Vec::Cast<U, T, 0>, InParam>;
    using B1 = Bind<Vec::Cast<U, T, 0>, InB1>;
    using OmB1 = Bind<Vec::Cast<U, T, 0>, InOmB1>;
    using B2 = Bind<Vec::Cast<U, T, 0>, InB2>;
    using OmB2 = Bind<Vec::Cast<U, T, 0>, InOmB2>;
    using Eps = Bind<Vec::Cast<U, T, 0>, InEps>;
    using Steps = Bind<Vec::Cast<U, T, 0>, InSteps>;
    using DoUse = Bind<Vec::Cast<U, T, 0>, InDoUse>;
    using Wd = Bind<Vec::Cast<U, T, 0>, InWd>;

    // next_v = g^2 * (1 - b2) + v * b2
    using G2 = Bind<Vec::Mul<U>, Grad, Grad>;
    using NV1 = Bind<Vec::Mul<U>, G2, OmB2>;
    using NV2 = Bind<Vec::Mul<U>, V, B2>;
    using NextV = Bind<Vec::Add<U>, NV1, NV2>;
    using NextVCast = Bind<Vec::Cast<T, U, 1>, NextV>;

    // next_m = m * b1 + g * (1 - b1)
    using NM1 = Bind<Vec::Mul<U>, M, B1>;
    using NM2 = Bind<Vec::Mul<U>, Grad, OmB1>;
    using NextM = Bind<Vec::Add<U>, NM1, NM2>;
    using NextMCast = Bind<Vec::Cast<T, U, 1>, NextM>;

    // bias correction: b_corr = 1 - exp(steps * ln(b))   (b1/b2/steps are scalar [1])
    using LnB1 = Bind<Vec::Log<U>, B1>;
    using ExpArg1 = Bind<Vec::Mul<U>, LnB1, Steps>;
    using B1Steps = Bind<Vec::Exp<U>, ExpArg1>;
    using NegB1Steps = Bind<Vec::Muls<U>, B1Steps, CNegOne>;
    using B1corr = Bind<Vec::Adds<U>, NegB1Steps, COne>;
    using LnB2 = Bind<Vec::Log<U>, B2>;
    using ExpArg2 = Bind<Vec::Mul<U>, LnB2, Steps>;
    using B2Steps = Bind<Vec::Exp<U>, ExpArg2>;
    using NegB2Steps = Bind<Vec::Muls<U>, B2Steps, CNegOne>;
    using B2corr = Bind<Vec::Adds<U>, NegB2Steps, COne>;

    // update = (next_m / b1corr) / (sqrt(next_v / b2corr) + eps) + param * wd * do_use_weight
    using MUnb = Bind<Vec::Div<U>, NextM, B1corr>;
    using VUnb = Bind<Vec::Div<U>, NextV, B2corr>;
    using SqrtVUnb = Bind<Vec::Sqrt<U>, VUnb>;
    using Den = Bind<Vec::Add<U>, SqrtVUnb, Eps>;
    using Upd0 = Bind<Vec::Div<U>, MUnb, Den>;
    using WdMul = Bind<Vec::Mul<U>, Param, Wd>;
    using Wdec = Bind<Vec::Mul<U>, WdMul, DoUse>;
    using Update = Bind<Vec::Add<U>, Wdec, Upd0>;
    using UpdateCast = Bind<Vec::Cast<T, U, 1>, Update>;

    using OpCopyOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, UpdateCast>;
    using OpCopyOut1 = Bind<Vec::CopyOut<T>, Placeholder::Out1<T>, NextVCast>;
    using OpCopyOut2 = Bind<Vec::CopyOut<T>, Placeholder::Out2<T>, NextMCast>;

    using Outputs = Elems<OpCopyOut0, OpCopyOut1, OpCopyOut2>;
    // 12 inputs + 3 outputs exceed the LEVEL_2 (double-buffered) 32-buffer budget; LEVEL_1 fits.
    using MemCfg = MemOptCfg<MemLevel::LEVEL_1>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LambApplyOptimizerAssignOp
#endif // LAMB_APPLY_OPTIMIZER_ASSIGN_DAG_H
