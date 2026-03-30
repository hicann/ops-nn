/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file apply_momentum_dag.h
 * \brief 
 */

#ifndef APPLY_MOMENTUM_DAG_H
#define APPLY_MOMENTUM_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace ApplyMomentumOp {
    using namespace Ops::Base;

    template <typename U, typename T = float>
    struct ApplyMomentumDag {
        using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
        using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
        using OpCopyInLr = Bind<Vec::Duplicate<U>, Placeholder::In2<U, Placeholder::ScalarAttr<true>>>;
        using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;
        using OpCopyInMomentum = Bind<Vec::Duplicate<U>, Placeholder::In4<U, Placeholder::ScalarAttr<true>>>;

        using OpVarCast = Bind<Vec::Cast<T, U, 0>, OpCopyInVar>;
        using OpAccumCast = Bind<Vec::Cast<T, U, 0>, OpCopyInAccum>;
        using OpLrCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
        using OpGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyInGrad>;
        using OpMomentumCast = Bind<Vec::Cast<T, U, 0>, OpCopyInMomentum>;

        using OpAccMulMom = Bind<Vec::Mul<T>, OpAccumCast, OpMomentumCast>;
        using OpAccumNew = Bind<Vec::Add<T>, OpGradCast, OpAccMulMom>;
        using OpAccumOutCast = Bind<Vec::Cast<U, T, 1>, OpAccumNew>;
        using OpCopyOutAccum = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpAccumOutCast>; // update input1: accum

        using OpLrMulAcc = Bind<Vec::Mul<T>, OpLrCast, OpAccumNew>;
        using OpVarNew = Bind<Vec::Sub<T>, OpVarCast, OpLrMulAcc>;
        using OpVarOutCast = Bind<Vec::Cast<U, T, 1>, OpVarNew>;
        using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>; // output: var

        using Outputs = Elems<OpCopyOutVar, OpCopyOutAccum>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };

    template <typename U, typename T = float>
    struct ApplyNesterovMomentumDag {
        using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
        using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
        using OpCopyInLr = Bind<Vec::Duplicate<U>, Placeholder::In2<U, Placeholder::ScalarAttr<true>>>;
        using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;
        using OpCopyInMomentum = Bind<Vec::Duplicate<U>, Placeholder::In4<U, Placeholder::ScalarAttr<true>>>;

        using OpVarCast = Bind<Vec::Cast<T, U, 0>, OpCopyInVar>;
        using OpAccumCast = Bind<Vec::Cast<T, U, 0>, OpCopyInAccum>;
        using OpLrCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
        using OpGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyInGrad>;
        using OpMomentumCast = Bind<Vec::Cast<T, U, 0>, OpCopyInMomentum>;

        using OpAccMulMom = Bind<Vec::Mul<T>, OpAccumCast, OpMomentumCast>;
        using OpAccumNew = Bind<Vec::Add<T>, OpGradCast, OpAccMulMom>;
        using OpAccumOutCast = Bind<Vec::Cast<U, T, 1>, OpAccumNew>;
        using OpCopyOutAccum = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpAccumOutCast>; // update input1: accum

        using OpAccNewMulMom = Bind<Vec::Mul<T>, OpAccumNew, OpMomentumCast>;
        using OpAccMulMomLr = Bind<Vec::Mul<T>, OpAccNewMulMom, OpLrCast>;
        using OpGradMulLr = Bind<Vec::Mul<T>, OpGradCast, OpLrCast>;
        using OpSGD = Bind<Vec::Add<T>, OpGradMulLr, OpAccMulMomLr>;
        using OpVarNew = Bind<Vec::Sub<T>, OpVarCast, OpSGD>;
        using OpVarOutCast = Bind<Vec::Cast<U, T, 1>, OpVarNew>;
        using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>; // output: var

        using Outputs = Elems<OpCopyOutVar, OpCopyOutAccum>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };
} // namespace ApplyMomentumOp

#endif  // APPLY_MOMENTUM_DAG_H 