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
 * \file apply_adam_w_dag.h
 * \brief
 */

#ifndef APPLY_ADAM_W_DAG_H
#define APPLY_ADAM_W_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace ApplyAdamWV2RegbaseOp {
using namespace Ops::Base;
constexpr int CAST_NONE_MODE = 0;
constexpr int CAST_RINT_MODE = 1;
constexpr int INDEX_0 = 0;
constexpr int INDEX_1 = 1;
constexpr int INDEX_2 = 2;
constexpr int INDEX_3 = 3;
constexpr int INDEX_4 = 4;
constexpr int INDEX_5 = 5;

template <typename T1, typename T2, typename T3, typename U = float>
struct ApplyAdamWV2DAG {
  using OpOneU = MAKE_CONST(U, 1);
  using OpVar = Bind<Vec::CopyIn<T1>, Placeholder::In0<T1>>;
  using OpM = Bind<Vec::CopyIn<T1>, Placeholder::In1<T1>>;
  using OpV = Bind<Vec::CopyIn<T1>, Placeholder::In2<T1>>;
  using OpGrad = Bind<Vec::CopyIn<T2>, Placeholder::In3<T2>>;
  using OpVarCast = Bind<Vec::Cast<U, T1, CAST_NONE_MODE>, OpVar>;
  using OpMCast = Bind<Vec::Cast<U, T1, CAST_NONE_MODE>, OpM>;
  using OpVCast = Bind<Vec::Cast<U, T1, CAST_NONE_MODE>, OpV>;
  using OpGradCast = Bind<Vec::Cast<U, T2, CAST_NONE_MODE>, OpGrad>;
  using OpGradCast_ = Bind<Vec::Muls<U>, OpGradCast, Placeholder::Var<U, INDEX_5>>;
  using OpConstOne = Bind<Vec::Duplicate<U>, OpOneU>;
  using OpBeta1Sub1 = Bind<Vec::Subs<U>, Placeholder::Var<U, INDEX_1>, OpConstOne>;
  using OpBeta2Sub1 = Bind<Vec::Subs<U>, Placeholder::Var<U, INDEX_2>, OpConstOne>;
  using OpBeta1Gt = Bind<Vec::Mul<U>, OpGradCast_, OpBeta1Sub1>;
  using OpGradSq = Bind<Vec::Mul<U>, OpGradCast_, OpGradCast_>;
  using OpBeta2Gt = Bind<Vec::Mul<U>, OpGradSq, OpBeta2Sub1>;
  using OpMMulBeta = Bind<Vec::Muls<U>, OpMCast, Placeholder::Var<U, INDEX_1>>;
  using OpVMulBeta = Bind<Vec::Muls<U>, OpVCast, Placeholder::Var<U, INDEX_2>>;
  using OpMOut = Bind<Vec::Sub<U>, OpMMulBeta, OpBeta1Gt>;
  using OpVOut = Bind<Vec::Sub<U>, OpVMulBeta, OpBeta2Gt>;

  using OpStepVec = Bind<Vec::Duplicate<T3>, Placeholder::In4<T3, Placeholder::ScalarAttr<true>>>;
  using OpStepCast = Bind<Vec::Cast<U, T3, CAST_RINT_MODE>, OpStepVec>;
  using OpStepAddOne = Bind<Vec::Adds<U>, OpStepCast, OpOneU>;
  using OpBeta1Power = Bind<Vec::Power<U>, Placeholder::Var<U, INDEX_1>, OpStepAddOne>;
  using OpBeta2Power = Bind<Vec::Power<U>, Placeholder::Var<U, INDEX_2>, OpStepAddOne>;

  using OpSqrtV = Bind<Vec::Sqrt<U>, OpVOut>;
  using Op1SubBeta2Power = Bind<Vec::Subs<U>, OpOneU, OpBeta2Power>;
  using OpSqrtBeta2Power = Bind<Vec::Sqrt<U>, Op1SubBeta2Power>;
  using OpDenom = Bind<Vec::Div<U>, OpSqrtV, OpSqrtBeta2Power>;
  using OpDenomAddEps = Bind<Vec::Adds<U>, OpDenom, Placeholder::Var<U, INDEX_4>>;
  using OpLr = Bind<Vec::Duplicate<U>, Placeholder::Var<U, INDEX_0>>;
  using OpLrDecay = Bind<Vec::Muls<U>, OpLr, Placeholder::Var<U, INDEX_3>>;
  using Op1SubLrDecay = Bind<Vec::Subs<U>, OpOneU, OpLrDecay>;
  using OpVarT = Bind<Vec::Mul<U>, OpVarCast, Op1SubLrDecay>;

  using OpBeta1PowerSub1 = Bind<Vec::Subs<U>, OpBeta1Power, OpOneU>;
  using OpLrBeta1Power = Bind<Vec::Divs<U>, Placeholder::Var<U, INDEX_0>, OpBeta1PowerSub1>;
  using OpMDivDenom = Bind<Vec::Div<U>, OpMOut, OpDenomAddEps>;
  using OpVarMoment = Bind<Vec::Mul<U>, OpLrBeta1Power, OpMDivDenom>;
  using OpVarOut = Bind<Vec::Add<U>, OpVarT, OpVarMoment>;

  using OpVarOutCast = Bind<Vec::Cast<T1, U, CAST_RINT_MODE>, OpVarOut>;
  using OpMOutCast = Bind<Vec::Cast<T1, U, CAST_RINT_MODE>, OpMOut>;
  using OpVOutCast = Bind<Vec::Cast<T1, U, CAST_RINT_MODE>, OpVOut>;
  using OpCopyVarOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, OpVarOutCast>;
  using OpCopyMOut = Bind<Vec::CopyOut<T1>, Placeholder::Out1<T1>, OpMOutCast>;
  using OpCopyVOut = Bind<Vec::CopyOut<T1>, Placeholder::Out2<T1>, OpVOutCast>;

  using Outputs = Elems<OpCopyVarOut, OpCopyMOut, OpCopyVOut>;
  using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
  using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2, typename T3, typename U = float>
struct ApplyAdamWV2AmsGradDAG {
    using OpOneU = MAKE_CONST(U, 1);
    using OpVar = Bind<Vec::CopyIn<T1>, Placeholder::In0<T1>>;
    using OpM = Bind<Vec::CopyIn<T1>, Placeholder::In1<T1>>;
    using OpV = Bind<Vec::CopyIn<T1>, Placeholder::In2<T1>>;
    using OpGrad = Bind<Vec::CopyIn<T2>, Placeholder::In3<T2>>;
    using OpVarCast = Bind<Vec::Cast<U, T1, CAST_NONE_MODE>, OpVar>;
    using OpMCast = Bind<Vec::Cast<U, T1, CAST_NONE_MODE>, OpM>;
    using OpVCast = Bind<Vec::Cast<U, T1, CAST_NONE_MODE>, OpV>;
    using OpGradCast = Bind<Vec::Cast<U, T2, CAST_NONE_MODE>, OpGrad>;
    using OpGradCast_ = Bind<Vec::Muls<U>, OpGradCast, Placeholder::Var<U, INDEX_5>>;
    using OpConstOne = Bind<Vec::Duplicate<U>, OpOneU>;
    using OpBeta1Sub1 = Bind<Vec::Subs<U>, Placeholder::Var<U, INDEX_1>, OpConstOne>;
    using OpBeta2Sub1 = Bind<Vec::Subs<U>, Placeholder::Var<U, INDEX_2>, OpConstOne>;
    using OpBeta1Gt = Bind<Vec::Mul<U>, OpGradCast_, OpBeta1Sub1>;
    using OpGradSq = Bind<Vec::Mul<U>, OpGradCast_, OpGradCast_>;
    using OpBeta2Gt = Bind<Vec::Mul<U>, OpGradSq, OpBeta2Sub1>;
    using OpMMulBeta = Bind<Vec::Muls<U>, OpMCast, Placeholder::Var<U, INDEX_1>>;
    using OpVMulBeta = Bind<Vec::Muls<U>, OpVCast, Placeholder::Var<U, INDEX_2>>;
    using OpMOut = Bind<Vec::Sub<U>, OpMMulBeta, OpBeta1Gt>;
    using OpVOut = Bind<Vec::Sub<U>, OpVMulBeta, OpBeta2Gt>;

    using OpStepVec = Bind<Vec::Duplicate<T3>, Placeholder::In4<T3, Placeholder::ScalarAttr<true>>>;
    using OpStepCast = Bind<Vec::Cast<U, T3, CAST_RINT_MODE>, OpStepVec>;
    using OpStepAddOne = Bind<Vec::Adds<U>, OpStepCast, OpOneU>;
    using OpBeta1Power = Bind<Vec::Power<U>, Placeholder::Var<U, INDEX_1>, OpStepAddOne>;
    using OpBeta2Power = Bind<Vec::Power<U>, Placeholder::Var<U, INDEX_2>, OpStepAddOne>;

    using OpMaxGradNorm = Bind<Vec::CopyIn<T2>, Placeholder::In5<T2>>;
    using OpMaxGradNormCast = Bind<Vec::Cast<U, T2, CAST_NONE_MODE>, OpMaxGradNorm>;
    using OpAmsGrad = Bind<Vec::Max<U>, OpVOut, OpMaxGradNormCast>;
    using OpSqrtV = Bind<Vec::Sqrt<U>, OpAmsGrad>;
    using Op1SubBeta2Power = Bind<Vec::Subs<U>, OpOneU, OpBeta2Power>;
    using OpSqrtBeta2Power = Bind<Vec::Sqrt<U>, Op1SubBeta2Power>;
    using OpDenom = Bind<Vec::Div<U>, OpSqrtV, OpSqrtBeta2Power>;
    using OpDenomAddEps = Bind<Vec::Adds<U>, OpDenom, Placeholder::Var<U, INDEX_4>>;
    using OpLr = Bind<Vec::Duplicate<U>, Placeholder::Var<U, INDEX_0>>;
    using OpLrDecay = Bind<Vec::Muls<U>, OpLr, Placeholder::Var<U, INDEX_3>>;
    using Op1SubLrDecay = Bind<Vec::Subs<U>, OpOneU, OpLrDecay>;
    using OpVarT = Bind<Vec::Mul<U>, OpVarCast, Op1SubLrDecay>;

    using OpBeta1PowerSub1 = Bind<Vec::Subs<U>, OpBeta1Power, OpOneU>;
    using OpLrBeta1Power = Bind<Vec::Divs<U>, Placeholder::Var<U, INDEX_0>, OpBeta1PowerSub1>;
    using OpMDivDenom = Bind<Vec::Div<U>, OpMOut, OpDenomAddEps>;
    using OpVarMoment = Bind<Vec::Mul<U>, OpLrBeta1Power, OpMDivDenom>;
    using OpVarOut = Bind<Vec::Add<U>, OpVarT, OpVarMoment>;

    using OpVarOutCast = Bind<Vec::Cast<T1, U, CAST_RINT_MODE>, OpVarOut>;
    using OpMOutCast = Bind<Vec::Cast<T1, U, CAST_RINT_MODE>, OpMOut>;
    using OpVOutCast = Bind<Vec::Cast<T1, U, CAST_RINT_MODE>, OpVOut>;
    using OpAmsGradCast = Bind<Vec::Cast<T2, U, CAST_RINT_MODE>, OpAmsGrad>;
    using OpCopyVarOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, OpVarOutCast>;
    using OpCopyMOut = Bind<Vec::CopyOut<T1>, Placeholder::Out1<T1>, OpMOutCast>;
    using OpCopyVOut = Bind<Vec::CopyOut<T1>, Placeholder::Out2<T1>, OpVOutCast>;
    using OpCopyMaxGradNormOut = Bind<Vec::CopyOut<T2>, Placeholder::Out3<T2>, OpAmsGradCast>;

    using Outputs = Elems<OpCopyVarOut, OpCopyMOut, OpCopyVOut, OpCopyMaxGradNormOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace ApplyAdamWV2RegbaseOp
#endif  // APPLY_ADAM_W_DAG_H
