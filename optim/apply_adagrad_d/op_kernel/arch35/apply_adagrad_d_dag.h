/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file apply_adagrad_d_dag.h
 * \brief 
 */

#ifndef CANN_CUSTOM_OPS_APPLY_ADAGRAD_D_DAG_H
#define CANN_CUSTOM_OPS_APPLY_ADAGRAD_D_DAG_H
 
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
namespace ApplyAdagradDOp {

    template <typename U, typename T = float>
    struct ApplyAdagradDUpdateSlots {
        using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
        using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
        using OpCopyInLr = Bind<Vec::Duplicate<U>, Placeholder::In2<U, Placeholder::ScalarAttr<true>>>;
        using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;

        using OpVarCast = Bind<Vec::Cast<T, U, 0>, OpCopyInVar>;
        using OpAccumCast = Bind<Vec::Cast<T, U, 0>, OpCopyInAccum>;
        using OpLrCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
        using OpGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyInGrad>;
        
        using OpGradPower = Bind<Vec::Mul<T>, OpGradCast, OpGradCast>;
        using OpAccumOut = Bind<Vec::Add<T>, OpAccumCast, OpGradPower>;
        using OpAccumOutCast = Bind<Vec::Cast<U, T, 1>, OpAccumOut>;
        using OpCopyOutAccum = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpAccumOutCast>;

        using OpAccumSqrt = Bind<Vec::Sqrt<T>, OpAccumOut>;
        using OpLrMulGrad = Bind<Vec::Mul<T>, OpGradCast, OpLrCast>;
        using OpVarT = Bind<Vec::DivHighPrecision<T>, OpLrMulGrad, OpAccumSqrt>;
        using OpVarOut = Bind<Vec::Sub<T>, OpVarCast, OpVarT>;
        using OpVarOutCast = Bind<Vec::Cast<U, T, 1>, OpVarOut>;
        using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>;

        using Outputs = Elems<OpCopyOutVar, OpCopyOutAccum>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };

    template <typename U, typename T = float>
    struct ApplyAdagradD {
        using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
        using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
        using OpCopyInLr = Bind<Vec::Duplicate<U>, Placeholder::In2<U, Placeholder::ScalarAttr<true>>>;
        using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;

        using OpVarCast = Bind<Vec::Cast<T, U, 0>, OpCopyInVar>;
        using OpAccumCast = Bind<Vec::Cast<T, U, 0>, OpCopyInAccum>;
        using OpLrCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
        using OpGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyInGrad>;

        using OpAccumSqrt = Bind<Vec::Sqrt<T>, OpAccumCast>;
        using OpLrMulGrad = Bind<Vec::Mul<T>, OpGradCast, OpLrCast>;
        using OpVarT = Bind<Vec::DivHighPrecision<T>, OpLrMulGrad, OpAccumSqrt>;
        using OpVarOut = Bind<Vec::Sub<T>, OpVarCast, OpVarT>;
        using OpVarOutCast = Bind<Vec::Cast<U, T, 1>, OpVarOut>;
        using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>;

        using Outputs = Elems<OpCopyOutVar>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };
}
#endif  // CANN_CUSTOM_OPS_SQRT_GRAD_DAG_H