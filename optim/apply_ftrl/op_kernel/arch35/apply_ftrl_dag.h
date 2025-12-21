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

#ifndef CANN_CUSTOM_OPS_APPLY_FTRL_DAG_H
#define CANN_CUSTOM_OPS_APPLY_FTRL_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace ApplyFtrlOp {
    using namespace Ops::Base;
    const int COMPARE_MODE_LT = 0;
    const int COMPARE_MODE_GT = 1;
    const int SELECT_MODE_T_S = 1;

    template <typename U, typename T = float>
    struct ApplyFtrlDag {
        using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
        using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
        using OpCopyInLinear = Bind<Vec::CopyIn<U>, Placeholder::In2<U>>;
        using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;
        using OpCopyInLr = Bind<Vec::Duplicate<U>, Placeholder::In4<U, Placeholder::ScalarAttr<true>>>;
        using OpCopyInL1 = std::conditional_t<std::is_same<U, bfloat16_t>::value,
                            Bind<Vec::Duplicate<U>, Placeholder::In5<U, Placeholder::ScalarAttr<true>>>,
                            Bind<Vec::CopyIn<U>, Placeholder::In5<U, Placeholder::ScalarAttr<true>>>>;
        using OpCopyInL2 = std::conditional_t<std::is_same<U, bfloat16_t>::value,
                            Bind<Vec::Duplicate<U>, Placeholder::In6<U, Placeholder::ScalarAttr<true>>>,
                            Bind<Vec::CopyIn<U>, Placeholder::In6<U, Placeholder::ScalarAttr<true>>>>;
        using OpCopyInLrPower = std::conditional_t<std::is_same<U, bfloat16_t>::value,
                                Bind<Vec::Duplicate<U>, Placeholder::In7<U, Placeholder::ScalarAttr<true>>>,
                                Bind<Vec::CopyIn<U>, Placeholder::In7<U, Placeholder::ScalarAttr<true>>>>;

        using OpVarCast = Bind<Vec::Cast<T, U, 0>, OpCopyInVar>;
        using OpAccumCast = Bind<Vec::Cast<T, U, 0>, OpCopyInAccum>;
        using OpLinearCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLinear>;
        using OpGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyInGrad>;
        using OpLrCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
        using OpL1Cast = Bind<Vec::Cast<T, U, 0>, OpCopyInL1>;
        using OpL2Cast = Bind<Vec::Cast<T, U, 0>, OpCopyInL2>;
        using OpLrPowerCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLrPower>;

        using OpGradSquare = Bind<Vec::Mul<T>, OpGradCast, OpGradCast>;
        using OpAccumNew = Bind<Vec::Add<T>, OpAccumCast, OpGradSquare>;
        using OpAccumOutCast = Bind<Vec::Cast<U, T, 1>, OpAccumNew>;
        using OpCopyOutAccum = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpAccumOutCast>; // update input1: accum

        using OpNegOne = MAKE_CONST(T, -1);
        using OpIndex = Bind<Vec::Muls<T>, OpLrPowerCast, OpNegOne>;
        using OpAccNewPower = Bind<Vec::Power<T>, OpAccumNew, OpIndex>;
        using OpAccPower = Bind<Vec::Power<T>, OpAccumCast, OpIndex>;
        using OpAccSub = Bind<Vec::Sub<T>, OpAccNewPower, OpAccPower>;
        using OpAccSMulVar = Bind<Vec::Mul<T>, OpAccSub, OpVarCast>;
        using OpAccSMulVarDivLr = Bind<Vec::Div<T>, OpAccSMulVar, OpLrCast>;
        using OpGradSub = Bind<Vec::Sub<T>, OpGradCast, OpAccSMulVarDivLr>;
        using OpLinearNew = Bind<Vec::Add<T>, OpLinearCast, OpGradSub>;
        using OpLinearOutCast = Bind<Vec::Cast<U, T, 1>, OpLinearNew>;
        using OpCopyOutLinear = Bind<Vec::CopyOut<U>, Placeholder::Out2<U>, OpLinearOutCast>; // update input2: linear

        using OpOne = MAKE_CONST(T, 1);
        using OpZero = MAKE_CONST(T, 0);
        using OpZeroTensor = Bind<Vec::Duplicate<T>, OpZero>;
        using OpSignCmpPos = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_GT>, OpLinearNew, OpZero>;
        using OpSignCmpNeg = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_LT>, OpLinearNew, OpZero>;
        using OpSignSelectPos = Bind<Vec::Select<uint8_t, T, SELECT_MODE_T_S>, OpSignCmpPos, OpOne, OpZeroTensor>;
        using OpSignSelectNeg = Bind<Vec::Select<uint8_t, T, SELECT_MODE_T_S>, OpSignCmpNeg, OpOne, OpZeroTensor>;
        using OpSign = Bind<Vec::Sub<T>, OpSignSelectPos, OpSignSelectNeg>;
        using OpSignMulL1 = std::conditional_t<std::is_same<U, bfloat16_t>::value,
                            Bind<Vec::Mul<T>, OpSign, OpL1Cast>,
                            Bind<Vec::Muls<T>, OpSign, OpL1Cast>>;
        using OpMulL1SubLinear = Bind<Vec::Sub<T>, OpSignMulL1, OpLinearNew>;
        using OpAccNewPowerDiv = Bind<Vec::Div<T>, OpAccNewPower, OpLrCast>;
        using OpTwo = MAKE_CONST(T, 2);
        using OpL2Mul2 = Bind<Vec::Muls<T>, OpL2Cast, OpTwo>;
        using OpAdd2L2 = std::conditional_t<std::is_same<U, bfloat16_t>::value,
                            Bind<Vec::Add<T>, OpAccNewPowerDiv, OpL2Mul2>,
                            Bind<Vec::Adds<T>, OpAccNewPowerDiv, OpL2Mul2>>;
        using OpVarSelectTrue = Bind<Vec::Div<T>, OpMulL1SubLinear, OpAdd2L2>;
        using OpLinearAbs = Bind<Vec::Abs<T>, OpLinearNew>;
        using OpAbsCmpL1 = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_GT>, OpLinearAbs, OpL1Cast>;
        using OpVarNew = Bind<Vec::Select<uint8_t, T, SELECT_MODE_T_S>, OpAbsCmpL1, OpVarSelectTrue, OpZero>;

        using OpVarOutCast = Bind<Vec::Cast<U, T, 1>, OpVarNew>;
        using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>; // output: var

        using Outputs = Elems<OpCopyOutVar, OpCopyOutAccum, OpCopyOutLinear>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };
}

#endif  // CANN_CUSTOM_OPS_APPLY_FTRL_DAG_H 