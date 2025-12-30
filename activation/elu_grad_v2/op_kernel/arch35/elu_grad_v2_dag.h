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
 * \file elu_grad_v2_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_ELU_GRAD_V2_DAG_H
#define CANN_CUSTOM_OPS_ELU_GRAD_V2_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

const int COMPARE_MODE_LE = 3;
const int PLACEHOLDER_INDEX_2 = 2;
const int SELECT_MODE_T_T = 2;
template <typename T>
struct EluGradV2IsResultOp {
    // 通过Compute构造计算图
    // negcoef = scale * alpha
    // y = grads * scale activations > 0
    //   = grads * input_scale * (activations + negcoef) activations <= 0
    using ConstValue = MAKE_CONST(float, 0.0);

    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using Cast0 = Bind<Vec::Cast<float, T, 0>, OpCopyIn0>;
    using Cast1 = Bind<Vec::Cast<float, T, 0>, OpCopyIn1>;
    using OpAdds = Bind<Vec::Adds<float>, Cast1, Placeholder::Var<float, 0>>;
    using OpMuls1 = Bind<Vec::Muls<float>, Cast0, Placeholder::Var<float, PLACEHOLDER_INDEX_2>>;
    using OpMuls2 = Bind<Vec::Mul<float>, OpMuls1, OpAdds>;
    using OpMuls0 = Bind<Vec::Muls<float>, Cast0, Placeholder::Var<float, 1>>;
    using OpCompare = Bind<Vec::Compare<uint8_t, float, COMPARE_MODE_LE>, Cast1, ConstValue>;
    using OpSelect = Bind<Vec::Select<uint8_t, float, SELECT_MODE_T_T>, OpCompare, OpMuls2, OpMuls0>;
    using OpResultCast = Bind<Vec::Cast<T, float, 1>, OpSelect>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResultCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;  // 设置输出
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct EluGradV2NoResultOp {
    // 通过Compute构造计算图
    // negcoef = scale * alpha
    // y = grads * scale activations > 0
    //   = grads * input_scale * negcoef * exp(activations * input_scale) activations <= 0
    using ConstValue = MAKE_CONST(float, 0.0);

    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using Cast0 = Bind<Vec::Cast<float, T, 0>, OpCopyIn0>;
    using Cast1 = Bind<Vec::Cast<float, T, 0>, OpCopyIn1>;
    using OpMuls1 = Bind<Vec::Muls<float>, Cast1, Placeholder::Var<float, PLACEHOLDER_INDEX_2>>;
    using OpExp = Bind<Vec::Exp<float>, OpMuls1>;
    using OpMuls2 = Bind<Vec::Muls<float>, Cast0, Placeholder::Var<float, PLACEHOLDER_INDEX_2>>;
    using OpMuls3 = Bind<Vec::Muls<float>, OpMuls2,  Placeholder::Var<float, 0>>;
    using OpMuls4 = Bind<Vec::Mul<float>, OpMuls3,  OpExp>;
    using OpMuls0 = Bind<Vec::Muls<float>, Cast0, Placeholder::Var<float, 1>>;
    using OpCompare = Bind<Vec::Compare<uint8_t, float, COMPARE_MODE_LE>, Cast1, ConstValue>;
    using OpSelect = Bind<Vec::Select<uint8_t, float, SELECT_MODE_T_T>, OpCompare, OpMuls4, OpMuls0>;
    using OpResultCast = Bind<Vec::Cast<T, float, 1>, OpSelect>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResultCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;  // 设置输出
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
#endif  // CANN_CUSTOM_OPS_ELU_GRAD_V2_DAG_H