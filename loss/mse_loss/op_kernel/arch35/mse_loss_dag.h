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
 * \file mse_loss_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_MSE_LOSS_DAG_H
#define CANN_CUSTOM_OPS_MSE_LOSS_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

namespace MseLoss {
using namespace Ops::Base;
using namespace AscendC;

template <typename T, typename PromteT = float>
struct MseLossOp {
    // 通过Compute构造计算图
    // (a-b)^2
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;

    using OpCopyIn0Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;

    using OpSub = Bind<Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpSub, OpSub>;

    using OpResultCast = Bind<Vec::Cast<T, PromteT, 1>, OpMul>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResultCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    // 指定计算顺序
    using OpDag = DAGSch<Outputs>;
};
template <typename T, typename PromteT>
struct MseLossSumDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using OpSub = Bind<Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpSub, OpSub>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, OpMul>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromteT>
struct MseLossMeanDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using OpSub = Bind<Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpSub, OpSub>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, OpMul>;
    using Mul0 = Bind<Vec::Muls<PromteT>, ReduceOp0, Placeholder::Var<PromteT, 0>>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, Mul0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace MseLoss

#endif // CANN_CUSTOM_OPS_MSE_LOSS_DAG_H