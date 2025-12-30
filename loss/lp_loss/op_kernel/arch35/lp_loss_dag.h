/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file lp_loss_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_LP_LOSS_DAG_H
#define CANN_CUSTOM_OPS_LP_LOSS_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

namespace LpLoss
{
using namespace AscendC;

template <typename T, typename PromteT = float>
struct LpLossOp {
    // 通过Compute构造计算图
    // a-b
    using OpCopyIn0 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In0<T>>;
    using OpCopyIn1 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In1<T>>;

    using OpCopyIn0Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpCopyIn1>;

    using OpSub = Ops::Base::Bind<Ops::Base::Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpAbs = Ops::Base::Bind<Ops::Base::Vec::Abs<PromteT>, OpSub>;

    using OpResultCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, PromteT, 1>, OpAbs>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpResultCast>;
    // 指定输出节点
    using Outputs = Ops::Base::Elems<OpCopyOut>;  // 设置输出
    // 指定计算顺序
    using OpDag = Ops::Base::DAGSch<Outputs>;
};
template <typename T, typename PromteT>
struct LpLossSumDag {
    using OpCopyIn0 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In0<T>>;
    using OpCopyIn1 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In1<T>>;
    using OpCopyIn0Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using OpSub = Ops::Base::Bind<Ops::Base::Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpAbs = Ops::Base::Bind<Ops::Base::Vec::Abs<PromteT>, OpSub>;
    using ReduceOp0 = Ops::Base::Bind<Ops::Base::Vec::ReduceSumOp<PromteT>, OpAbs>;
    using Cast1 = Ops::Base::Bind<Ops::Base::Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, Cast1>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromteT>
struct LpLossMeanDag {
    using OpCopyIn0 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In0<T>>;
    using OpCopyIn1 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In1<T>>;
    using OpCopyIn0Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using OpSub = Ops::Base::Bind<Ops::Base::Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpAbs = Ops::Base::Bind<Ops::Base::Vec::Abs<PromteT>, OpSub>;
    using ReduceOp0 = Ops::Base::Bind<Ops::Base::Vec::ReduceSumOp<PromteT>, OpAbs>;
    using Mul0 = Ops::Base::Bind<Ops::Base::Vec::Muls<PromteT>, ReduceOp0, Ops::Base::Placeholder::Var<PromteT, 0>>;
    using Cast1 = Ops::Base::Bind<Ops::Base::Vec::Cast<T, PromteT, 1>, Mul0>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, Cast1>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};
}  // namespace LpLoss

#endif  // CANN_CUSTOM_OPS_LP_LOSS_DAG_H