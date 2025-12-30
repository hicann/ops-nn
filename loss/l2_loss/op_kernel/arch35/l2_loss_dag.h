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
 * \file l2_loss_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_L2_LOSS_DAG_H
#define CANN_CUSTOM_OPS_L2_LOSS_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

constexpr float COEFF_SQRT = 0.70710678118;  // equals 1 / np.sqrt(2)
using namespace Ops::Base;
namespace L2Loss
{
using namespace AscendC;
template <typename T, typename PromteT>
struct L2LossDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    // 计算 1.0/(2^0.5)
    using ConstValue = MAKE_CONST(PromteT, COEFF_SQRT);
    using OpMuls = Bind<Vec::Muls<PromteT>, OpCopyIn0Cast, ConstValue>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpMuls, OpMuls>;
    using OpReduce = Bind<Vec::ReduceSumOp<PromteT>, OpMul>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, OpReduce>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace L2Loss

#endif  // CANN_CUSTOM_OPS_L2_LOSS_DAG_H