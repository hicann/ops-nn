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
 * \file prelu_grad_reduce_dag.h
 * \brief prelu_grad_reduce_dag
 */

#ifndef PRELU_GRAD_REDUCE_DAG_H
#define PRELU_GRAD_REDUCE_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

namespace PreluGradReduce
{
using namespace Ops::Base;
using namespace AscendC;

template <typename T, typename PromteT>
struct PreluGradReduceDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace PreluGradReduce

#endif