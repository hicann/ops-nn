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
 * \file renorm_dag.h
 * \brief renorm dag
 */

#ifndef RENORM_DAG_H
#define RENORM_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

using namespace Ops::Base;

namespace Renorm
{

constexpr uint32_t TEMPLATE_P0 = 0;       // 0.0
constexpr uint32_t TEMPLATE_P1 = 1;       // 1.0
constexpr uint32_t TEMPLATE_P2 = 2;       // 2.0
constexpr uint32_t TEMPLATE_P3 = 3;       // 3.0
constexpr uint32_t TEMPLATE_P_NINF = 4;   // -inf
constexpr uint32_t TEMPLATE_P_INF = 5;    // inf
constexpr uint32_t TEMPLATE_P_OTHER = 6;  // other

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;
constexpr int COMPARE_MODE_EQ = 2;
constexpr int SEL_MODE_TENSOR_SCALAR = 1;

constexpr int VAR_INDEX_1 = 1;
constexpr int VAR_INDEX_2 = 2;
constexpr int VAR_INDEX_3 = 3;

using namespace AscendC;

// sum(x!=0)
template <typename T, typename PromoteT, typename OutT>
struct RenormP0Dag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using Cast0 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyIn0>;
    using ConstZero = MAKE_CONST(PromoteT, 0);
    using OpMask = Bind<Vec::Compare<uint8_t, PromoteT, COMPARE_MODE_EQ>, Cast0, ConstZero>;  // EQ
    using ConstOne = MAKE_CONST(PromoteT, 1);
    using DupZero = Bind<Vec::Duplicate<PromoteT>, ConstZero>;
    using OpSel =
        Bind<Vec::Select<uint8_t, PromoteT, SEL_MODE_TENSOR_SCALAR>, OpMask, DupZero, ConstOne>;  // tensor scalar
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, OpSel>;

    // 竞品没有max, epsilon通过tiling参数传递
    using OpMaxs = Bind<Vec::Maxs<PromoteT>, ReduceOp0, Placeholder::Var<PromoteT, 0>>;

    // 计算缩放因子
    using Mins0 = Bind<Vec::Mins<PromoteT>, OpMaxs, Placeholder::Var<PromoteT, 1>>;
    using Div0 = Bind<Vec::Div<PromoteT>, Mins0, OpMaxs>;

    using Cast1 = Bind<Vec::Cast<OutT, PromoteT, CAST_MODE_RINT>, Div0>;

    using OpCopyOut = Bind<Vec::CopyOut<OutT>, Placeholder::Out0<OutT>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromoteT, typename OutT>
struct RenormP1Dag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using Cast0 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyIn0>;
    using Abs0 = Bind<Vec::Abs<PromoteT>, Cast0>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, Abs0>;

    // 竞品没有max, epsilon通过tiling参数传递
    using OpMaxs = Bind<Vec::Maxs<PromoteT>, ReduceOp0, Placeholder::Var<PromoteT, 0>>;
    // 计算缩放因子
    using Mins0 = Bind<Vec::Mins<PromoteT>, OpMaxs, Placeholder::Var<PromoteT, 1>>;
    using Div0 = Bind<Vec::Div<PromoteT>, Mins0, OpMaxs>;

    using Cast1 = Bind<Vec::Cast<OutT, PromoteT, CAST_MODE_RINT>, Div0>;

    using OpCopyOut = Bind<Vec::CopyOut<OutT>, Placeholder::Out0<OutT>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// sqrt(sum(x*x))
template <typename T, typename PromoteT, typename OutT>
struct RenormP2Dag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using Cast0 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyIn0>;
    using Mul0 = Bind<Vec::Mul<PromoteT>, Cast0, Cast0>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, Mul0>;
    using Sqrt0 = Bind<Vec::Sqrt<PromoteT>, ReduceOp0>;

    // 竞品没有max, epsilon通过tiling参数传递
    using OpMaxs = Bind<Vec::Maxs<PromoteT>, Sqrt0, Placeholder::Var<PromoteT, 0>>;

    // 计算缩放因子
    using Mins0 = Bind<Vec::Mins<PromoteT>, OpMaxs, Placeholder::Var<PromoteT, 1>>;
    using Div0 = Bind<Vec::Div<PromoteT>, Mins0, OpMaxs>;

    using Cast1 = Bind<Vec::Cast<OutT, PromoteT, CAST_MODE_RINT>, Div0>;

    using OpCopyOut = Bind<Vec::CopyOut<OutT>, Placeholder::Out0<OutT>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// sum(abs(x)^3)^(1/3)
template <typename T, typename PromoteT, typename OutT>
struct RenormP3Dag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using Cast0 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyIn0>;
    using Abs0 = Bind<Vec::Abs<PromoteT>, Cast0>;
    using Mul0 = Bind<Vec::Mul<PromoteT>, Abs0, Abs0>;
    using Mul1 = Bind<Vec::Mul<PromoteT>, Mul0, Abs0>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, Mul1>;
    using Pow1 = Bind<Vec::Power<PromoteT>, ReduceOp0, Placeholder::Var<PromoteT, VAR_INDEX_1>>;

    // 竞品没有max, epsilon通过tiling参数传递
    using OpMaxs = Bind<Vec::Maxs<PromoteT>, Pow1, Placeholder::Var<PromoteT, 0>>;

    // 计算缩放因子
    using Mins0 = Bind<Vec::Mins<PromoteT>, OpMaxs, Placeholder::Var<PromoteT, 2>>;
    using Div0 = Bind<Vec::Div<PromoteT>, Mins0, OpMaxs>;
   
    using Cast1 = Bind<Vec::Cast<OutT, PromoteT, CAST_MODE_RINT>, Div0>;

    using OpCopyOut = Bind<Vec::CopyOut<OutT>, Placeholder::Out0<OutT>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// // inf: max(abs(x))
template <typename T, typename PromoteT = T, typename OutT = T>
struct RenormPInfDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using Cast0 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyIn0>;
    using Abs0 = Bind<Vec::Abs<PromoteT>, Cast0>;
    using ReduceOp0 = Bind<Vec::ReduceMaxOp<PromoteT>, Abs0>;

    // 竞品没有max, epsilon通过tiling参数传递
    using OpMaxs = Bind<Vec::Maxs<PromoteT>, ReduceOp0, Placeholder::Var<PromoteT, 0>>;

    // 计算缩放因子
    using Mins0 = Bind<Vec::Mins<PromoteT>, OpMaxs, Placeholder::Var<PromoteT, 1>>;
    using Div0 = Bind<Vec::Div<PromoteT>, Mins0, OpMaxs>;
   
    using Cast1 = Bind<Vec::Cast<OutT, PromoteT, CAST_MODE_RINT>, Div0>;

    using OpCopyOut = Bind<Vec::CopyOut<OutT>, Placeholder::Out0<OutT>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

 // -inf: min(abs(x))
template <typename T, typename PromoteT = T, typename OutT = T>
struct RenormPNInfDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using Cast0 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyIn0>;
    using Abs0 = Bind<Vec::Abs<PromoteT>, Cast0>;
    using ReduceOp0 = Bind<Vec::ReduceMinOp<PromoteT>, Abs0>;

    // 竞品没有max, epsilon通过tiling参数传递
    using OpMaxs = Bind<Vec::Maxs<PromoteT>, ReduceOp0, Placeholder::Var<PromoteT, 0>>;

    // 计算缩放因子
    using Mins0 = Bind<Vec::Mins<PromoteT>, OpMaxs, Placeholder::Var<PromoteT, 1>>;
    using Div0 = Bind<Vec::Div<PromoteT>, Mins0, OpMaxs>;

    using Cast1 = Bind<Vec::Cast<OutT, PromoteT, CAST_MODE_RINT>, Div0>;

    using OpCopyOut = Bind<Vec::CopyOut<OutT>, Placeholder::Out0<OutT>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// sum(abs(x)^p)^(1/p)
template <typename T, typename PromoteT, typename OutT>
struct RenormPOtherDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using Cast0 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyIn0>;
    using Abs0 = Bind<Vec::Abs<PromoteT>, Cast0>;
    using Pow0 = Bind<Vec::Power<PromoteT>, Abs0, Placeholder::Var<PromoteT, VAR_INDEX_1>>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, Pow0>;
    using Pow1 = Bind<Vec::Power<PromoteT>, ReduceOp0, Placeholder::Var<PromoteT, VAR_INDEX_2>>;

    // 竞品没有max, epsilon通过tiling参数传递
    using OpMaxs = Bind<Vec::Maxs<PromoteT>, Pow1, Placeholder::Var<PromoteT, 0>>;

    // 计算缩放因子
    using Mins0 = Bind<Vec::Mins<PromoteT>, OpMaxs, Placeholder::Var<PromoteT, 3>>;
    using Div0 = Bind<Vec::Div<PromoteT>, Mins0, OpMaxs>;
   
    using Cast1 = Bind<Vec::Cast<OutT, PromoteT, CAST_MODE_RINT>, Div0>;    

    using OpCopyOut = Bind<Vec::CopyOut<OutT>, Placeholder::Out0<OutT>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

}  // namespace Renorm

#endif