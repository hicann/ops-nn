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
 * \file smooth_l1_loss_v2_dag.h
 * \brief
 */

#ifndef SMOOTH_L1_LOSS_V2_DAG_H
#define SMOOTH_L1_LOSS_V2_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"  

using namespace Ops::Base;
namespace SmoothL1LossV2 {
using namespace AscendC;
constexpr int COMPARE_MODE_LT = 0;
constexpr int SELECT_TENSOR = 2;
constexpr int SCALAR_INDEX_2 = 2;
constexpr int SCALAR_INDEX_3 = 3;
template <typename T, typename PromteT = float>
struct SmoothL1LossV2OpDag {
    // 通过Compute构造计算图
    // |xi - yi| < sigma ? (xi - yi) ** 2  * 0.5 / sigma  :  |xi - yi| - 0.5 * sigma
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using OpSub = Bind<Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpAbs = Bind<Vec::Abs<PromteT>, OpSub>;
    using ConstValueHalf = MAKE_CONST(PromteT, 0.5);
    using OpMuls = Bind<Vec::Muls<PromteT>, OpSub, ConstValueHalf>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpSub, OpMuls>;
    using OpMuls1 = Bind<Vec::Muls<PromteT>, OpMul, Placeholder::Var<PromteT, 1>>;
    using OpAdds = Bind<Vec::Adds<PromteT>, OpAbs, Placeholder::Var<PromteT, SCALAR_INDEX_2>>;
    using OpCommpare = Bind<Vec::Compare<uint8_t, PromteT, COMPARE_MODE_LT>, OpAbs, Placeholder::Var<PromteT, 0>>;
    using OpSelect = Bind<Vec::Select<uint8_t, PromteT, SELECT_TENSOR>, OpCommpare, OpMuls1, OpAdds>;
    using OpResultCast = Bind<Vec::Cast<T, PromteT, 1>, OpSelect>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResultCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;  // 设置输出
    // 指定计算顺序
    using OpDag = DAGSch<Outputs>;
};
template <typename T, typename PromteT>
struct SmoothL1LossV2SumDag {
    // 通过Compute构造计算图
    // |xi - yi| < sigma ? (xi - yi) ** 2  * 0.5 / sigma  :  |xi - yi| - 0.5 * sigma
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using OpSub = Bind<Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpAbs = Bind<Vec::Abs<PromteT>, OpSub>;
    using ConstValueHalf = MAKE_CONST(PromteT, 0.5);
    using OpMuls = Bind<Vec::Muls<PromteT>, OpSub, ConstValueHalf>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpSub, OpMuls>;
    using OpMuls1 = Bind<Vec::Muls<PromteT>, OpMul, Placeholder::Var<PromteT, 1>>;
    using OpAdds = Bind<Vec::Adds<PromteT>, OpAbs, Placeholder::Var<PromteT, SCALAR_INDEX_2>>;
    using OpCommpare = Bind<Vec::Compare<uint8_t, PromteT, COMPARE_MODE_LT>, OpAbs, Placeholder::Var<PromteT, 0>>;
    using OpSelect = Bind<Vec::Select<uint8_t, PromteT, SELECT_TENSOR>, OpCommpare, OpMuls1, OpAdds>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, OpSelect>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromteT>
struct SmoothL1LossV2MeanDag {
    // 通过Compute构造计算图
    // |xi - yi| < sigma ? (xi - yi) ** 2  * 0.5 / sigma  :  |xi - yi| - 0.5 * sigma
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using OpSub = Bind<Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpAbs = Bind<Vec::Abs<PromteT>, OpSub>;
    using ConstValueHalf = MAKE_CONST(PromteT, 0.5);
    using OpMuls = Bind<Vec::Muls<PromteT>, OpSub, ConstValueHalf>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpSub, OpMuls>;
    using OpMuls1 = Bind<Vec::Muls<PromteT>, OpMul, Placeholder::Var<PromteT, 1>>;
    using OpAdds = Bind<Vec::Adds<PromteT>, OpAbs, Placeholder::Var<PromteT, SCALAR_INDEX_2>>;
    using OpCommpare = Bind<Vec::Compare<uint8_t, PromteT, COMPARE_MODE_LT>, OpAbs, Placeholder::Var<PromteT, 0>>;
    using OpSelect = Bind<Vec::Select<uint8_t, PromteT, SELECT_TENSOR>, OpCommpare, OpMuls1, OpAdds>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, OpSelect>;
    using Mul0 = Bind<Vec::Muls<PromteT>, ReduceOp0, Placeholder::Var<PromteT, SCALAR_INDEX_3>>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, Mul0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace SmoothL1LossV2

#endif  // SMOOTH_L1_LOSS_V2_DAG_H