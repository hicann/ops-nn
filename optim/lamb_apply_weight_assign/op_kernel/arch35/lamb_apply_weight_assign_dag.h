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
 * \file    lamb_apply_weight_assign_dag.h
 * \brief lamb_apply_weight_assign_dag head file
 */

#ifndef LAMB_APPLY_WEIGHT_ASSIGN_DAG_H
#define LAMB_APPLY_WEIGHT_ASSIGN_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LambApplyWeightAssignOp {
using namespace AscendC;
using namespace Ops::Base;
constexpr int COMPARE_MODE_GT = 1;
constexpr int SELECT_MODE_T_T = 2;
// In : 0 input0(w_norm, scalar), 1 input1(g_norm, scalar), 2 input2(lr, scalar),
//      3 input3(update, full), 4 input_param(param, full)
// Out: 0 input_param(next_param, in-place)
// ratio = where(w_norm>0, where(g_norm>0, w_norm/g_norm, 1), 1); next_param = param - ratio*(update*lr).
// Compute in U (=float): half casts to float (div needs float precision); fp32 native.
template <typename T, typename U>
struct LambApplyWeightAssignCompute {
    using CZero = MAKE_CONST(U, 0.0);
    using COne = MAKE_CONST(U, 1.0);

    using InWNorm = Bind<Vec::Duplicate<T>, Placeholder::In0<T, Placeholder::ScalarAttr<true>>>;
    using InGNorm = Bind<Vec::Duplicate<T>, Placeholder::In1<T, Placeholder::ScalarAttr<true>>>;
    using InLr = Bind<Vec::Duplicate<T>, Placeholder::In2<T, Placeholder::ScalarAttr<true>>>;
    using InUpdate = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;
    using InParam = Bind<Vec::CopyInBrc<T>, Placeholder::In4<T>>;

    using WNorm = Bind<Vec::Cast<U, T, 0>, InWNorm>;
    using GNorm = Bind<Vec::Cast<U, T, 0>, InGNorm>;
    using Lr = Bind<Vec::Cast<U, T, 0>, InLr>;
    using Update = Bind<Vec::Cast<U, T, 0>, InUpdate>;
    using Param = Bind<Vec::Cast<U, T, 0>, InParam>;

    // trust ratio: where(w_norm>0, where(g_norm>0, w_norm/g_norm, 1), 1)
    using OneTensor = Bind<Vec::Duplicate<U>, COne>;
    using GreaterG = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, GNorm, CZero>;
    using GreaterW = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, WNorm, CZero>;
    using WDivG = Bind<Vec::Div<U>, WNorm, GNorm>;
    using Select1 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, GreaterG, WDivG, OneTensor>;
    using Ratio = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, GreaterW, Select1, OneTensor>;

    // next_param = param - ratio * (update * lr)
    using UpdLr = Bind<Vec::Mul<U>, Update, Lr>;
    using RatioUpdLr = Bind<Vec::Mul<U>, Ratio, UpdLr>;
    using NextParam = Bind<Vec::Sub<U>, Param, RatioUpdLr>;
    using NextParamCast = Bind<Vec::Cast<T, U, 1>, NextParam>;

    using OpCopyOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, NextParamCast>;

    using Outputs = Elems<OpCopyOut0>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LambApplyWeightAssignOp
#endif // LAMB_APPLY_WEIGHT_ASSIGN_DAG_H
