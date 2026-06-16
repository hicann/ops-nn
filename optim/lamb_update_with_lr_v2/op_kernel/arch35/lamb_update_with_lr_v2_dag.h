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
 * \file    lamb_update_with_lr_v2_dag.h
 * \brief lamb_update_with_lr_v2_dag head file
 */

#ifndef LAMB_UPDATE_WITH_LR_V2_DAG_H
#define LAMB_UPDATE_WITH_LR_V2_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LambUpdateWithLrV2Op {
using namespace AscendC;
using namespace Ops::Base;
constexpr int COMPARE_MODE_GT = 1;
constexpr int SELECT_MODE_T_T = 2;
// In : 0 x1(w_norm) 1 x2(g_norm) 2 x3(lr) 3 x4(update, full) 4 x5(param, full)
//      5 greater_y(threshold) 6 select_e(fallback)   [0,1,2,5,6 scalar]
// Out: 0 y = x5 - x3 * ratio * x4
// ratio = where(x1>greater_y, where(x2>greater_y, x1/x2, select_e), select_e). Compute in U (=float).
template <typename T, typename U>
struct LambUpdateWithLrV2Compute {
    using InX1 = Bind<Vec::Duplicate<T>, Placeholder::In0<T, Placeholder::ScalarAttr<true>>>;
    using InX2 = Bind<Vec::Duplicate<T>, Placeholder::In1<T, Placeholder::ScalarAttr<true>>>;
    using InX3 = Bind<Vec::Duplicate<T>, Placeholder::In2<T, Placeholder::ScalarAttr<true>>>;
    using InX4 = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;
    using InX5 = Bind<Vec::CopyInBrc<T>, Placeholder::In4<T>>;
    using InGreaterY = Bind<Vec::Duplicate<T>, Placeholder::In5<T, Placeholder::ScalarAttr<true>>>;
    using InSelectE = Bind<Vec::Duplicate<T>, Placeholder::In6<T, Placeholder::ScalarAttr<true>>>;

    using X1 = Bind<Vec::Cast<U, T, 0>, InX1>;
    using X2 = Bind<Vec::Cast<U, T, 0>, InX2>;
    using X3 = Bind<Vec::Cast<U, T, 0>, InX3>;
    using X4 = Bind<Vec::Cast<U, T, 0>, InX4>;
    using X5 = Bind<Vec::Cast<U, T, 0>, InX5>;
    using GreaterY = Bind<Vec::Cast<U, T, 0>, InGreaterY>;
    using SelectE = Bind<Vec::Cast<U, T, 0>, InSelectE>;

    using Greater0 = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, X1, GreaterY>;
    using Greater1 = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, X2, GreaterY>;
    using TrueDiv0 = Bind<Vec::Div<U>, X1, X2>;
    using Select0 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, Greater1, TrueDiv0, SelectE>;
    using Ratio = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, Greater0, Select0, SelectE>;

    using Sub0 = Bind<Vec::Sub<U>, X5, Bind<Vec::Mul<U>, Bind<Vec::Mul<U>, X3, Ratio>, X4>>;
    using YCast = Bind<Vec::Cast<T, U, 1>, Sub0>;

    using OpOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, YCast>;

    using Outputs = Elems<OpOut0>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LambUpdateWithLrV2Op
#endif // LAMB_UPDATE_WITH_LR_V2_DAG_H
