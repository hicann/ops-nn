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
 * \file    lamb_update_with_lr_dag.h
 * \brief lamb_update_with_lr_dag head file
 */

#ifndef LAMB_UPDATE_WITH_LR_DAG_H
#define LAMB_UPDATE_WITH_LR_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LambUpdateWithLrOp {
using namespace AscendC;
using namespace Ops::Base;
constexpr int COMPARE_MODE_GT = 1;
constexpr int SELECT_MODE_T_T = 2;
// In : 0 input_greater1 1 input_greater_realdiv 2 input_realdiv 3 input_mul0(lr) 4 input_mul1(update, full)
//      5 input_sub(param, full) 6 greater_y 7 select_e 8 minimum_y   [0,1,2,3,6,7,8 scalar]
// Out: 0 y. ratio = where(in0>gy, in1/in2, se) gated again by where(in1>gy, ., se);
//      clip = max(min(ratio, minimum_y), greater_y); y = param - clip*lr*update. Compute in U (=float).
template <typename T, typename U>
struct LambUpdateWithLrCompute {
    using InG1 = Bind<Vec::Duplicate<T>, Placeholder::In0<T, Placeholder::ScalarAttr<true>>>;
    using InGRd = Bind<Vec::Duplicate<T>, Placeholder::In1<T, Placeholder::ScalarAttr<true>>>;
    using InRd = Bind<Vec::Duplicate<T>, Placeholder::In2<T, Placeholder::ScalarAttr<true>>>;
    using InLr = Bind<Vec::Duplicate<T>, Placeholder::In3<T, Placeholder::ScalarAttr<true>>>;
    using InUpd = Bind<Vec::CopyInBrc<T>, Placeholder::In4<T>>;
    using InParam = Bind<Vec::CopyInBrc<T>, Placeholder::In5<T>>;
    using InGreaterY = Bind<Vec::Duplicate<T>, Placeholder::In6<T, Placeholder::ScalarAttr<true>>>;
    using InSelectE = Bind<Vec::Duplicate<T>, Placeholder::In7<T, Placeholder::ScalarAttr<true>>>;
    using InMinY = Bind<Vec::Duplicate<T>, Placeholder::In8<T, Placeholder::ScalarAttr<true>>>;

    using G1 = Bind<Vec::Cast<U, T, 0>, InG1>;
    using GRd = Bind<Vec::Cast<U, T, 0>, InGRd>;
    using Rd = Bind<Vec::Cast<U, T, 0>, InRd>;
    using Lr = Bind<Vec::Cast<U, T, 0>, InLr>;
    using Upd = Bind<Vec::Cast<U, T, 0>, InUpd>;
    using Param = Bind<Vec::Cast<U, T, 0>, InParam>;
    using GreaterY = Bind<Vec::Cast<U, T, 0>, InGreaterY>;
    using SelectE = Bind<Vec::Cast<U, T, 0>, InSelectE>;
    using MinY = Bind<Vec::Cast<U, T, 0>, InMinY>;

    using Greater0 = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, G1, GreaterY>;
    using Greater1 = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, GRd, GreaterY>;
    using RealDiv0 = Bind<Vec::Div<U>, GRd, Rd>;
    using Select0 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, Greater0, RealDiv0, SelectE>;
    using Select1 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, Greater1, Select0, SelectE>;
    using Minimum0 = Bind<Vec::Min<U>, Select1, MinY>;
    using Maximum0 = Bind<Vec::Max<U>, Minimum0, GreaterY>;

    using Res = Bind<Vec::Sub<U>, Param, Bind<Vec::Mul<U>, Bind<Vec::Mul<U>, Maximum0, Lr>, Upd>>;
    using YCast = Bind<Vec::Cast<T, U, 1>, Res>;

    using OpOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, YCast>;

    using Outputs = Elems<OpOut0>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LambUpdateWithLrOp
#endif // LAMB_UPDATE_WITH_LR_DAG_H
