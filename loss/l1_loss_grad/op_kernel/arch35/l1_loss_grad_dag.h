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
 * \file l1_loss_grad_dag.h
 * \brief l1_loss_grad_dag head file
 */
#ifndef ASCENDC_L1_LOSS_GRAD_DAG_H_
#define ASCENDC_L1_LOSS_GRAD_DAG_H_
#include <cmath>
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/elems.h"

using namespace Ops::Base;

namespace L1LossGradKernel {
using namespace AscendC;
const int COMPARE_MODE_LT = 0;
const int COMPARE_MODE_GT = 1;
const int SELECT_MODE_SCALAR = 1;
const int SELECT_MODE_TENSOR = 2;

template <typename U, typename T = float>
struct L1LossGradCast {
    constexpr static int CAST_MODE_RINT = 1;
    using ConstValueZero = MAKE_CONST(T, 0.0);
    using ConstValueOne = MAKE_CONST(T, 1.0);
    using ConstValueNegOne = MAKE_CONST(T, -1.0);
    using OpCopyGrads = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyPredict = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyLabel = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyGradsCast = Bind<Vec::Cast<T, U, 0>, OpCopyGrads>;

    using OpCompareGT = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, OpCopyPredict, OpCopyLabel>;
    using OpOneTensor = Bind<Vec::Duplicate<T>, ConstValueOne>;
    using OpSelect = Bind<Vec::Select<uint8_t, T, SELECT_MODE_SCALAR>, OpCompareGT, OpOneTensor, ConstValueZero>;

    using OpCompareLT = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_LT>, OpCopyPredict, OpCopyLabel>;
    using OpNegOneTensor = Bind<Vec::Duplicate<T>, ConstValueNegOne>;
    using OpSign = Bind<Vec::Select<uint8_t, T, SELECT_MODE_TENSOR>, OpCompareLT, OpNegOneTensor, OpSelect>;

    using OpGrads = Bind<Vec::Muls<T>, OpCopyGradsCast, Placeholder::Var<T, 0>>;

    using OpResCast = Bind<Vec::Mul<T>, OpSign, OpGrads>;
    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResCast>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;

    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct L1LossGradScalarCast {
    constexpr static int CAST_MODE_RINT = 1;
    using ConstValueZero = MAKE_CONST(T, 0.0);
    using ConstValueOne = MAKE_CONST(T, 1.0);
    using ConstValueNegOne = MAKE_CONST(T, -1.0);
    using OpCopyPredict = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyLabel = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCompareGT = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, OpCopyPredict, OpCopyLabel>;
    using OpOneTensor = Bind<Vec::Duplicate<T>, ConstValueOne>;
    using OpSelect = Bind<Vec::Select<uint8_t, T, SELECT_MODE_SCALAR>, OpCompareGT, OpOneTensor, ConstValueZero>;

    using OpCompareLT = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_LT>, OpCopyPredict, OpCopyLabel>;
    using OpNegOneTensor = Bind<Vec::Duplicate<T>, ConstValueNegOne>;
    using OpSign = Bind<Vec::Select<uint8_t, T, SELECT_MODE_TENSOR>, OpCompareLT, OpNegOneTensor, OpSelect>;

    using OpCopyGrads = Bind<Vec::Duplicate<U>,  Placeholder::In0<U, Placeholder::ScalarAttr<true>>>;
    using OpCopyGradsCast = Bind<Vec::Cast<T, U, 0>, OpCopyGrads>;
    using OpGrads = Bind<Vec::Muls<T>, OpCopyGradsCast, Placeholder::Var<T, 0>>;

    using OpResCast = Bind<Vec::Mul<T>, OpSign, OpGrads>;
    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResCast>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;

    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace L1LossGradKernel
#endif //ASCENDC_L1_LOSS_GRAD_DAG_H_