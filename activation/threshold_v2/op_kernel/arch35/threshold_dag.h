/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file threshold_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_THRESHOLD_DAG_H
#define CANN_CUSTOM_OPS_THRESHOLD_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace ThresholdOp {
using namespace Ops::Base;

constexpr int CAST_NONE = 0;
constexpr int CAST_RINT = 1;
constexpr int VSEL_TENSOR_TENSOR_MODE = 2;
constexpr int CMPMODE_LE = 3;

template <typename U, typename T>
struct ToFloatScalar : public Vec::ElemwiseUnaryOP<U, T> {
    __aicore__ inline ToFloatScalar(U& dst, T& scalar, int /*count*/)
    {
#ifdef __CCE_AICORE__
        dst = ToFloat(scalar);
#endif
    }
};

template <typename T>
struct ThresholdDag {
    using OpCopyInX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using ThresholdScalarHolder = Placeholder::In1<T, Placeholder::ScalarAttr<1>>;
    using ValueScalarHolder = Placeholder::In2<T, Placeholder::ScalarAttr<1>>;

    using OpCompare = Bind<Vec::Compare<uint8_t, T, CMPMODE_LE>, OpCopyInX, ThresholdScalarHolder>;
    using OpSelectResult =
        Bind<Vec::Select<uint8_t, T, VSEL_TENSOR_TENSOR_MODE>, OpCompare, ValueScalarHolder, OpCopyInX>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSelectResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct ThresholdDagNoValue {
    using OpCopyInX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using ThresholdScalarHolder = Placeholder::In1<T, Placeholder::ScalarAttr<1>>;

    using OpCompare = Bind<Vec::Compare<uint8_t, T, CMPMODE_LE>, OpCopyInX, ThresholdScalarHolder>;
    using ConstValueZero = MAKE_CONST(T, 0);
    using OpSelectResult = Bind<Vec::Select<uint8_t, T, VSEL_TENSOR_TENSOR_MODE>, OpCompare, ConstValueZero, OpCopyInX>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSelectResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename U>
struct ThresholdCastDag {
    using OpCopyInX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using ThresholdScalarHolder = Placeholder::In1<T, Placeholder::ScalarAttr<1>>;
    using ValueScalarHolder = Placeholder::In2<T, Placeholder::ScalarAttr<1>>;

    using OpCopyInXCast = Bind<Vec::Cast<U, T, CAST_NONE>, OpCopyInX>;
    using ThresholdScalarHolderCast = Bind<ToFloatScalar<U, T>, ThresholdScalarHolder>;
    using ValueScalarHolderCast = Bind<ToFloatScalar<U, T>, ValueScalarHolder>;

    using OpCompare = Bind<Vec::Compare<uint8_t, U, CMPMODE_LE>, OpCopyInXCast, ThresholdScalarHolderCast>;
    using OpSelect =
        Bind<Vec::Select<uint8_t, U, VSEL_TENSOR_TENSOR_MODE>, OpCompare, ValueScalarHolderCast, OpCopyInXCast>;
    using OpSelectResult = Bind<Vec::Cast<T, U, CAST_RINT>, OpSelect>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSelectResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename U>
struct ThresholdCastDagNoValue {
    using OpCopyInX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using ThresholdScalarHolder = Placeholder::In1<T, Placeholder::ScalarAttr<1>>;

    using OpCopyInXCast = Bind<Vec::Cast<U, T, CAST_NONE>, OpCopyInX>;
    using ThresholdScalarHolderCast = Bind<ToFloatScalar<U, T>, ThresholdScalarHolder>;

    using OpCompare = Bind<Vec::Compare<uint8_t, U, CMPMODE_LE>, OpCopyInXCast, ThresholdScalarHolderCast>;
    using ConstValueZero = MAKE_CONST(U, 0);
    using OpSelect = Bind<Vec::Select<uint8_t, U, VSEL_TENSOR_TENSOR_MODE>, OpCompare, ConstValueZero, OpCopyInXCast>;
    using OpSelectResult = Bind<Vec::Cast<T, U, CAST_RINT>, OpSelect>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSelectResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}

#endif // CANN_CUSTOM_OPS_THRESHOLD_DAG_H