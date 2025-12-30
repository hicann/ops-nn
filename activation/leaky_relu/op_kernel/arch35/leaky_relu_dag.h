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
 * \file leaky_relu_dag.h
 * \brief
 */
#ifndef ASCENDC_LEAKY_RELU_DAG_H_
#define ASCENDC_LEAKY_RELU_DAG_H_
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace AscendC;
using namespace Ops::Base;
template <typename U, typename T = float>
struct LeakyReluDag {
    using OpCopyInX = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpLeakRelu = Bind<Vec::LeakyRelu<U>, OpCopyInX, Placeholder::Var<T, 0>>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpLeakRelu>;
    using Outputs = Elems<OpCopyOut>;
    using OpDag = DAGSch<Outputs>;
};

template <typename U, typename T = float>
struct LeakyReluCastDag {
    using OpCopyInX = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyInXCast = Bind<Vec::Cast<T, U, 0>, OpCopyInX>;
    using OpLeakRelu = Bind<Vec::LeakyRelu<T>, OpCopyInXCast, Placeholder::Var<T, 0>>;
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyOutCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpLeakRelu>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyOutCast>;
    using Outputs = Elems<OpCopyOut>;
    using OpDag = DAGSch<Outputs>;
};

#endif //ASCENDC_LEAKY_RELU_DAG_H_