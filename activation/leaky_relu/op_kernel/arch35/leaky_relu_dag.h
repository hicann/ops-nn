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
#ifndef OPS_NN_ACTIVATION_LEAKY_RELU_OP_KERNEL_ARCH35_LEAKY_RELU_DAG_H
#define OPS_NN_ACTIVATION_LEAKY_RELU_OP_KERNEL_ARCH35_LEAKY_RELU_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

template <typename U, typename T = float>
struct LeakyReluDag {
    using OpCopyInX = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpLeakRelu = Ops::Base::Bind<Ops::Base::Vec::LeakyRelu<U>, OpCopyInX, Ops::Base::Placeholder::Var<T, 0>>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpLeakRelu>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using OpDag = Ops::Base::DAGSch<Outputs>;
};

template <typename U, typename T = float>
struct LeakyReluCastDag {
    using OpCopyInX = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpCopyInXCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyInX>;
    using OpLeakRelu = Ops::Base::Bind<Ops::Base::Vec::LeakyRelu<T>, OpCopyInXCast, Ops::Base::Placeholder::Var<T, 0>>;
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyOutCast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_RINT>, OpLeakRelu>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpCopyOutCast>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using OpDag = Ops::Base::DAGSch<Outputs>;
};

#endif // OPS_NN_ACTIVATION_LEAKY_RELU_OP_KERNEL_ARCH35_LEAKY_RELU_DAG_H
