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
 * \file relu_v2_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_RELU_V2_DAG_H
#define CANN_CUSTOM_OPS_RELU_V2_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace ReluV2Op {
using namespace AscendC;
using namespace Ops::Base;

constexpr int COMPARE_MODE_GT = 1;

template <typename U, typename T = float>
struct ReluV2DAG {
    using ConstValue = MAKE_CONST(float, 0);

    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, 0>, OpCopyIn0>;
    using OpResult0 = Bind<Vec::Relu<T>, OpCopyIn0Cast>;
    using OpResultCast = Bind<Vec::Cast<U, T, 1>, OpResult0>;
    using OpCopyOut0 = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using OpResult1 = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, OpCopyIn0, ConstValue>;
    using OpCopyOut1 = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out1<uint1_t>, OpResult1>;

    using Outputs = Elems<OpCopyOut0, OpCopyOut1>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ReluV2Op
#endif // CANN_CUSTOM_OPS_RELU_V2_DAG_H