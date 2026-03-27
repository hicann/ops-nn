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
 * \file sync_batch_norm_backward_reduce_dag.h
 * \brief
 */

#ifndef SYNC_BATCH_NORM_BACKWARD_REDUCE_DAG_H
#define SYNC_BATCH_NORM_BACKWARD_REDUCE_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

template <typename U>
struct SyncBatchNormBackwardReduceDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpCopyIn2 = Bind<Vec::CopyIn<U>, Placeholder::In2<U>>;
    using OpCopyIn3 = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;

    using OpCopyIn0Cast = Bind<Vec::Cast<float, U, 0>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<float, U, 0>, OpCopyIn1>;
    using OpCopyIn2Cast = Bind<Vec::Cast<float, U, 0>, OpCopyIn2>;
    using OOpCopyIn3Cast = Bind<Vec::Cast<float, U, 0>, OpCopyIn3>;

    using OpTMul = Bind<Vec::Mul<float>, OpCopyIn2Cast, OpCopyIn0Cast>;
    using OpResult0 = Bind<Vec::Sub<float>, OpCopyIn1Cast, OpTMul>;
    using OpResult1 = Bind<Vec::Mul<float>, OpResult0, OOpCopyIn3Cast>;

    using OpResult0Cast = Bind<Vec::Cast<U, float, 1>, OpResult0>;
    using OpResult1Cast = Bind<Vec::Cast<U, float, 1>, OpResult1>;

    using OpCopyOut0 = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResult0Cast>;
    using OpCopyOut1 = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpResult1Cast>;

    using Outputs = Elems<OpCopyOut0, OpCopyOut1>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
#endif