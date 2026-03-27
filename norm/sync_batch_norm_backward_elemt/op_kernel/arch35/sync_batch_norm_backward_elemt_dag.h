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
 * \file sync_batch_norm_backward_elemt_dag.h
 * \brief
 */

#ifndef SYNC_BATCH_NORM_BACKWARD_ELEMT_DAG_H
#define SYNC_BATCH_NORM_BACKWARD_ELEMT_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

constexpr int CAST_NONE = 0;
constexpr int CAST_RINT = 1;

template <typename U, typename T>
struct SyncBatchNormBackwardElemtDag {
    using OpCopyInGradOut = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyInInput = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpCopyInMean = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using OpCopyInInvstd = Bind<Vec::CopyIn<T>, Placeholder::In3<T>>;
    using OpCopyInWeight = Bind<Vec::CopyIn<T>, Placeholder::In4<T>>;
    using OpCopyInMeanDy = Bind<Vec::CopyIn<T>, Placeholder::In5<T>>;
    using OpCopyInMeanDyXmu = Bind<Vec::CopyIn<T>, Placeholder::In6<T>>;
    
    using OpCopyInGradOutCast = Bind<Vec::Cast<float, U, CAST_NONE>, OpCopyInGradOut>;
    using OpCopyInInputCast = Bind<Vec::Cast<float, U, CAST_NONE>, OpCopyInInput>;
    using OpCopyInMeanCast = Bind<Vec::Cast<float, T, CAST_NONE>, OpCopyInMean>;
    using OpCopyInInvstdCast = Bind<Vec::Cast<float, T, CAST_NONE>, OpCopyInInvstd>;
    using OpCopyInWeightCast = Bind<Vec::Cast<float, T, CAST_NONE>, OpCopyInWeight>;
    using OpCopyInMeanDyCast = Bind<Vec::Cast<float, T, CAST_NONE>, OpCopyInMeanDy>;
    using OpCopyInMeanDyXmuCast = Bind<Vec::Cast<float, T, CAST_NONE>, OpCopyInMeanDyXmu>;

    using OpResultSub1 = Bind<Vec::Sub<float>, OpCopyInGradOutCast, OpCopyInMeanDyCast>;
    using OpResultSub2 = Bind<Vec::Sub<float>, OpCopyInInputCast, OpCopyInMeanCast>;
    using OpResultMul1 = Bind<Vec::Mul<float>, OpCopyInInvstdCast, OpCopyInInvstdCast>;
    using OpResultMul2 = Bind<Vec::Mul<float>, OpCopyInMeanDyXmuCast, OpResultMul1>;
    using OpResultMul3 = Bind<Vec::Mul<float>, OpResultSub2, OpResultMul2>;
    using OpResultSub3 = Bind<Vec::Sub<float>, OpResultSub1, OpResultMul3>;
    using OpResultMulWeight = Bind<Vec::Mul<float>, OpCopyInInvstdCast, OpCopyInWeightCast>;
    
    using OpResultMulGradInput = Bind<Vec::Mul<float>, OpResultSub3, OpResultMulWeight>;
    using OpResultMulGradInputCast = Bind<Vec::Cast<U, float, CAST_RINT>, OpResultMulGradInput>;
    using OpCopyOutGradInput = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultMulGradInputCast>;
    using Outputs = Elems<OpCopyOutGradInput>;

    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

#endif  // SYNC_BATCH_NORM_BACKWARD_ELEMT_DAG_H