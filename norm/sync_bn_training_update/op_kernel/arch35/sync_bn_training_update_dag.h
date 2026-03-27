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
 * \file sync_bn_training_update_dag.h
 * \brief
 */

#ifndef SYNC_BN_TRAINING_UPDATE_DAG_H
#define SYNC_BN_TRAINING_UPDATE_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

template <typename T>
struct SyncBNTrainingUpdateDag {
    using inputMean = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using inputRunningMean = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;

    using castMean = Bind<Vec::Cast<float, T, 0>, inputMean>;
    using castRunningMean = Bind<Vec::Cast<float, T, 0>, inputRunningMean>;
    using mulPart1 = Bind<Vec::Muls<float>, castMean, Placeholder::Var<float, 0>>;
    using mulPart2 = Bind<Vec::Muls<float>, castRunningMean, Placeholder::Var<float, 0>>;
    using mulPart2Sub = Bind<Vec::Sub<float>, castRunningMean, mulPart2>;
    using res = Bind<Vec::Add<float>, mulPart1, mulPart2Sub>;
    using resCast = Bind<Vec::Cast<T, float, 1>, res>;
    using opCopyOut= Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, resCast>;
    using Output = Elems<opCopyOut>;

    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Output, void, MemCfg>;
};

#endif // SYNC_BN_TRAINING_UPDATE_DAG_H
