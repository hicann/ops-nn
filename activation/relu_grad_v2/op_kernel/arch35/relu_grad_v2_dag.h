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
 * \file relu_grad_v2_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_RELU_GRAD_V2_DAG_H_
#define CANN_CUSTOM_OPS_RELU_GRAD_V2_DAG_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
namespace ReluGradV2Ns {

const int VSEL_TENSOR_TENSOR_MODE = 2;

template <typename T>
struct ReluGradV2 {
    using ConstValue = MAKE_CONST(float, 0);
    using OpDup = Bind<Vec::Duplicate<T>, ConstValue>;

    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyInMask = Bind<Vec::CopyIn<uint8_t>, Placeholder::In1<uint1_t>>;

    using OpSelect = Bind<Vec::Select<uint8_t, T, VSEL_TENSOR_TENSOR_MODE>, OpCopyInMask, OpCopyIn0, OpDup>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSelect>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace ReluGradV2Ns
#endif // CANN_CUSTOM_OPS_RELU_GRAD_V2_DAG_H_
