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
 * \file threshold_grad_v2_d_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_THRESHOLD_GRAD_V2_D_DAG_H
#define CANN_CUSTOM_OPS_THRESHOLD_GRAD_V2_D_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace ThresholdGradV2DOp {
using namespace Ops::Base;

constexpr int COMPARE_MODE_GT = 1;
constexpr int SELECT_MODE_TENSOR = 2;

template <typename U>
struct ThresholdGradV2DDag {
    using const_zero = MAKE_CONST(float, 0.0);
    using data_threshold = Bind<Vec::Duplicate<float>, Placeholder::Var<float, 0>>;
    using data_zero = Bind<Vec::Duplicate<float>, const_zero>;
    using OpCopyInGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyInGradCast = Bind<Vec::Cast<float, U, 0>, OpCopyInGrad>;
    using OpCopyInSelf = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyInSelfCast = Bind<Vec::Cast<float, U, 0>, OpCopyInSelf>;
    using Compare = Bind<Vec::Compare<uint8_t, float, COMPARE_MODE_GT>, OpCopyInSelfCast, data_threshold>;
    using Select = Bind<Vec::Select<uint8_t, float, SELECT_MODE_TENSOR>, Compare, OpCopyInGradCast, data_zero>;
    using SelectCast = Bind<Vec::Cast<U, float, 1>, Select>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, SelectCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ThresholdGradV2DOp
#endif // CANN_CUSTOM_OPS_THRESHOLD_GRAD_V2_D_DAG_H
