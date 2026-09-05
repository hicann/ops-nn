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

constexpr int COMPARE_MODE_LE = 3;
constexpr int SELECT_MODE_TENSOR = 2;

template <typename U>
struct ThresholdGradV2D8BDag {
    using const_zero = MAKE_CONST(float, 0.0);
    using data_threshold = Bind<Vec::Duplicate<float>, Placeholder::Var<float, 0>>;
    using data_zero = Bind<Vec::Duplicate<float>, const_zero>;
    using OpCopyInGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyInGradHalf = Bind<Vec::Cast<half, U, 0>, OpCopyInGrad>;
    using OpCopyInGradCast = Bind<Vec::Cast<float, half, 0>, OpCopyInGradHalf>;
    using OpCopyInSelf = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyInSelfHalf = Bind<Vec::Cast<half, U, 0>, OpCopyInSelf>;
    using OpCopyInSelfCast = Bind<Vec::Cast<float, half, 0>, OpCopyInSelfHalf>;
    using Compare = Bind<Vec::Compare<uint8_t, float, COMPARE_MODE_LE>, OpCopyInSelfCast, data_threshold>;
    using Select = Bind<Vec::Select<uint8_t, float, SELECT_MODE_TENSOR>, Compare, data_zero, OpCopyInGradCast>;
    using SelectHalf = Bind<Vec::Cast<half, float, 1>, Select>;
    using SelectCast = Bind<Vec::Cast<U, half, 1>, SelectHalf>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, SelectCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

struct ThresholdGradV2DInt32Dag {
    using const_zero = MAKE_CONST(int32_t, 0);
    using data_threshold = Bind<Vec::Duplicate<int32_t>, Placeholder::Var<int32_t, 0>>;
    using data_zero = Bind<Vec::Duplicate<int32_t>, const_zero>;
    using OpCopyInGrad = Bind<Vec::CopyInBrc<int32_t>, Placeholder::In0<int32_t>>;
    using OpCopyInSelf = Bind<Vec::CopyInBrc<int32_t>, Placeholder::In1<int32_t>>;
    using Compare = Bind<Vec::Compare<uint8_t, int32_t, COMPARE_MODE_LE>, OpCopyInSelf, data_threshold>;
    using Select = Bind<Vec::Select<uint8_t, int32_t, SELECT_MODE_TENSOR>, Compare, data_zero, OpCopyInGrad>;
    using OpCopyOut = Bind<Vec::CopyOut<int32_t>, Placeholder::Out0<int32_t>, Select>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct ThresholdGradV2DDag {
    using const_zero = MAKE_CONST(float, 0.0);
    using data_threshold = Bind<Vec::Duplicate<float>, Placeholder::Var<float, 0>>;
    using data_zero = Bind<Vec::Duplicate<float>, const_zero>;
    using OpCopyInGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyInGradCast = Bind<Vec::Cast<float, U, 0>, OpCopyInGrad>;
    using OpCopyInSelf = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyInSelfCast = Bind<Vec::Cast<float, U, 0>, OpCopyInSelf>;
    using Compare = Bind<Vec::Compare<uint8_t, float, COMPARE_MODE_LE>, OpCopyInSelfCast, data_threshold>;
    using Select = Bind<Vec::Select<uint8_t, float, SELECT_MODE_TENSOR>, Compare, data_zero, OpCopyInGradCast>;
    using SelectCast = Bind<Vec::Cast<U, float, 1>, Select>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, SelectCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ThresholdGradV2DOp
#endif // CANN_CUSTOM_OPS_THRESHOLD_GRAD_V2_D_DAG_H
