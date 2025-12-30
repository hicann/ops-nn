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
 * \file hardtanh_grad_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_HARDTANH_GRAD_DAG_H
#define CANN_CUSTOM_OPS_HARDTANH_GRAD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace HardtanhGradOp {
using namespace Ops::Base;

template <typename U, typename T = float>
struct HardtanhGradDag {
    // calculate: torch.where((result <= min_val) | (result >= max_val), 0.0, grad)
    using ConstValueZero = MAKE_CONST(T, 0.0);
    constexpr static std::int32_t COMPARE_MODE_LT = 0;
    constexpr static std::int32_t COMPARE_MODE_GT = 1;
    constexpr static std::int32_t COMPARE_MODE_NE = 5;
    constexpr static std::int32_t CAST_MODE_NONE = 0;
    constexpr static std::int32_t SELECT_MODE_VS = 1;

    // copy in & cast to float
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;

    // compare
    using OpCompareNaN = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_NE>, OpCopyIn0Cast, OpCopyIn0Cast>;
    using OpCompareLowerThanMax = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_LT>, OpCopyIn0Cast, Placeholder::Var<T, 1>>;
    using OpCompareGreaterThanMin = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_GT>, OpCopyIn0Cast, Placeholder::Var<T, 0>>;

    // mask
    using OpMaskAnd = Bind<Vec::And<uint8_t>, OpCompareLowerThanMax, OpCompareGreaterThanMin>;
    using OpMaskOr = Bind<Vec::Or<uint8_t>, OpMaskAnd, OpCompareNaN>;

    // select
    using OpSelect = Bind<Vec::Select<uint8_t, U, SELECT_MODE_VS>, OpMaskOr, OpCopyIn1, ConstValueZero>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpSelect>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace HardtanhGradOp

#endif // CANN_CUSTOM_OPS_HARD_TANHGRAD_DAG_H