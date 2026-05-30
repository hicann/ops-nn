/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file relu6_grad_dag.h
 * \brief relu6_grad dag
 */

#ifndef RELU6_GRAD_DAG_H
#define RELU6_GRAD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace Relu6GradOp {
using namespace AscendC;
using namespace Ops::Base;
constexpr int32_t CAST_MODE_NONE = 0;
constexpr int32_t CAST_MODE_RINT = 1;
// CMPMODE values, see asc-devkit/impl/basic_api/utils/kernel_utils_mode.h.
// LT = 0, GT = 1, EQ = 2, LE = 3, GE = 4, NE = 5.
constexpr int32_t CMP_MODE_LT = 0;
constexpr int32_t CMP_MODE_GT = 1;
// SELMODE::VSEL_TENSOR_TENSOR_MODE = 2, semantics: dst = mask ? src0 : src1.
constexpr int32_t SEL_MODE_TT = 2;

// Native path for dtype that natively supports Compare/Select without
// precision concerns (only float on this platform; half/bf16 take the
// promoted path below).
//   dx = (0 < x < 6) ? dy : 0
template <typename T>
struct Relu6Grad {
    using InputDy = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using ConstZero = MAKE_CONST(T, 0);
    using ConstSix = MAKE_CONST(T, 6);
    using ZeroF = Bind<Vec::Duplicate<T>, ConstZero>;
    using SixF = Bind<Vec::Duplicate<T>, ConstSix>;

    using MaskGtZero = Bind<Vec::Compare<uint8_t, T, CMP_MODE_GT>, InputX, ZeroF>;
    using MaskLtSix = Bind<Vec::Compare<uint8_t, T, CMP_MODE_LT>, InputX, SixF>;
    using MaskIn = Bind<Vec::And<uint8_t>, MaskGtZero, MaskLtSix>;

    using SelectRes = Bind<Vec::Select<uint8_t, T, SEL_MODE_TT>, MaskIn, InputDy, ZeroF>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// Promoted path for half / bfloat16_t: cast inputs to PromoteT (float) for
// cmp/and/sel to (a) match the DSL fallback in relu6_grad.py that lifts to
// fp32 when fp16 vcmpsel is unavailable, (b) keep boundary comparison
// against 0 and 6 stable across half/bf16. Result is cast back to T with
// round-to-nearest-even.
template <typename T, typename PromoteT>
struct Relu6GradFloatCast {
    using InputDy = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputDyCast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputDy>;
    using InputXCast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputX>;

    using ConstZero = MAKE_CONST(PromoteT, 0);
    using ConstSix = MAKE_CONST(PromoteT, 6);
    using ZeroF = Bind<Vec::Duplicate<PromoteT>, ConstZero>;
    using SixF = Bind<Vec::Duplicate<PromoteT>, ConstSix>;

    using MaskGtZero = Bind<Vec::Compare<uint8_t, PromoteT, CMP_MODE_GT>, InputXCast, ZeroF>;
    using MaskLtSix = Bind<Vec::Compare<uint8_t, PromoteT, CMP_MODE_LT>, InputXCast, SixF>;
    using MaskIn = Bind<Vec::And<uint8_t>, MaskGtZero, MaskLtSix>;

    using SelectRes = Bind<Vec::Select<uint8_t, PromoteT, SEL_MODE_TT>, MaskIn, InputDyCast, ZeroF>;
    using SelectResCast = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, SelectRes>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectResCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace Relu6GradOp
#endif // RELU6_GRAD_DAG_H
