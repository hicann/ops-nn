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
 * \file binary_cross_entropy_grad_dag.h
 * \brief
 */
#ifndef ASCENDC_BINARY_CORSS_ENTROPY_GRAD_DAG_H_
#define ASCENDC_BINARY_CORSS_ENTROPY_GRAD_DAG_H_
#include <cmath>
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#endif

namespace BinaryCrossEntropyGrad {
using namespace Ops::Base;
using namespace AscendC;

template <typename U, typename T = float>
struct BCEGMeanHasWeight {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyGradOutput = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyWeight = Bind<Vec::CopyInBrc<U>, Placeholder::In3<U>>;
    using OpCopyXCast = Bind<Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Bind<Vec::Cast<T, U, 0>, OpCopyY>;
    using OpCopyGradOutputCast = Bind<Vec::Cast<T, U, 0>, OpCopyGradOutput>;
    using OpCopyWeightCast = Bind<Vec::Cast<T, U, 0>, OpCopyWeight>;
    // grad_output * (1/n) * weight * (x - y) / ((1 - x) * x)
    using OpOne = MAKE_CONST(T, 1);
    using OpXSubY = Bind<Vec::Sub<T>, OpCopyXCast, OpCopyYCast>;
    using OpOneSubX = Bind<Vec::Subs<T>, OpOne, OpCopyXCast>;
    using OpOneSubXMulX = Bind<Vec::Mul<T>, OpOneSubX, OpCopyXCast>;
    using OpOneE = MAKE_CONST(T, 1e-12);
    using OpOneSubXMulXMax = Bind<Vec::Maxs<T>, OpOneE, OpOneSubXMulX>;
    using OpResGrad = Bind<Vec::Mul<T>, OpCopyGradOutputCast, OpXSubY>;
    using OpXSubYDivOneSubXMulX = Bind<Vec::Div<T>, OpResGrad, OpOneSubXMulXMax>;

    using OpResWeightCast = Bind<Vec::Mul<T>, OpXSubYDivOneSubXMulX, OpCopyWeightCast>;
    using OpResCast = Bind<Vec::Muls<T>, OpResWeightCast, Placeholder::Var<T, 0>>;
    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResCast>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEGMeanHasNoWeight {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyGradOutput = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyXCast = Bind<Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Bind<Vec::Cast<T, U, 0>, OpCopyY>;
    using OpCopyGradOutputCast = Bind<Vec::Cast<T, U, 0>, OpCopyGradOutput>;
    // grad_output * (1/n) * (x - y) / ((1 - x) * x)
    using OpOne = MAKE_CONST(T, 1);
    using OpXSubY = Bind<Vec::Sub<T>, OpCopyXCast, OpCopyYCast>;
    using OpOneSubX = Bind<Vec::Subs<T>, OpOne, OpCopyXCast>;
    using OpOneSubXMulX = Bind<Vec::Mul<T>, OpOneSubX, OpCopyXCast>;
    using OpOneE = MAKE_CONST(T, 1e-12);
    using OpOneSubXMulXMax = Bind<Vec::Maxs<T>, OpOneE, OpOneSubXMulX>;
    using OpResGrad = Bind<Vec::Mul<T>, OpCopyGradOutputCast, OpXSubY>;
    using OpXSubYDivOneSubXMulX = Bind<Vec::Div<T>, OpResGrad, OpOneSubXMulXMax>;

    using OpResCast = Bind<Vec::Muls<T>, OpXSubYDivOneSubXMulX, Placeholder::Var<T, 0>>;
    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResCast>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEGSumHasWeight {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyGradOutput = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyWeight = Bind<Vec::CopyInBrc<U>, Placeholder::In3<U>>;
    using OpCopyXCast = Bind<Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Bind<Vec::Cast<T, U, 0>, OpCopyY>;
    using OpCopyWeightCast = Bind<Vec::Cast<T, U, 0>, OpCopyWeight>;
    using OpCopyGradOutputCast = Bind<Vec::Cast<T, U, 0>, OpCopyGradOutput>;
    // grad_output * weight * (x - y) / ((1 - x) * x)
    using OpOne = MAKE_CONST(T, 1);
    using OpXSubY = Bind<Vec::Sub<T>, OpCopyXCast, OpCopyYCast>;
    using OpOneSubX = Bind<Vec::Subs<T>, OpOne, OpCopyXCast>;
    using OpOneSubXMulX = Bind<Vec::Mul<T>, OpOneSubX, OpCopyXCast>;
    using OpOneE = MAKE_CONST(T, 1e-12);
    using OpOneSubXMulXMax = Bind<Vec::Maxs<T>, OpOneE, OpOneSubXMulX>;
    using OpResGrad = Bind<Vec::Mul<T>, OpCopyGradOutputCast, OpXSubY>;
    using OpXSubYDivOneSubXMulX = Bind<Vec::Div<T>, OpResGrad, OpOneSubXMulXMax>;

    using OpResWeightCast = Bind<Vec::Mul<T>, OpXSubYDivOneSubXMulX, OpCopyWeightCast>;
    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResWeightCast>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEGSumHasNoWeight {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyGradOutput = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyGradOutputCast = Bind<Vec::Cast<T, U, 0>, OpCopyGradOutput>;
    using OpCopyXCast = Bind<Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Bind<Vec::Cast<T, U, 0>, OpCopyY>;
    // grad_output * (x - y) / ((1 - x) * x)
    using OpOne = MAKE_CONST(T, 1);
    using OpXSubY = Bind<Vec::Sub<T>, OpCopyXCast, OpCopyYCast>;
    using OpOneSubX = Bind<Vec::Subs<T>, OpOne, OpCopyXCast>;
    using OpOneSubXMulX = Bind<Vec::Mul<T>, OpOneSubX, OpCopyXCast>;
    using OpOneE = MAKE_CONST(T, 1e-12);
    using OpOneSubXMulXMax = Bind<Vec::Maxs<T>, OpOneE, OpOneSubXMulX>;
    using OpResGrad = Bind<Vec::Mul<T>, OpCopyGradOutputCast, OpXSubY>;
    using OpXSubYDivOneSubXMulX = Bind<Vec::Div<T>, OpResGrad, OpOneSubXMulXMax>;

    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpXSubYDivOneSubXMulX>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace BinaryCrossEntropyGrad
#endif //ASCENDC_BINARY_CORSS_ENTROPY_GRAD_DAG_H_