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
 * \file softplus_dag.h
 * \brief softplus implement
 */

#ifndef _ACTIVATION_KERNEL_SOFTPLUS_DAG_H_
#define _ACTIVATION_KERNEL_SOFTPLUS_DAG_H_

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace SoftplusOp {

using namespace Ops::Base;

template <typename U, typename T = float>
struct SoftplusDag {
    using ConstZero = MAKE_CONST(float, 0);
    using ConstNegOne = MAKE_CONST(float, -1);
    using ConstOne = MAKE_CONST(float, 1);
    using constNegLn2 = MAKE_CONST(float, -0.69314718055994530941723212145818);

    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, 0>, OpCopyIn0>;
    using xPositive = Bind<Vec::Maxs<T>, OpCopyIn0Cast, ConstZero>;
    using xPosToNeg = Bind<Vec::Muls<T>, xPositive, ConstNegOne>;
    using xPosExp = Bind<Vec::Exp<T>, xPosToNeg>;
    using xPosExpAddOne = Bind<Vec::Adds<T>, xPosExp, ConstOne>;
    using xPosLog = Bind<Vec::Log<T>, xPosExpAddOne>;
    using xPosRes = Bind<Vec::Add<T>, xPosLog, xPositive>;

    using xNegative = Bind<Vec::Mins<T>, OpCopyIn0Cast, ConstZero>;
    using xNegExp = Bind<Vec::Exp<T>, xNegative>;
    using xNegExpAddOne = Bind<Vec::Adds<T>, xNegExp, ConstOne>;
    using xNegRes = Bind<Vec::Log<T>, xNegExpAddOne>;

    using xRes = Bind<Vec::Add<T>, xPosRes, xNegRes>;
    using yRes = Bind<Vec::Adds<T>, xRes, constNegLn2>;

    using yResCast = Bind<Vec::Cast<U, T, 1>, yRes>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, yResCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

}

#endif  // _ACTIVATION_KERNEL_SOFTPLUS_DAG_H_
