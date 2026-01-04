/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file    adam_apply_one_dag.h
 * \brief adam_apply_one_dag head file
 */

#ifndef ADAM_APPLY_ONE_DAG_H
#define ADAM_APPLY_ONE_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace AdamApplyOneOp {
using namespace AscendC;
using namespace Ops::Base;
template <typename T, typename U>
struct AdamApplyOneCompute {
    using OpInput0 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInput1 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpInput2 = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using OpInput3 = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;
    using OpInput4 = Bind<Vec::CopyInBrc<T>, Placeholder::In4<T>>;
    using OpMul0X = Bind<Vec::CopyInBrc<T>, Placeholder::In5<T>>;
    using OpMul1X = Bind<Vec::CopyInBrc<T>, Placeholder::In6<T>>;
    using OpMul2X = Bind<Vec::CopyInBrc<T>, Placeholder::In7<T>>;
    using OpMul3X = Bind<Vec::CopyInBrc<T>, Placeholder::In8<T>>;
    using OpAdd2Y = Bind<Vec::CopyInBrc<T>, Placeholder::In9<T>>;

    using OpInput0Cast = Bind<Vec::Cast<U, T, 0>, OpInput0>;
    using OpInput1Cast = Bind<Vec::Cast<U, T, 0>, OpInput1>;
    using OpInput2Cast = Bind<Vec::Cast<U, T, 0>, OpInput2>;
    using OpInput3Cast = Bind<Vec::Cast<U, T, 0>, OpInput3>;
    using OpInput4Cast = Bind<Vec::Cast<U, T, 0>, OpInput4>;
    using OpMul0XCast = Bind<Vec::Cast<U, T, 0>, OpMul0X>;
    using OpMul1XCast = Bind<Vec::Cast<U, T, 0>, OpMul1X>;
    using OpMul2XCast = Bind<Vec::Cast<U, T, 0>, OpMul2X>;
    using OpMul3XCast = Bind<Vec::Cast<U, T, 0>, OpMul3X>;
    using OpAdd2YCast = Bind<Vec::Cast<U, T, 0>, OpAdd2Y>;

    using OpPower = Bind<Vec::Mul<U>, OpInput0Cast, OpInput0Cast>;
    using OpTmp1 = Bind<Vec::Mul<U>, OpPower, OpMul3XCast>;
    using OpTmp2 = Bind<Vec::Mul<U>, OpInput1Cast, OpMul2XCast>;
    using OpTmp3 = Bind<Vec::Mul<U>, OpInput2Cast, OpMul0XCast>;
    using OpTmp4 = Bind<Vec::Mul<U>, OpInput0Cast, OpMul1XCast>;

    using OpOutput0 = Bind<Vec::Add<U>, OpTmp1, OpTmp2>;
    using OpOutput1 = Bind<Vec::Add<U>, OpTmp3, OpTmp4>;
    using OpOutput0Cast = Bind<Vec::Cast<T, U, 1>, OpOutput0>;
    using OpOutput1Cast = Bind<Vec::Cast<T, U, 1>, OpOutput1>;

    using OpSqrt = Bind<Vec::Sqrt<U>, OpOutput0>;
    using OpTmpAdd = Bind<Vec::Add<U>, OpSqrt, OpAdd2YCast>;
    using OpTmpDiv = Bind<Vec::Div<U>, OpOutput1, OpTmpAdd>;
    using OpTmpMul = Bind<Vec::Mul<U>, OpTmpDiv, OpInput4Cast>;

    using OpOutput2 = Bind<Vec::Sub<U>, OpInput3Cast, OpTmpMul>;
    using OpOutput2Cast = Bind<Vec::Cast<T, U, 1>, OpOutput2>;

    using OpCopyOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpOutput0Cast>;
    using OpCopyOut1 = Bind<Vec::CopyOut<T>, Placeholder::Out1<T>, OpOutput1Cast>;
    using OpCopyOut2 = Bind<Vec::CopyOut<T>, Placeholder::Out2<T>, OpOutput2Cast>;

    using Outputs = Elems<OpCopyOut0, OpCopyOut1, OpCopyOut2>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace AdamApplyOneOp
#endif // ADAM_APPLY_ONE_DAG_H
