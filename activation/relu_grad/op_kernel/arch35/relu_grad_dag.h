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
 * \file relu_grad_dag.h
 * \brief if features > 0 return gradients, else return 0.
 */

#ifndef RELU_GRAD_DAG_H
#define RELU_GRAD_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
namespace ReluGradOp
{
const int COMPARE_MODE_LE = 3;
const int SEL_MODE = 2;

template <typename T>
struct ReluGradDag {
    // support float16, float32, int32, bfloat16, int64
    using const_zero = MAKE_CONST(float, 0.0);
    using const_one = MAKE_CONST(float, 1.0);
    using data_zero = Bind<Vec::Duplicate<T>, const_zero>;
    using data_one = Bind<Vec::Duplicate<T>, const_one>;
    using input_x1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using input_x2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using compare = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_LE>, input_x2, data_zero>;
    using select = Bind<Vec::Select<uint8_t, T, SEL_MODE>, compare, data_zero, data_one>;
    // follow tensorflow, res = gradients * (features > 0)
    using mul = Bind<Vec::Mul<T>, input_x1, select>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, mul>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct ReluGradCastDag {
    constexpr static int CAST_MODE_NONE = 0;
    constexpr static int CAST_MODE_RINT = 1;
    // support int8, uint8
    using const_zero = MAKE_CONST(float, 0.0);
    using const_one = MAKE_CONST(float, 1.0);
    using data_zero = Bind<Vec::Duplicate<T>, const_zero>;
    using data_one = Bind<Vec::Duplicate<T>, const_one>;
    using input_x1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using input_x2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using compare = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_LE>, input_x2, data_zero>;
    using select = Bind<Vec::Select<uint8_t, T, SEL_MODE>, compare, data_zero, data_one>;
    // follow tensorflow, res = gradients * (features > 0)
    using input_x1_cast = Bind<Vec::Cast<half, T, CAST_MODE_NONE>, input_x1>;
    using select_cast = Bind<Vec::Cast<half, T, CAST_MODE_NONE>, select>;
    using mul_cast = Bind<Vec::Mul<half>, input_x1_cast, select_cast>;
    // microapi mul not support uint8/int8, so need to cast
    using mul = Bind<Vec::Cast<T, half, CAST_MODE_RINT>, mul_cast>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, mul>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace ReluGradOp
#endif  // RELU_GRAD_DAG_H
