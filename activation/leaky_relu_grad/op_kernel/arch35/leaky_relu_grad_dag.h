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
 * \file leaky_relu_grad_dag.h
 * \brief if features > 0 return gradients, else return *gradients.
 */

 #ifndef LEAKY_RELU_GRAD_DAG_H
 #define LEAKY_RELU_GRAD_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

 using namespace AscendC;
 using namespace Ops::Base;
 namespace LeakyReluGradOp
 {
 const int COMPARE_MODE_GT = 1;
 const int SELECT_MODE_TENSOR = 2;
 constexpr int CastModeBf16ToFp32 = 0;
 constexpr int CastModeFp32ToBf16 = 1;
 
 template <typename U>
 struct LeakyReluGradDag {
     using const_zero = MAKE_CONST(U, 0.0);
     using data_zero = Bind<Vec::Duplicate<U>, const_zero>;
     using OpCopyInY = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
     using OpCopyInYCast = Bind<Vec::Cast<float, U, CastModeBf16ToFp32>, OpCopyInY>;
     using OpCopyInYCastMul = Bind<Vec::Muls<float>, OpCopyInYCast, Placeholder::Var<float, 0>>;
     using OpCopyInX = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
     using Compare = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>,OpCopyInX,data_zero>;
     using Select = Bind<Vec::Select<uint8_t, float, SELECT_MODE_TENSOR>, Compare, OpCopyInYCast, OpCopyInYCastMul>;
     using SelectCast = Bind<Vec::Cast<U, float, CastModeFp32ToBf16>, Select>;
     using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, SelectCast>;
     // 指定输出节点
     using Outputs = Elems<OpCopyOut>;
     using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
     using OpDag = DAGSch<Outputs, void, MemCfg>;
 };
 }  // namespace ReluGradOp
 #endif  // LEAKY_RELU_GRAD_DAG_H