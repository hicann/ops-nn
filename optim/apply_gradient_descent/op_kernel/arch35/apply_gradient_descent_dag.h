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
 * \file apply_gradient_descent_dag.h
 * \brief apply_gradient_descent_dag
 */

 #ifndef CANN_CUSTOM_OPS_APPLY_GRADIENT_DEESCENT_DAG_H
 #define CANN_CUSTOM_OPS_APPLY_GRADIENT_DEESCENT_DAG_H
 #include "atvoss/util/dag.h"
 #include "atvoss/util/vec.h"
 #include "atvoss/util/placeholder.h"

 namespace ApplyGradientDescentOp{
   using namespace AscendC;
   using namespace Ops::Base;
   template <typename T, typename U = float>
   struct ApplyGradientDescentDAG {
      using OpVar = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
      using OpDelta = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
      using OpAlpha = Bind<Vec::Duplicate<T>, Placeholder::In1<T, Placeholder::ScalarAttr<true>>>;
      using OpVarCast = Bind<Vec::Cast<U, T, 0>, OpVar>;
      using OpAlphaCast = Bind<Vec::Cast<U, T, 0>, OpAlpha>;
      using OpDeltaCast = Bind<Vec::Cast<U, T, 0>, OpDelta>;

      using OpMul = Bind<Vec::Mul<U>, OpDeltaCast, OpAlphaCast>;
      using OpOutput = Bind<Vec::Sub<U>, OpVarCast, OpMul>;
      using OpOutputCast = Bind<Vec::Cast<T, U, 1>, OpOutput>;

      using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpOutputCast>;

      using Outputs = Elems<OpCopyOut>;
      using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
      using OpDag = DAGSch<Outputs, void, MemCfg>;
   };
 }
 #endif  // CANN_CUSTOM_OPS_APPLY_GRADIENT_DEESCENT_DAG_H