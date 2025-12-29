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
 * \file  mse_loss_grad_dag.h
 * \brief mse_loss_grad_dag
 */

 #ifndef MSE_LOSS_GRAD_DAG_H
 #define MSE_LOSS_GRAD_DAG_H
 #include "atvoss/util/dag.h"
 #include "atvoss/util/vec.h"
 #include "atvoss/util/placeholder.h"
 
 namespace MseLossGradOp{
     using namespace AscendC;
     using namespace Ops::Base;
 
     template <typename T, typename U>
     struct MseLossGradDag {
         using OpInputPredict = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
         using OpInputLabel = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
         using OpInputDout = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
 
         using OpPredictCast = Bind<Vec::Cast<U, T, 0>, OpInputPredict>;
         using OpLabelCast = Bind<Vec::Cast<U, T, 0>, OpInputLabel>;
         using OpDoutCast = Bind<Vec::Cast<U, T, 0>, OpInputDout>;
         using OpSubRes = Bind<Vec::Sub<U>, OpPredictCast, OpLabelCast>;
         using OpNormGrad = Bind<Vec::Muls<U>, OpSubRes, Placeholder::Var<U, 0>>;
         using OpOutput = Bind<Vec::Mul<U>, OpNormGrad, OpDoutCast>;
         using OpOutputCast = Bind<Vec::Cast<T, U, 1>, OpOutput>;
 
         using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpOutputCast>;
         using Outputs = Elems<OpCopyOut>;
         using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
         using OpDag = DAGSch<Outputs, void, MemCfg>;
     };
 
     template <typename T, typename U>
     struct MseLossGradScalarDag {
         using OpInputPredict = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
         using OpInputLabel = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
         using OpInputDout = Bind<Vec::Duplicate<T>, Placeholder::In2<T, Placeholder::ScalarAttr<true>>>;
 
         using OpPredictCast = Bind<Vec::Cast<U, T, 0>, OpInputPredict>;
         using OpLabelCast = Bind<Vec::Cast<U, T, 0>, OpInputLabel>;
         using OpDoutCast = Bind<Vec::Cast<U, T, 0>, OpInputDout>;
 
         using OpSubRes = Bind<Vec::Sub<U>, OpPredictCast, OpLabelCast>;
         using OpNormGrad = Bind<Vec::Muls<U>, OpSubRes, Placeholder::Var<U, 0>>;
         using OpOutput = Bind<Vec::Mul<U>, OpNormGrad, OpDoutCast>;
         using OpOutputCast = Bind<Vec::Cast<T, U, 1>, OpOutput>;
         using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpOutputCast>;
 
         using Outputs = Elems<OpCopyOut>;
         using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
         using OpDag = DAGSch<Outputs, void, MemCfg>;
     };
 }
 #endif // MSE_LOSS_GRAD_DAG_H