/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file smooth_l1_loss_grad_v2_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_SMOOTH_L1_LOSS_GRAD_V2_DAG_H
#define CANN_CUSTOM_OPS_SMOOTH_L1_LOSS_GRAD_V2_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace SmoothL1LossGradV2Op {
    using namespace AscendC;
    constexpr int COMPARE_MODE_LT = 0;
    constexpr int COMPARE_MODE_LE = 3;
    constexpr int COMPARE_MODE_GE = 4;
    constexpr int SELECT_SCALAR = 1;
    constexpr int SELECT_TENSOR = 2;
    constexpr int SCALAR_SIGMA_IDX = 0;
    constexpr int SCALAR_NEG_SIGMA_IDX = 1;
    constexpr int SCALAR_INVERT_SIGMA_IDX = 2;
    constexpr int SCALAR_NORM_IDX = 3;

    /*
    * norm = 1
    * x = predict - label
    * if x < -sigma:
    *    return -norm * dout
    * else if x > sigma:
    *    return norm * dout
    * else:
    *    return norm * x * dout / sigma 
    */
    template <typename T, typename PromteT = float>
    struct SmoothL1LossGradV2OpDag {
        // 通过Compute构造计算图
        using ConstValueZero = MAKE_CONST(PromteT, 0.0);
        using ConstValueOne = MAKE_CONST(PromteT, 1.0);
        using ConstValueNegOne = MAKE_CONST(PromteT, -1.0);
        using OpZeroTensor = Ops::Base::Bind<Ops::Base::Vec::Duplicate<PromteT>, ConstValueZero>;

        using OpPredict = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In0<T>>;
        using OpLabel = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;
        using OpDout = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In2<T>>;

        using OpPredictCast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpPredict>;
        using OpLabelCast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpLabel>;
        using OpDoutCast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpDout>;

        using OpX = Ops::Base::Bind<Ops::Base::Vec::Sub<PromteT>, OpPredictCast, OpLabelCast>;
        using OpXAbs = Ops::Base::Bind<Ops::Base::Vec::Abs<PromteT>, OpX>;
        using OpXInvertSigma = Ops::Base::Bind<Ops::Base::Vec::Muls<PromteT>, OpX, Ops::Base::Placeholder::Var<PromteT, SCALAR_INVERT_SIGMA_IDX>>;
        
        using OpCommpareNeg = Ops::Base::Bind<Ops::Base::Vec::Compare<uint8_t, PromteT, COMPARE_MODE_LE>, OpX, Ops::Base::Placeholder::Var<PromteT, SCALAR_NEG_SIGMA_IDX>>;
        using OpCommparePos = Ops::Base::Bind<Ops::Base::Vec::Compare<uint8_t, PromteT, COMPARE_MODE_GE>, OpX, Ops::Base::Placeholder::Var<PromteT, SCALAR_SIGMA_IDX>>;
        using OpCommpareX = Ops::Base::Bind<Ops::Base::Vec::Compare<uint8_t, PromteT, COMPARE_MODE_GE>, OpXAbs, Ops::Base::Placeholder::Var<PromteT, SCALAR_SIGMA_IDX>>;
        // x > sigma ? 1.0 : 0.0
        using OpSelectPos = Ops::Base::Bind<Ops::Base::Vec::Select<uint8_t, PromteT, SELECT_SCALAR>, OpCommparePos, ConstValueOne, OpZeroTensor>;
        // x < -sigma ? -1.0 : 0.0
        using OpSelectNeg = Ops::Base::Bind<Ops::Base::Vec::Select<uint8_t, PromteT, SELECT_SCALAR>, OpCommpareNeg, ConstValueNegOne, OpZeroTensor>;
        // |x| >= sigma ? 0.0 : (predict-latbel) / beta
        using OpSelectX = Ops::Base::Bind<Ops::Base::Vec::Select<uint8_t, PromteT, SELECT_SCALAR>, OpCommpareX, ConstValueZero, OpXInvertSigma>;

        using OpAddPosNeg = Ops::Base::Bind<Ops::Base::Vec::Add<PromteT>, OpSelectPos, OpSelectNeg>; 
        using OpAdd = Ops::Base::Bind<Ops::Base::Vec::Add<PromteT>, OpAddPosNeg, OpSelectX>;
        using OpRes = Ops::Base::Bind<Ops::Base::Vec::Mul<PromteT>, OpAdd, OpDoutCast>;
        using Mul0 = Ops::Base::Bind<Ops::Base::Vec::Muls<PromteT>, OpRes, Ops::Base::Placeholder::Var<PromteT, SCALAR_NORM_IDX>>;
        using OpResultCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, PromteT, 1>, Mul0>;

        using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpResultCast>;
        using Outputs = Ops::Base::Elems<OpCopyOut>; 
        using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
        using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
    };

    template <typename T, typename PromteT = float>
    struct SmoothL1LossGradV2OpScalarDag {
        // 通过Compute构造计算图
        using ConstValueZero = MAKE_CONST(PromteT, 0.0);
        using ConstValueOne = MAKE_CONST(PromteT, 1.0);
        using ConstValueNegOne = MAKE_CONST(PromteT, -1.0);
        using OpZeroTensor = Ops::Base::Bind<Ops::Base::Vec::Duplicate<PromteT>, ConstValueZero>;

        using OpPredict = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In0<T>>;
        using OpLabel = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;
        using OpDout = Ops::Base::Bind<Ops::Base::Vec::Duplicate<T>, Ops::Base::Placeholder::In2<T, Ops::Base::Placeholder::ScalarAttr<true>>>;

        using OpPredictCast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpPredict>;
        using OpLabelCast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpLabel>;
        using OpDoutCast = Ops::Base::Bind<Ops::Base::Vec::Cast<PromteT, T, 0>, OpDout>;

        using OpX = Ops::Base::Bind<Ops::Base::Vec::Sub<PromteT>, OpPredictCast, OpLabelCast>;
        using OpXAbs = Ops::Base::Bind<Ops::Base::Vec::Abs<PromteT>, OpX>;
        using OpXInvertSigma = Ops::Base::Bind<Ops::Base::Vec::Muls<PromteT>, OpX, Ops::Base::Placeholder::Var<PromteT, SCALAR_INVERT_SIGMA_IDX>>;
        
        using OpCommpareNeg = Ops::Base::Bind<Ops::Base::Vec::Compare<uint8_t, PromteT, COMPARE_MODE_LE>, OpX, Ops::Base::Placeholder::Var<PromteT, SCALAR_NEG_SIGMA_IDX>>;
        using OpCommparePos = Ops::Base::Bind<Ops::Base::Vec::Compare<uint8_t, PromteT, COMPARE_MODE_GE>, OpX, Ops::Base::Placeholder::Var<PromteT, SCALAR_SIGMA_IDX>>;
        using OpCommpareX = Ops::Base::Bind<Ops::Base::Vec::Compare<uint8_t, PromteT, COMPARE_MODE_GE>, OpXAbs, Ops::Base::Placeholder::Var<PromteT, SCALAR_SIGMA_IDX>>;
        // x > sigma ? 1.0 : 0.0
        using OpSelectPos = Ops::Base::Bind<Ops::Base::Vec::Select<uint8_t, PromteT, SELECT_SCALAR>, OpCommparePos, ConstValueOne, OpZeroTensor>;
        // x < -sigma ? -1.0 : 0.0
        using OpSelectNeg = Ops::Base::Bind<Ops::Base::Vec::Select<uint8_t, PromteT, SELECT_SCALAR>, OpCommpareNeg, ConstValueNegOne, OpZeroTensor>;
        // |x| >= sigma ? 0.0 : (predict-latbel) / beta
        using OpSelectX = Ops::Base::Bind<Ops::Base::Vec::Select<uint8_t, PromteT, SELECT_SCALAR>, OpCommpareX, ConstValueZero, OpXInvertSigma>;

        using OpAddPosNeg = Ops::Base::Bind<Ops::Base::Vec::Add<PromteT>, OpSelectPos, OpSelectNeg>; 
        using OpAdd = Ops::Base::Bind<Ops::Base::Vec::Add<PromteT>, OpAddPosNeg, OpSelectX>;
        using OpRes = Ops::Base::Bind<Ops::Base::Vec::Mul<PromteT>, OpAdd, OpDoutCast>;
        using Mul0 = Ops::Base::Bind<Ops::Base::Vec::Muls<PromteT>, OpRes, Ops::Base::Placeholder::Var<PromteT, SCALAR_NORM_IDX>>;
        using OpResultCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, PromteT, 1>, Mul0>;

        using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpResultCast>;
        using Outputs = Ops::Base::Elems<OpCopyOut>; 
        using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
        using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
    };

} // namespace SmoothL1LossGradV2

#endif  // CANN_CUSTOM_OPS_SMOOTH_L1_LOSS_GRAD_V2_DAG_H