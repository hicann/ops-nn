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
 * \file kl_div_target_loss_grad_dag.h
 * \brief kl_div_target_loss_grad_dag def
 */
#ifndef ASCENDC_KL_DIV_TARGET_LOSS_GRAD_DAG_H
#define ASCENDC_KL_DIV_TARGET_LOSS_GRAD_DAG_H
#include <cmath>
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#endif

namespace KlDivTargetLossGrad {
using namespace AscendC;
using namespace Ops::Base;

template <typename U, typename T = float>
struct KDTLGLogTargetTrue {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;  // gradOutput
    using OpCopySelf = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;   // self
    using OpCopyTarget = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>; // target

    // 转换为内部计算类型T
    using OpCopyGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyGrad>;
    using OpCopySelfCast = Bind<Vec::Cast<T, U, 0>, OpCopySelf>;
    using OpCopyTargetCast = Bind<Vec::Cast<T, U, 0>, OpCopyTarget>;

    // 提前应用reduction缩放因子（防溢出）
    using OpGradWithReduction = Bind<Vec::Muls<T>, OpCopyGradCast, Placeholder::Var<T,0>>;  // gradOutput * reduction_factor

    // log_target=True时: gradTarget = gradOutput * reduction_factor * (target - self) * exp(target) + gradOutput * reduction_factor * exp(target)
    using OpTargetExp = Bind<Vec::Exp<T>, OpCopyTargetCast>;           // exp(target)
    using OpTargetSubSelf = Bind<Vec::Sub<T>, OpCopyTargetCast, OpCopySelfCast>;  // target - self
    using OpGradMulDiff = Bind<Vec::Mul<T>, OpGradWithReduction, OpTargetSubSelf>;    //  gradOutput * reduction_factor * (target - self)
    using OpGradMulDiffMulExp = Bind<Vec::Mul<T>, OpGradMulDiff, OpTargetExp>;  // gradOutput * reduction_factor * (target - self) * exp(target)
    using OpGradMulExp = Bind<Vec::Mul<T>, OpGradWithReduction, OpTargetExp>;     // gradOutput * reduction_factor * exp(target)
    using OpCustom = Bind<Vec::Add<T>, OpGradMulDiffMulExp, OpGradMulExp>;    //  gradOutput * reduction_factor * (target - self) * exp(target) + gradOutput * reduction_factor * exp(target)

    // 转换回输出类型
    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpCustom>;
    
    using OpCopyYOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    
    using Outputs = Elems<OpCopyYOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct  KDTLGLogTargetFalse {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;  // gradOutput
    using OpCopySelf = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;   // self
    using OpCopyTarget = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>; // target

    // 转换为内部计算类型T
    using OpCopyGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyGrad>;
    using OpCopySelfCast = Bind<Vec::Cast<T, U, 0>, OpCopySelf>;
    using OpCopyTargetCast = Bind<Vec::Cast<T, U, 0>, OpCopyTarget>;

    // 提前应用reduction缩放因子（防溢出）
    using OpGradWithReduction = Bind<Vec::Muls<T>, OpCopyGradCast, Placeholder::Var<T,0>>;  // gradOutput * reduction_factor

    // log_target=False时: gradTarget = gradOutput * log(target) - gradOutput * self + gradOutput * target / target
    using OpGradMulSelf = Bind<Vec::Mul<T>, OpGradWithReduction, OpCopySelfCast>;   // gradOutput * self
    using OpTargetLog = Bind<Vec::Log<T>, OpCopyTargetCast>;           // log(target)
    using OpXLogy = Bind<Vec::Mul<T>, OpGradWithReduction, OpTargetLog>;   // gradOutput * log(target) 
    using OpLogSubGrad = Bind<Vec::Sub<T>, OpXLogy, OpGradMulSelf>;  // gradOutput * log(target) - gradOutput * self
    using OpGradMulTarget = Bind<Vec::Mul<T>, OpGradWithReduction, OpCopyTargetCast>;   // gradOutput * target
    using OpGradMulTargetDivTarget = Bind<Vec::Div<T>, OpGradMulTarget, OpCopyTargetCast>;   // gradOutput * target / target
    using OpCustom = Bind<Vec::Add<T>, OpLogSubGrad, OpGradMulTargetDivTarget>;  // gradOutput * log(target) - gradOutput * self + gradOutput * target / target

    // 转换回输出类型
    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpCustom>;

    using OpCopyYOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    
    using Outputs = Elems<OpCopyYOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace KlDivTargetLossGrad
#endif //ASCENDC_KL_DIV_TARGET_LOSS_GRAD_DAG_H