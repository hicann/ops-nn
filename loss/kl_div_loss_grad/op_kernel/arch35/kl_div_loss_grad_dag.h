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
 * \file kl_div_loss_grad_dag.h
 * \brief kl_div_loss_grad_dag def
 */
#ifndef ASCENDC_KL_DIV_LOSS_GRAD_DAG_H
#define ASCENDC_KL_DIV_LOSS_GRAD_DAG_H
#include <cmath>
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#endif

namespace AscendC {
namespace KlDivLossGradVec {
using namespace Ops::Base;
using namespace Ops::Base::Vec;

#ifdef __CCE_AICORE__
using AscendC::MicroAPI::RegTensor;
constexpr static uint16_t VECTOR_LENGTH = platform::GetVRegSize();
#endif

template <typename T, typename U = float>
struct CalcInput : public ElemwiseBinaryOP<U, U, T> {
    __aicore__ inline CalcInput(LocalTensor<U> &gradOut, LocalTensor<U> &grad, T &inputScalar, int32_t count) {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = ops::CeilDiv<uint32_t>(totalLen, oneRepeat);

        __ubuf__ U *gradAddr = (__ubuf__ U *)grad.GetPhyAddr();
        __ubuf__ U *gradOutAddr = (__ubuf__ U *)gradOut.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regGrad;

            // gt = maximizeFactor * gt
            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regGrad, gradAddr, (int32_t)oneRepeat);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(gradOutAddr, regGrad, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};
}   // namespace KlDivLossGradVec
}   // namespace AscendC
namespace KlDivLossGrad {
using namespace AscendC;
using namespace Ops::Base;

template <typename U, typename T = float>
struct KDLGLogTargetTrue {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyTarget = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;

    using OpCopyGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyGrad>;
    using OpCopyTargetCast = Bind<Vec::Cast<T, U, 0>, OpCopyTarget>;

    // -e^target * Grad + 0*input
    using OpSubOne =  MAKE_CONST(T, -1);

    using OpTargetExp = Bind<Vec::Exp<T>, OpCopyTargetCast>;
    using OpTargetExpMulGrad = Bind<Vec::Mul<T>, OpTargetExp, OpCopyGradCast>;
    using OpSubTargetExpMulGrad = Bind<Vec::Muls<T>, OpSubOne, OpTargetExpMulGrad>;
    using OpCustom = Bind<KlDivLossGradVec::CalcInput<U, T>, OpSubTargetExpMulGrad, Placeholder::In1<U, Placeholder::ScalarAttr<true>>>;
    using Oprst = Bind<Vec::Muls<T>, OpCustom, Placeholder::Var<T,0>>;
    
    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, Oprst>;
    using OpCopyYOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyYOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct  KDLGLogTargetFalse {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
   // using OpCopyInput = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyTarget = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;

    using OpCopyGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyGrad>;
   // using OpCopyInputCast = Bind<Vec::Cast<T, U, 0>, OpCopyInput>;
    using OpCopyTargetCast = Bind<Vec::Cast<T, U, 0>, OpCopyTarget>;

    // -target * Grad + 0*input
    using OpSubOne =  MAKE_CONST(T, -1);
    using OpTargetMulGrad = Bind<Vec::Mul<T>, OpCopyTargetCast, OpCopyGradCast>;
    using OpSubTargetMulGrad = Bind<Vec::Muls<T>, OpSubOne, OpTargetMulGrad>;

    using OpCustom = Bind<KlDivLossGradVec::CalcInput<U, T>, OpSubTargetMulGrad, Placeholder::In1<U, Placeholder::ScalarAttr<true>>>;

    using Oprst = Bind<Vec::Muls<T>, OpCustom, Placeholder::Var<T,0>>;
    
    using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, Oprst>;
    using OpCopyYOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyYOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace KlDivLossGrad
#endif //ASCENDC_KL_DIV_LOSS_GRAD_DAG_H_