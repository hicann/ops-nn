/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file apply_adam_dag.h
 * \brief apply_adam_dag head file
 */

#ifndef CANN_CUSTOM_OPS_APPLY_ADAM_D_DAG_H
#define CANN_CUSTOM_OPS_APPLY_ADAM_D_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#endif

namespace AscendC {
namespace Vec {
#ifdef __CCE_AICORE__
using Reg::RegTensor;

constexpr static uint16_t VECTOR_LENGTH = Ops::Base::GetVRegSize();

template <typename U = float>
__aicore__ inline void CalcLr(Reg::RegTensor<U>& regLrT, Reg::MaskReg& pregUp, U beta1PowerUp, U beta2PowerUp, U lrUp)
{
    Reg::RegTensor<U> regBeta1Power;
    Reg::RegTensor<U> regBeta2Power;
    Reg::RegTensor<U> regNegBeta2Power;
    Reg::RegTensor<U> regAddBeta2Power;
    Reg::RegTensor<U> regSqrtBeta2Power;
    Reg::RegTensor<U> regNegBeta1Power;
    Reg::RegTensor<U> regAddBeta1Power;
    Reg::RegTensor<U> regMulLrSqrt;

    Reg::Duplicate(regBeta1Power, beta1PowerUp, pregUp);
    Reg::Duplicate(regBeta2Power, beta2PowerUp, pregUp);
    Reg::Muls(regNegBeta2Power, regBeta2Power, -1.0f, pregUp);
    Reg::Adds(regAddBeta2Power, regNegBeta2Power, 1.0f, pregUp);
    Reg::Sqrt(regSqrtBeta2Power, regAddBeta2Power, pregUp);
    Reg::Muls(regNegBeta1Power, regBeta1Power, -1.0f, pregUp);
    Reg::Adds(regAddBeta1Power, regNegBeta1Power, 1.0f, pregUp);
    Reg::Muls(regMulLrSqrt, regSqrtBeta2Power, lrUp, pregUp);
    Reg::Div(regLrT, regMulLrSqrt, regAddBeta1Power, pregUp);
}

template <typename U = float>
__aicore__ inline void CalcVarTWithLr(Reg::RegTensor<U>& regVarT, Reg::RegTensor<U>& regVar, Reg::RegTensor<U>& regLrT,
                                      Reg::RegTensor<U>& regMt, Reg::RegTensor<U>& regVt, Reg::MaskReg& pregUp,
                                      U epsilonUp)
{
    Reg::RegTensor<U> regMulLeft;
    Reg::RegTensor<U> regSqrtVt;
    Reg::RegTensor<U> regAddSqrtV;
    Reg::RegTensor<U> regDivRes;

    Reg::Mul(regMulLeft, regLrT, regMt, pregUp);
    Reg::Sqrt(regSqrtVt, regVt, pregUp);
    Reg::Adds(regAddSqrtV, regSqrtVt, epsilonUp, pregUp);
    Reg::Div(regDivRes, regMulLeft, regAddSqrtV, pregUp);
    Reg::Sub(regVarT, regVar, regDivRes, pregUp);
}

template <typename U = float>
__aicore__ inline void CalcMtLookAhead(Reg::RegTensor<U>& regMtAhead, Reg::RegTensor<U>& regMt,
                                       Reg::RegTensor<U>& regGrad, Reg::MaskReg& pregUp, U beta1Up)
{
    Reg::RegTensor<U> regBeta1;
    Reg::RegTensor<U> regMulMtBeta1;
    Reg::RegTensor<U> regNegBeta1;
    Reg::RegTensor<U> regSub1Beta1;
    Reg::RegTensor<U> regMulGrad;

    Reg::Duplicate(regBeta1, beta1Up, pregUp);
    Reg::Mul(regMulMtBeta1, regBeta1, regMt, pregUp);
    Reg::Muls(regNegBeta1, regBeta1, -1.0f, pregUp);
    Reg::Adds(regSub1Beta1, regNegBeta1, 1.0f, pregUp);
    Reg::Mul(regMulGrad, regSub1Beta1, regGrad, pregUp);
    Reg::Add(regMtAhead, regMulMtBeta1, regMulGrad, pregUp);
}

#endif

template <typename T, typename U = float>
struct CalcMt : public Ops::Base::Vec::ElemwiseTernaryOP<U, U, U, float> {
    __aicore__ inline CalcMt(Ops::Base::LocalTensor<U>& mT, Ops::Base::LocalTensor<U>& m,
                             Ops::Base::LocalTensor<U>& grad, float& beta1, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);

        __ubuf__ U* mAddr = (__ubuf__ U*)m.GetPhyAddr();
        __ubuf__ U* gradAddr = (__ubuf__ U*)grad.GetPhyAddr();
        __ubuf__ U* mTAddr = (__ubuf__ U*)mT.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg pregUp;
            Reg::RegTensor<U> regM;
            Reg::RegTensor<U> regBeta1;
            Reg::RegTensor<U> regGrad;
            Reg::RegTensor<U> regSubMGrad;
            Reg::RegTensor<U> regMulM;
            Reg::RegTensor<U> regMt;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = Reg::UpdateMask<U>(totalLen);

                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regM, mAddr, (int32_t)oneRepeat);
                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regGrad, gradAddr, (int32_t)oneRepeat);

                Reg::Duplicate(regBeta1, beta1, pregUp);
                Reg::Adds(regBeta1, regBeta1, -1.0f, pregUp);
                Reg::Sub(regSubMGrad, regM, regGrad, pregUp);
                Reg::Mul(regMulM, regBeta1, regSubMGrad, pregUp);
                Reg::Add(regMt, regM, regMulM, pregUp);

                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(mTAddr, regMt, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct CalcVt : public Ops::Base::Vec::ElemwiseTernaryOP<U, U, U, float> {
    __aicore__ inline CalcVt(Ops::Base::LocalTensor<U>& vT, Ops::Base::LocalTensor<U>& v,
                             Ops::Base::LocalTensor<U>& grad, float& beta2, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);

        __ubuf__ U* vAddr = (__ubuf__ U*)v.GetPhyAddr();
        __ubuf__ U* gradAddr = (__ubuf__ U*)grad.GetPhyAddr();
        __ubuf__ U* vTAddr = (__ubuf__ U*)vT.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg pregUp;
            Reg::RegTensor<U> regV;
            Reg::RegTensor<U> regBeta2;
            Reg::RegTensor<U> regGrad;
            Reg::RegTensor<U> regGradSquare;
            Reg::RegTensor<U> regSubVGrad;
            Reg::RegTensor<U> regMulV;
            Reg::RegTensor<U> regVt;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = Reg::UpdateMask<U>(totalLen);

                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regV, vAddr, (int32_t)oneRepeat);
                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regGrad, gradAddr, (int32_t)oneRepeat);

                Reg::Duplicate(regBeta2, beta2, pregUp);
                Reg::Adds(regBeta2, regBeta2, -1.0f, pregUp);
                Reg::Mul(regGradSquare, regGrad, regGrad, pregUp);
                Reg::Sub(regSubVGrad, regV, regGradSquare, pregUp);
                Reg::Mul(regMulV, regBeta2, regSubVGrad, pregUp);
                Reg::Add(regVt, regV, regMulV, pregUp);

                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(vTAddr, regVt, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct CalcVarT : public Ops::Base::Vec::Elemwise7OP<U, U, U, U, float, float, float, float> {
    __aicore__ inline CalcVarT(Ops::Base::LocalTensor<U>& varT, Ops::Base::LocalTensor<U>& var,
                               Ops::Base::LocalTensor<U>& mT, Ops::Base::LocalTensor<U>& vT, float& beta1Power,
                               float& beta2Power, float& lr, float& epsilon, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);

        __ubuf__ U* varAddr = (__ubuf__ U*)var.GetPhyAddr();
        __ubuf__ U* varTAddr = (__ubuf__ U*)varT.GetPhyAddr();
        __ubuf__ U* mTAddr = (__ubuf__ U*)mT.GetPhyAddr();
        __ubuf__ U* vTAddr = (__ubuf__ U*)vT.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg pregUp;
            Reg::RegTensor<U> regVar;
            Reg::RegTensor<U> regMt;
            Reg::RegTensor<U> regVt;
            Reg::RegTensor<U> regVarT;
            Reg::RegTensor<U> regEpsilon;
            Reg::RegTensor<U> regLrT;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = Reg::UpdateMask<U>(totalLen);

                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regVar, varAddr, (int32_t)oneRepeat);
                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regMt, mTAddr, (int32_t)oneRepeat);
                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regVt, vTAddr, (int32_t)oneRepeat);

                CalcLr<U>(regLrT, pregUp, beta1Power, beta2Power, lr);
                CalcVarTWithLr<U>(regVarT, regVar, regLrT, regMt, regVt, pregUp, epsilon);

                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(varTAddr, regVarT, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct CalcVarTNesterov : public Ops::Base::Vec::Elemwise9OP<U, U, U, U, U, float, float, float, float, float> {
    __aicore__ inline CalcVarTNesterov(Ops::Base::LocalTensor<U>& varT, Ops::Base::LocalTensor<U>& var,
                                       Ops::Base::LocalTensor<U>& mT, Ops::Base::LocalTensor<U>& vT,
                                       Ops::Base::LocalTensor<U>& grad, float& beta1Power, float& beta2Power, float& lr,
                                       float& beta1, float& epsilon, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);

        __ubuf__ U* varTAddr = (__ubuf__ U*)varT.GetPhyAddr();
        __ubuf__ U* varAddr = (__ubuf__ U*)var.GetPhyAddr();
        __ubuf__ U* mTAddr = (__ubuf__ U*)mT.GetPhyAddr();
        __ubuf__ U* vTAddr = (__ubuf__ U*)vT.GetPhyAddr();
        __ubuf__ U* gradAddr = (__ubuf__ U*)grad.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg pregUp;
            Reg::RegTensor<U> regVar, regMt, regVt, regGrad, regVarT, regLrT, regMtAhead;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = Reg::UpdateMask<U>(totalLen);
                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regVar, varAddr, (int32_t)oneRepeat);
                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regMt, mTAddr, (int32_t)oneRepeat);
                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regVt, vTAddr, (int32_t)oneRepeat);
                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(regGrad, gradAddr, (int32_t)oneRepeat);

                CalcLr<U>(regLrT, pregUp, beta1Power, beta2Power, lr);
                CalcMtLookAhead<U>(regMtAhead, regMt, regGrad, pregUp, beta1);
                CalcVarTWithLr<U>(regVarT, regVar, regLrT, regMtAhead, regVt, pregUp, epsilon);

                Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE>(varTAddr, regVarT, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};

} // namespace Vec
} // namespace AscendC

const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;

template <typename T, typename U = float>
struct ApplyAdamDagFusion {
    // copy in
    using OpCopyInVarOri = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In0<T>>;
    using OpCopyInMOri = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In1<T>>;
    using OpCopyInVOri = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In2<T>>;
    using OpCopyInGradOri = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In9<T>>;

    // cast
    using OpCopyInVar = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_NONE>, OpCopyInVarOri>;
    using OpCopyInM = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_NONE>, OpCopyInMOri>;
    using OpCopyInV = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_NONE>, OpCopyInVOri>;
    using OpCopyInGrad = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_NONE>, OpCopyInGradOri>;

    // calc m_t
    using OpMt = Ops::Base::Bind<AscendC::Vec::CalcMt<T, U>, OpCopyInM, OpCopyInGrad,
                                 Ops::Base::Placeholder::In6<float, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // calc v_t
    using OpVt = Ops::Base::Bind<AscendC::Vec::CalcVt<T, U>, OpCopyInV, OpCopyInGrad,
                                 Ops::Base::Placeholder::In7<float, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // calc var_t
    using OpVarT = Ops::Base::Bind<AscendC::Vec::CalcVarT<T, U>, OpCopyInVar, OpMt, OpVt,
                                   Ops::Base::Placeholder::In3<float, Ops::Base::Placeholder::ScalarAttr<true>>,
                                   Ops::Base::Placeholder::In4<float, Ops::Base::Placeholder::ScalarAttr<true>>,
                                   Ops::Base::Placeholder::In5<float, Ops::Base::Placeholder::ScalarAttr<true>>,
                                   Ops::Base::Placeholder::In8<float, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // cast back
    using OpVarTCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_RINT>, OpVarT>;
    using OpMtCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_RINT>, OpMt>;
    using OpVtCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_RINT>, OpVt>;

    // copy out
    using OpCopyOutVarT = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpVarTCast>;
    using OpCopyOutMt = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out1<T>, OpMtCast>;
    using OpCopyOutVt = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out2<T>, OpVtCast>;

    // sch
    using Outputs = Ops::Base::Elems<OpCopyOutVarT, OpCopyOutMt, OpCopyOutVt>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename U = float>
struct ApplyAdamDagFusionNesterov {
    // copy in
    using OpCopyInVarOriNes = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In0<T>>;
    using OpCopyInMOriNes = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In1<T>>;
    using OpCopyInVOriNes = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In2<T>>;
    using OpCopyInGradOriNes = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In9<T>>;

    // cast
    using OpCopyInVarNes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_NONE>, OpCopyInVarOriNes>;
    using OpCopyInMNes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_NONE>, OpCopyInMOriNes>;
    using OpCopyInVNes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_NONE>, OpCopyInVOriNes>;
    using OpCopyInGradNes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_NONE>, OpCopyInGradOriNes>;

    // calc m_t
    using OpMt = Ops::Base::Bind<AscendC::Vec::CalcMt<T, U>, OpCopyInMNes, OpCopyInGradNes,
                                 Ops::Base::Placeholder::In6<float, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // calc v_t
    using OpVt = Ops::Base::Bind<AscendC::Vec::CalcVt<T, U>, OpCopyInVNes, OpCopyInGradNes,
                                 Ops::Base::Placeholder::In7<float, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // calc var_t
    using OpVarT = Ops::Base::Bind<AscendC::Vec::CalcVarTNesterov<T, U>, OpCopyInVarNes, OpMt, OpVt, OpCopyInGradNes,
                                   Ops::Base::Placeholder::In3<float, Ops::Base::Placeholder::ScalarAttr<true>>,
                                   Ops::Base::Placeholder::In4<float, Ops::Base::Placeholder::ScalarAttr<true>>,
                                   Ops::Base::Placeholder::In5<float, Ops::Base::Placeholder::ScalarAttr<true>>,
                                   Ops::Base::Placeholder::In6<float, Ops::Base::Placeholder::ScalarAttr<true>>,
                                   Ops::Base::Placeholder::In8<float, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // cast back
    using OpVarTCastNes = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_RINT>, OpVarT>;
    using OpMtCastNes = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_RINT>, OpMt>;
    using OpVtCastNes = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_RINT>, OpVt>;

    // copy out
    using OpCopyOutVarTNes = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>,
                                             OpVarTCastNes>;
    using OpCopyOutMtNes = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out1<T>, OpMtCastNes>;
    using OpCopyOutVtNes = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out2<T>, OpVtCastNes>;

    // sch
    using OutputsNes = Ops::Base::Elems<OpCopyOutVarTNes, OpCopyOutMtNes, OpCopyOutVtNes>;
    using MemCfgNes = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<OutputsNes, void, MemCfgNes>;
};

#endif // CANN_CUSTOM_OPS_APPLY_ADAM_D_DAG_H
