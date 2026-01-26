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
 * \file apply_adam_d_dag.h
 * \brief apply_adam_d_dag head file
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
using MicroAPI::RegTensor;

constexpr static uint16_t VECTOR_LENGTH = Ops::Base::GetVRegSize();

template <typename U = float>
__aicore__ inline void CalcLr(MicroAPI::RegTensor<U> &regLrT, MicroAPI::MaskReg &pregUp,
                              U beta1PowerUp, U beta2PowerUp, U lrUp)
{
    MicroAPI::RegTensor<U> regBeta1Power;
    MicroAPI::RegTensor<U> regBeta2Power;
    MicroAPI::RegTensor<U> regNegBeta2Power;
    MicroAPI::RegTensor<U> regAddBeta2Power;
    MicroAPI::RegTensor<U> regSqrtBeta2Power;
    MicroAPI::RegTensor<U> regNegBeta1Power;
    MicroAPI::RegTensor<U> regAddBeta1Power;
    MicroAPI::RegTensor<U> regMulLrSqrt;

    MicroAPI::Duplicate(regBeta1Power, beta1PowerUp, pregUp);
    MicroAPI::Duplicate(regBeta2Power, beta2PowerUp, pregUp);
    MicroAPI::Muls(regNegBeta2Power, regBeta2Power, -1.0f, pregUp);
    MicroAPI::Adds(regAddBeta2Power, regNegBeta2Power, 1.0f, pregUp);
    MicroAPI::Sqrt(regSqrtBeta2Power, regAddBeta2Power, pregUp);
    MicroAPI::Muls(regNegBeta1Power, regBeta1Power, -1.0f, pregUp);
    MicroAPI::Adds(regAddBeta1Power, regNegBeta1Power, 1.0f, pregUp);
    MicroAPI::Muls(regMulLrSqrt, regSqrtBeta2Power, lrUp, pregUp);
    MicroAPI::Div(regLrT, regMulLrSqrt, regAddBeta1Power, pregUp);
}

template <typename U = float>
__aicore__ inline void CalcVarTWithLr(MicroAPI::RegTensor<U> &regVarT, MicroAPI::RegTensor<U> &regVar,
                                      MicroAPI::RegTensor<U> &regLrT, MicroAPI::RegTensor<U> &regMt,
                                      MicroAPI::RegTensor<U> &regVt, MicroAPI::MaskReg &pregUp, U epsilonUp) {
    MicroAPI::RegTensor<U> regMulLeft;
    MicroAPI::RegTensor<U> regSqrtVt;
    MicroAPI::RegTensor<U> regAddSqrtV;
    MicroAPI::RegTensor<U> regDivRes;

    MicroAPI::Mul(regMulLeft, regLrT, regMt, pregUp);
    MicroAPI::Sqrt(regSqrtVt, regVt, pregUp);
    MicroAPI::Adds(regAddSqrtV, regSqrtVt, epsilonUp, pregUp);
    MicroAPI::Div(regDivRes, regMulLeft, regAddSqrtV, pregUp);
    MicroAPI::Sub(regVarT, regVar, regDivRes, pregUp);
}

template <typename U = float>
__aicore__ inline void CalcMtLookAhead(MicroAPI::RegTensor<U> &regMtAhead, MicroAPI::RegTensor<U> &regMt,
                                       MicroAPI::RegTensor<U> &regGrad, MicroAPI::MaskReg &pregUp, U beta1Up) {
    MicroAPI::RegTensor<U> regBeta1;
    MicroAPI::RegTensor<U> regMulMtBeta1;
    MicroAPI::RegTensor<U> regNegBeta1;
    MicroAPI::RegTensor<U> regSub1Beta1;
    MicroAPI::RegTensor<U> regMulGrad;

    MicroAPI::Duplicate(regBeta1, beta1Up, pregUp);
    MicroAPI::Mul(regMulMtBeta1, regBeta1, regMt, pregUp);
    MicroAPI::Muls(regNegBeta1, regBeta1, -1.0f, pregUp);
    MicroAPI::Adds(regSub1Beta1, regNegBeta1, 1.0f, pregUp);
    MicroAPI::Mul(regMulGrad, regSub1Beta1, regGrad, pregUp);
    MicroAPI::Add(regMtAhead, regMulMtBeta1, regMulGrad, pregUp);
}

#endif

template <typename T, typename U = float>
struct CalcMt : public Ops::Base::Vec::ElemwiseTernaryOP<U, U, U, T> {
    __aicore__ inline CalcMt(Ops::Base::LocalTensor<U>& mT, Ops::Base::LocalTensor<U>& m, Ops::Base::LocalTensor<U>& grad, T &beta1, int32_t count) {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U beta1Up = 0.0f;
        if constexpr (IsSameType<T, bfloat16_t>::value && IsSameType<U, float>::value) {
            beta1Up = ToFloat(beta1);
        } else {
            beta1Up = beta1;
        }

        __ubuf__ U * mAddr = (__ubuf__ U *)m.GetPhyAddr();
        __ubuf__ U * gradAddr = (__ubuf__ U *)grad.GetPhyAddr();
        __ubuf__ U * mTAddr = (__ubuf__ U *)mT.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regM;
            MicroAPI::RegTensor<U> regBeta1;
            MicroAPI::RegTensor<U> regGrad;
            MicroAPI::RegTensor<U> regSubMGrad;
            MicroAPI::RegTensor<U> regMulM;
            MicroAPI::RegTensor<U> regMt;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);

                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regM, mAddr, (int32_t)oneRepeat);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regGrad, gradAddr, (int32_t)oneRepeat);

                MicroAPI::Duplicate(regBeta1, beta1Up, pregUp);
                MicroAPI::Adds(regBeta1, regBeta1, -1.0f, pregUp);
                MicroAPI::Sub(regSubMGrad, regM, regGrad, pregUp);
                MicroAPI::Mul(regMulM, regBeta1, regSubMGrad, pregUp);
                MicroAPI::Add(regMt, regM, regMulM, pregUp);

                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(mTAddr, regMt, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct CalcVt : public Ops::Base::Vec::ElemwiseTernaryOP<U, U, U, T> {
    __aicore__ inline CalcVt(Ops::Base::LocalTensor<U>& vT, Ops::Base::LocalTensor<U>& v, Ops::Base::LocalTensor<U>& grad, T &beta2, int32_t count) {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U beta2Up = 0.0f;
        if constexpr (IsSameType<T, bfloat16_t>::value && IsSameType<U, float>::value) {
            beta2Up = ToFloat(beta2);
        } else {
            beta2Up = beta2;
        }

        __ubuf__ U * vAddr = (__ubuf__ U *)v.GetPhyAddr();
        __ubuf__ U * gradAddr = (__ubuf__ U *)grad.GetPhyAddr();
        __ubuf__ U * vTAddr = (__ubuf__ U *)vT.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regV;
            MicroAPI::RegTensor<U> regBeta2;
            MicroAPI::RegTensor<U> regGrad;
            MicroAPI::RegTensor<U> regGradSquare;
            MicroAPI::RegTensor<U> regSubVGrad;
            MicroAPI::RegTensor<U> regMulV;
            MicroAPI::RegTensor<U> regVt;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);

                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regV, vAddr, (int32_t)oneRepeat);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regGrad, gradAddr, (int32_t)oneRepeat);

                MicroAPI::Duplicate(regBeta2, beta2Up, pregUp);
                MicroAPI::Adds(regBeta2, regBeta2, -1.0f, pregUp);
                MicroAPI::Mul(regGradSquare, regGrad, regGrad, pregUp);
                MicroAPI::Sub(regSubVGrad, regV, regGradSquare, pregUp);
                MicroAPI::Mul(regMulV, regBeta2, regSubVGrad, pregUp);
                MicroAPI::Add(regVt, regV, regMulV, pregUp);

                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vTAddr, regVt, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct CalcVarT : public Ops::Base::Vec::Elemwise7OP<U, U, U, U, T, T, T, T> {
    __aicore__ inline CalcVarT(Ops::Base::LocalTensor<U>& varT, Ops::Base::LocalTensor<U>& var, Ops::Base::LocalTensor<U>& mT, Ops::Base::LocalTensor<U>& vT,
                               T &beta1Power, T &beta2Power, T &lr, T &epsilon, int32_t count) {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);

        U beta1PowerUp = 0.0f;
        U beta2PowerUp = 0.0f;
        U lrUp = 0.0f;
        U epsilonUp = 0.0f;

        if constexpr (IsSameType<T, bfloat16_t>::value && IsSameType<U, float>::value) {
            beta1PowerUp = ToFloat(beta1Power);
            beta2PowerUp = ToFloat(beta2Power);
            lrUp = ToFloat(lr);
            epsilonUp = ToFloat(epsilon);
        } else {
            beta1PowerUp = beta1Power;
            beta2PowerUp = beta2Power;
            lrUp = lr;
            epsilonUp = epsilon;
        }

        __ubuf__ U * varAddr = (__ubuf__ U *)var.GetPhyAddr();
        __ubuf__ U * varTAddr = (__ubuf__ U *)varT.GetPhyAddr();
        __ubuf__ U * mTAddr = (__ubuf__ U *)mT.GetPhyAddr();
        __ubuf__ U * vTAddr = (__ubuf__ U *)vT.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regVar;
            MicroAPI::RegTensor<U> regMt;
            MicroAPI::RegTensor<U> regVt;
            MicroAPI::RegTensor<U> regVarT;
            MicroAPI::RegTensor<U> regEpsilon;
            MicroAPI::RegTensor<U> regLrT;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);

                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regVar, varAddr, (int32_t)oneRepeat);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regMt, mTAddr, (int32_t)oneRepeat);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regVt, vTAddr, (int32_t)oneRepeat);

                CalcLr<U>(regLrT, pregUp, beta1PowerUp, beta2PowerUp, lrUp);
                CalcVarTWithLr<U>(regVarT, regVar, regLrT, regMt, regVt, pregUp, epsilonUp);

                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(varTAddr, regVarT, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct CalcVarTNesterov : public Ops::Base::Vec::Elemwise9OP<U, U, U, U, U, T, T, T, T, T> {
    __aicore__ inline CalcVarTNesterov(Ops::Base::LocalTensor<U>& varT, Ops::Base::LocalTensor<U>& var,
                                       Ops::Base::LocalTensor<U>& mT, Ops::Base::LocalTensor<U>& vT, Ops::Base::LocalTensor<U>& grad,
                                       T &beta1Power, T &beta2Power, T &lr, T &beta1, T &epsilon, int32_t count) {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);

        U beta1PowerUp = 0.0f;
        U beta2PowerUp = 0.0f;
        U lrUp = 0.0f;
        U epsilonUp = 0.0f;
        U beta1Up = 0.0f;

        if constexpr (IsSameType<T, bfloat16_t>::value && IsSameType<U, float>::value) {
            beta1PowerUp = ToFloat(beta1Power);
            beta2PowerUp = ToFloat(beta2Power);
            lrUp = ToFloat(lr);
            beta1Up = ToFloat(beta1);
            epsilonUp = ToFloat(epsilon);
        } else {
            beta1PowerUp = beta1Power;
            beta2PowerUp = beta2Power;
            lrUp = lr;
            beta1Up = beta1;
            epsilonUp = epsilon;
        }

        __ubuf__ U * varTAddr = (__ubuf__ U *)varT.GetPhyAddr();
        __ubuf__ U * varAddr = (__ubuf__ U *)var.GetPhyAddr();
        __ubuf__ U * mTAddr = (__ubuf__ U *)mT.GetPhyAddr();
        __ubuf__ U * vTAddr = (__ubuf__ U *)vT.GetPhyAddr();
        __ubuf__ U * gradAddr = (__ubuf__ U *)grad.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regVar;
            MicroAPI::RegTensor<U> regMt;
            MicroAPI::RegTensor<U> regVt;
            MicroAPI::RegTensor<U> regGrad;
            MicroAPI::RegTensor<U> regVarT;
            MicroAPI::RegTensor<U> regLrT;
            MicroAPI::RegTensor<U> regMtAhead;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);

                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regVar, varAddr, (int32_t)oneRepeat);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regMt, mTAddr, (int32_t)oneRepeat);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regVt, vTAddr, (int32_t)oneRepeat);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regGrad, gradAddr, (int32_t)oneRepeat);

                CalcLr<U>(regLrT, pregUp, beta1PowerUp, beta2PowerUp, lrUp);
                CalcMtLookAhead<U>(regMtAhead, regMt, regGrad, pregUp, beta1Up);
                CalcVarTWithLr<U>(regVarT, regVar, regLrT, regMtAhead, regVt, pregUp, epsilonUp);

                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(varTAddr, regVarT, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};

}
}


const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;

template <typename T, typename U = float>
struct ApplyAdamDDagFusion {
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
    using OpMt = Ops::Base::Bind<AscendC::Vec::CalcMt<T, U>, OpCopyInM, OpCopyInGrad, Ops::Base::Placeholder::In6<T, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // calc v_t
    using OpVt = Ops::Base::Bind<AscendC::Vec::CalcVt<T, U>, OpCopyInV, OpCopyInGrad, Ops::Base::Placeholder::In7<T, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // calc var_t
    using OpVarT = Ops::Base::Bind<AscendC::Vec::CalcVarT<T, U>, OpCopyInVar, OpMt, OpVt,\
                        Ops::Base::Placeholder::In3<T, Ops::Base::Placeholder::ScalarAttr<true>>,\
                        Ops::Base::Placeholder::In4<T, Ops::Base::Placeholder::ScalarAttr<true>>,\
                        Ops::Base::Placeholder::In5<T, Ops::Base::Placeholder::ScalarAttr<true>>,\
                        Ops::Base::Placeholder::In8<T, Ops::Base::Placeholder::ScalarAttr<true>>>;

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
struct ApplyAdamDDagFusionNesterov {
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
    using OpMt = Ops::Base::Bind<AscendC::Vec::CalcMt<T, U>, OpCopyInMNes, OpCopyInGradNes, Ops::Base::Placeholder::In6<T, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // calc v_t
    using OpVt = Ops::Base::Bind<AscendC::Vec::CalcVt<T, U>, OpCopyInVNes, OpCopyInGradNes, Ops::Base::Placeholder::In7<T, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // calc var_t
    using OpVarT = Ops::Base::Bind<AscendC::Vec::CalcVarTNesterov<T, U>, OpCopyInVarNes, OpMt, OpVt, OpCopyInGradNes,\
                        Ops::Base::Placeholder::In3<T, Ops::Base::Placeholder::ScalarAttr<true>>,\
                        Ops::Base::Placeholder::In4<T, Ops::Base::Placeholder::ScalarAttr<true>>,\
                        Ops::Base::Placeholder::In5<T, Ops::Base::Placeholder::ScalarAttr<true>>,\
                        Ops::Base::Placeholder::In6<T, Ops::Base::Placeholder::ScalarAttr<true>>,\
                        Ops::Base::Placeholder::In8<T, Ops::Base::Placeholder::ScalarAttr<true>>>;

    // cast back
    using OpVarTCastNes = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_RINT>, OpVarT>;
    using OpMtCastNes = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_RINT>, OpMt>;
    using OpVtCastNes = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_RINT>, OpVt>;

    // copy out
    using OpCopyOutVarTNes = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpVarTCastNes>;
    using OpCopyOutMtNes = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out1<T>, OpMtCastNes>;
    using OpCopyOutVtNes = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out2<T>, OpVtCastNes>;

    // sch
    using OutputsNes = Ops::Base::Elems<OpCopyOutVarTNes, OpCopyOutMtNes, OpCopyOutVtNes>;
    using MemCfgNes = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<OutputsNes, void, MemCfgNes>;
};

#endif  // CANN_CUSTOM_OPS_APPLY_ADAM_D_DAG_H