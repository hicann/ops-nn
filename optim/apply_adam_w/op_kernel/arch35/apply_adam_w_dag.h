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
 * \file apply_adam_w_dag.h
 * \brief
 */

#ifndef APPLY_ADAM_W_DAG_H
#define APPLY_ADAM_W_DAG_H
#include "op_kernel/atvoss/atp/dag.h"
#include "op_kernel/atvoss/atp/vec.h"
#include "op_kernel/atvoss/atp/placeholder.h"

#ifdef __CCE_AICORE__
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#endif

namespace AscendC {
namespace Vec {
#ifdef __CCE_AICORE__
using AscendC::MicroAPI::RegTensor;

constexpr static MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING,
                                                   RoundMode::UNKNOWN};  // bf16/fp16 --float

constexpr static MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                   MicroAPI::MaskMergeMode::ZEROING,
                                                   RoundMode::CAST_RINT};  // float---bf16/fp16

constexpr static uint16_t VECTOR_LENGTH = platform::GetVRegSize();

template <typename T, typename U = float>
__aicore__ inline void LoadOneTensor(MicroAPI::RegTensor<U> &dst, __local_mem__ T *&input, MicroAPI::MaskReg &pregUp,
                                     int32_t oneRepeat) {
    MicroAPI::RegTensor<T> regTmp;
    MicroAPI::RegTensor<T> regCopyIn;
    if constexpr (IsSameType<U, float>::value && !IsSameType<T, U>::value) {
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regCopyIn, input, (int32_t)oneRepeat);
        MicroAPI::UnPack((RegTensor<int32_t> &)regTmp, (RegTensor<int16_t> &)regCopyIn);
        MicroAPI::Cast<U, T, castTrait0>(dst, regTmp, pregUp);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, input, (int32_t)oneRepeat);
    }
}

template <typename T, typename U = float>
__aicore__ inline void StoreOneTensor(__local_mem__ T *&output, MicroAPI::RegTensor<U> &src, MicroAPI::MaskReg &pregUp,
                                      int32_t oneRepeat) {
    MicroAPI::RegTensor<T> regTmp;
    MicroAPI::RegTensor<T> regCopyOut;
    MicroAPI::MaskReg pregT;

    if constexpr (IsSameType<U, float>::value && !IsSameType<T, U>::value) {
        MicroAPI::Cast<T, U, castTrait1>(regTmp, src, pregUp);
        MicroAPI::Pack((RegTensor<uint16_t> &)regCopyOut, (RegTensor<uint32_t> &)regTmp);
        MicroAPI::MaskPack(pregT, pregUp);
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(output, regCopyOut, (int32_t)oneRepeat, pregT);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(output, src, (int32_t)oneRepeat, pregUp);
    }
}

template <typename U = float>
__aicore__ inline void CalcVarT(MicroAPI::RegTensor<U> &regVarT, MicroAPI::MaskReg &pregUp,
                                MicroAPI::RegTensor<U> &regVar, U weightDecayUp, U lrUp) {
    // // var_t = var * (1 + (-lr * weight_decay))
    MicroAPI::RegTensor<U> regWeightDecay;
    MicroAPI::RegTensor<U> regLr;

    MicroAPI::Duplicate(regWeightDecay, weightDecayUp, pregUp);
    MicroAPI::Muls(regLr, regWeightDecay, lrUp, pregUp);
    MicroAPI::Muls(regLr, regLr, -1.0f, pregUp);
    MicroAPI::Adds(regLr, regLr, 1.0f, pregUp);
    MicroAPI::Mul(regVarT, regVar, regLr, pregUp);
}

template <typename U = float>
__aicore__ inline void CalcDenom(MicroAPI::RegTensor<U> &regDenom, MicroAPI::MaskReg &pregUp,
                                 MicroAPI::RegTensor<U> &regVOut, U beta2PowerUp, U beta2Up, U epsilonUp) {
    MicroAPI::RegTensor<U> regBeta2Power;
    MicroAPI::RegTensor<U> regTmp1;
    MicroAPI::RegTensor<U> regAddSqrtV;
    MicroAPI::RegTensor<U> regDivRes;
    MicroAPI::RegTensor<U> regSqrtVt;
    MicroAPI::RegTensor<U> regVtLeft;
    // v_t_left = -1 * v_out
    // beta2_power_out = beta2_power * beta2
    // denom = sqrt(v_t_left / (beta2_power_out - 1)) + epsilon
    MicroAPI::Muls(regVtLeft, regVOut, -1.0f, pregUp);
    MicroAPI::Duplicate(regBeta2Power, beta2PowerUp, pregUp);
    MicroAPI::Muls(regTmp1, regBeta2Power, beta2Up, pregUp);
    MicroAPI::Adds(regTmp1, regTmp1, -1.0f, pregUp);
    MicroAPI::Div(regDivRes, regVtLeft, regTmp1, pregUp);
    MicroAPI::Sqrt(regSqrtVt, regDivRes, pregUp);
    MicroAPI::Adds(regDenom, regSqrtVt, epsilonUp, pregUp);
}

// CalcDataVarOut<U>(regVarOut, pregUp, regVarT, regDenom, regMOut, beta1PowerUp, beta1Up, lrUp);
template <typename U = float>
__aicore__ inline void CalcDataVarOut(MicroAPI::RegTensor<U> &regVarOut, MicroAPI::MaskReg &pregUp,
                                      MicroAPI::RegTensor<U> &regVarT, MicroAPI::RegTensor<U> &regDenom,
                                      MicroAPI::RegTensor<U> &regMOut, U beta1PowerUp, U beta1Up, U lrUp) {
    MicroAPI::RegTensor<U> regTmp1;
    MicroAPI::RegTensor<U> regTmp2;
    MicroAPI::RegTensor<U> regTmp3;
    MicroAPI::RegTensor<U> regLr;
    MicroAPI::RegTensor<U> regStepSize;
    MicroAPI::RegTensor<U> regBeta1Power;
    // beta1_power_out = beta1_power * beta1
    // step_size = lr / (beta1_power_out - 1)
    // data_var_out = var_t + step_size * (m_out / denom)
    MicroAPI::Duplicate(regBeta1Power, beta1PowerUp, pregUp);
    MicroAPI::Muls(regTmp1, regBeta1Power, beta1Up, pregUp);
    MicroAPI::Adds(regTmp1, regTmp1, -1.0f, pregUp);
    MicroAPI::Duplicate(regLr, lrUp, pregUp);
    MicroAPI::Div(regStepSize, regLr, regTmp1, pregUp);
    MicroAPI::Mul(regTmp2, regStepSize, regMOut, pregUp);
    MicroAPI::Div(regTmp3, regTmp2, regDenom, pregUp);
    MicroAPI::Add(regVarOut, regVarT, regTmp3, pregUp);
}

#endif
template <typename T, typename U = float>
struct CalcGt : public ElemwiseBinaryOP<U, T, U> {
    __aicore__ inline CalcGt(LocalTensor<U> &gTOut, LocalTensor<T> &grad, U &maximizeFactor, int32_t count) {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = ops::CeilDiv<uint32_t>(totalLen, oneRepeat);

        __ubuf__ T *gradAddr = (__ubuf__ T *)grad.GetPhyAddr();
        __ubuf__ U *gTOutAddr = (__ubuf__ U *)gTOut.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regGrad;
            MicroAPI::RegTensor<U> regGtOut;

            // gt = maximizeFactor * gt
            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);

                LoadOneTensor(regGrad, gradAddr, pregUp, (int32_t)oneRepeat);
                MicroAPI::Muls(regGtOut, regGrad, maximizeFactor, pregUp);
                StoreOneTensor<U, U>(gTOutAddr, regGtOut, pregUp, (int32_t)oneRepeat);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct CalcM : public ElemwiseTernaryOP<U, T, U, T> {
    __aicore__ inline CalcM(LocalTensor<U> &mOut, LocalTensor<T> &m, LocalTensor<U> &grad, T &beta1, int32_t count) {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = ops::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U beta1Up = 0.0f;
        if constexpr (IsSameType<T, bfloat16_t>::value && IsSameType<U, float>::value) {
            beta1Up = ToFloat(beta1);
        } else {
            beta1Up = beta1;
        }

        __ubuf__ T *mAddr = (__ubuf__ T *)m.GetPhyAddr();
        __ubuf__ U *gradAddr = (__ubuf__ U *)grad.GetPhyAddr();
        __ubuf__ U *mOutAddr = (__ubuf__ U *)mOut.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regM;
            MicroAPI::RegTensor<U> regBeta1;
            MicroAPI::RegTensor<U> regGrad;
            MicroAPI::RegTensor<U> regBeta1Sub1;
            MicroAPI::RegTensor<U> regMOut;

            // m_out = m * beta1 - (beta1 - 1) * gt
            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);

                LoadOneTensor(regM, mAddr, pregUp, (int32_t)oneRepeat);
                LoadOneTensor<U, U>(regGrad, gradAddr, pregUp, (int32_t)oneRepeat);

                MicroAPI::Duplicate(regBeta1, beta1Up, pregUp);
                MicroAPI::Adds(regBeta1Sub1, regBeta1, -1.0f, pregUp);
                MicroAPI::Mul(regBeta1Sub1, regBeta1Sub1, regGrad, pregUp);
                MicroAPI::Mul(regM, regM, regBeta1, pregUp);
                MicroAPI::Sub(regMOut, regM, regBeta1Sub1, pregUp);

                StoreOneTensor<U, U>(mOutAddr, regMOut, pregUp, (int32_t)oneRepeat);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct CalcV : public ElemwiseTernaryOP<U, T, U, T> {
    __aicore__ inline CalcV(LocalTensor<U> &vOut, LocalTensor<T> &v, LocalTensor<U> &grad, T &beta2, int32_t count) {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = ops::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U beta2Up = 0.0f;
        if constexpr (IsSameType<T, bfloat16_t>::value && IsSameType<U, float>::value) {
            beta2Up = ToFloat(beta2);
        } else {
            beta2Up = beta2;
        }

        __ubuf__ T *vAddr = (__ubuf__ T *)v.GetPhyAddr();
        __ubuf__ U *gradAddr = (__ubuf__ U *)grad.GetPhyAddr();
        __ubuf__ U *vOutAddr = (__ubuf__ U *)vOut.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regV;
            MicroAPI::RegTensor<U> regBeta2;
            MicroAPI::RegTensor<U> regGrad;
            MicroAPI::RegTensor<U> regTmp1;
            MicroAPI::RegTensor<U> regBeta2Sub1;
            MicroAPI::RegTensor<U> regVOut;
            // v_out = v * beta2 - (beta2 - 1) * gt * gt
            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);

                LoadOneTensor(regV, vAddr, pregUp, (int32_t)oneRepeat);
                LoadOneTensor<U, U>(regGrad, gradAddr, pregUp, (int32_t)oneRepeat);
                MicroAPI::Duplicate(regBeta2, beta2Up, pregUp);
                MicroAPI::Adds(regBeta2Sub1, regBeta2, -1.0f, pregUp);
                MicroAPI::Mul(regBeta2Sub1, regBeta2Sub1, regGrad, pregUp);
                MicroAPI::Mul(regBeta2Sub1, regBeta2Sub1, regGrad, pregUp);
                MicroAPI::Mul(regTmp1, regBeta2, regV, pregUp);
                MicroAPI::Sub(regVOut, regTmp1, regBeta2Sub1, pregUp);

                StoreOneTensor<U, U>(vOutAddr, regVOut, pregUp, (int32_t)oneRepeat);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct CalcVar : public Elemwise10OP<U, T, U, U, T, T, T, T, T, T, T> {
    __aicore__ inline CalcVar(LocalTensor<U> &varOut, LocalTensor<T> &var, LocalTensor<U> &mOut, LocalTensor<U> &vOut,
                              T &beta1Power, T &beta2Power, T &lr, T &weightDecay, T &beta1, T &beta2, T &epsilon,
                              int32_t count) {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = ops::CeilDiv<uint32_t>(totalLen, oneRepeat);

        U beta1PowerUp = 0.0f;
        U beta2PowerUp = 0.0f;
        U lrUp = 0.0f;
        U weightDecayUp = 0.0f;
        U beta1Up = 0.0f;
        U beta2Up = 0.0f;
        U epsilonUp = 0.0f;

        if constexpr (IsSameType<T, bfloat16_t>::value && IsSameType<U, float>::value) {
            beta1PowerUp = ToFloat(beta1Power);
            beta2PowerUp = ToFloat(beta2Power);
            lrUp = ToFloat(lr);
            weightDecayUp = ToFloat(weightDecay);
            beta1Up = ToFloat(beta1);
            beta2Up = ToFloat(beta2);
            epsilonUp = ToFloat(epsilon);
        } else {
            beta1PowerUp = beta1Power;
            beta2PowerUp = beta2Power;
            lrUp = lr;
            weightDecayUp = weightDecay;
            beta1Up = beta1;
            beta2Up = beta2;
            epsilonUp = epsilon;
        }

        __ubuf__ U *varOutAddr = (__ubuf__ U *)varOut.GetPhyAddr();
        __ubuf__ T *varAddr = (__ubuf__ T *)var.GetPhyAddr();
        __ubuf__ U *mOutAddr = (__ubuf__ U *)mOut.GetPhyAddr();
        __ubuf__ U *vOutAddr = (__ubuf__ U *)vOut.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regVar, regMOut, regVOut, regVarT, regDenom, regVarOut;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);

                LoadOneTensor(regVar, varAddr, pregUp, (int32_t)oneRepeat);
                LoadOneTensor<U, U>(regMOut, mOutAddr, pregUp, (int32_t)oneRepeat);
                LoadOneTensor<U, U>(regVOut, vOutAddr, pregUp, (int32_t)oneRepeat);
                CalcVarT<U>(regVarT, pregUp, regVar, weightDecayUp, lrUp);
                CalcDenom<U>(regDenom, pregUp, regVOut, beta2PowerUp, beta2Up, epsilonUp);
                CalcDataVarOut<U>(regVarOut, pregUp, regVarT, regDenom, regMOut, beta1PowerUp, beta1Up, lrUp);
                StoreOneTensor<U, U>(varOutAddr, regVarOut, pregUp, (int32_t)oneRepeat);
            }
        }
#endif
    }
};

}  // namespace Vec
}  // namespace AscendC

namespace ApplyAdamWOp {
using namespace AscendC;

template <typename T, typename U = float>
struct ApplyAdamWDAG {
  using OpVar = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
  using OpM = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
  using OpV = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
  using OpGrad = Bind<Vec::CopyIn<T>, Placeholder::In10<T>>;
  using OpGt = Bind<Vec::CalcGt<T, U>, OpGrad, Placeholder::Var<U, 0>>;
  using OpMOut = Bind<Vec::CalcM<T, U>, OpM, OpGt, Placeholder::In7<T, Placeholder::ScalarAttr<true>>>;
  using OpVOut = Bind<Vec::CalcV<T, U>, OpV, OpGt, Placeholder::In8<T, Placeholder::ScalarAttr<true>>>;
  using OpVarOut =
      Bind<Vec::CalcVar<T, U>, OpVar, OpMOut, OpVOut, Placeholder::In3<T, Placeholder::ScalarAttr<true>>,\
           Placeholder::In4<T, Placeholder::ScalarAttr<true>>, Placeholder::In5<T, Placeholder::ScalarAttr<true>>,\
           Placeholder::In6<T, Placeholder::ScalarAttr<true>>, Placeholder::In7<T, Placeholder::ScalarAttr<true>>,\
           Placeholder::In8<T, Placeholder::ScalarAttr<true>>, Placeholder::In9<T, Placeholder::ScalarAttr<true>>>;
  
  using OpVarOutCast = Bind<Vec::Cast<T, U, 1>, OpVarOut>;
  using OpMOutCast = Bind<Vec::Cast<T, U, 1>, OpMOut>;
  using OpVOutCast = Bind<Vec::Cast<T, U, 1>, OpVOut>;
  using OpCopyVarOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpVarOutCast>;
  using OpCopyMOut = Bind<Vec::CopyOut<T>, Placeholder::Out1<T>, OpMOutCast>;
  using OpCopyVOut = Bind<Vec::CopyOut<T>, Placeholder::Out2<T>, OpVOutCast>;

  using Outputs = Elems<OpCopyVarOut, OpCopyMOut, OpCopyVOut>;
  using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
  using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename U = float>
struct ApplyAdamWAmsGradDAG {
  using OpVar = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
  using OpM = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
  using OpV = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
  using OpGrad = Bind<Vec::CopyIn<T>, Placeholder::In10<T>>;
  using OpMaxGradNorm = Bind<Vec::CopyIn<T>, Placeholder::In11<T>>;
  using OpMaxGradNormCast = Bind<Vec::Cast<U, T, 0>, OpMaxGradNorm>;

  using OpGt = Bind<Vec::CalcGt<T, U>, OpGrad, Placeholder::Var<U, 0>>;
  using OpMOut = Bind<Vec::CalcM<T, U>, OpM, OpGt, Placeholder::In7<T, Placeholder::ScalarAttr<true>>>;
  using OpVOut = Bind<Vec::CalcV<T, U>, OpV, OpGt, Placeholder::In8<T, Placeholder::ScalarAttr<true>>>;
  using OpVMax = Bind<Vec::Max<U>, OpVOut, OpMaxGradNormCast>;
  using OpVarOut =
      Bind<Vec::CalcVar<T, U>, OpVar, OpMOut, OpVMax, Placeholder::In3<T, Placeholder::ScalarAttr<true>>,\
           Placeholder::In4<T, Placeholder::ScalarAttr<true>>, Placeholder::In5<T, Placeholder::ScalarAttr<true>>,\
           Placeholder::In6<T, Placeholder::ScalarAttr<true>>, Placeholder::In7<T, Placeholder::ScalarAttr<true>>,\
           Placeholder::In8<T, Placeholder::ScalarAttr<true>>, Placeholder::In9<T, Placeholder::ScalarAttr<true>>>;

  using OpVarOutCast = Bind<Vec::Cast<T, U, 1>, OpVarOut>;
  using OpMOutCast = Bind<Vec::Cast<T, U, 1>, OpMOut>;
  using OpVOutCast = Bind<Vec::Cast<T, U, 1>, OpVOut>;
  using OpCopyVarOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpVarOutCast>;
  using OpCopyMOut = Bind<Vec::CopyOut<T>, Placeholder::Out1<T>, OpMOutCast>;
  using OpCopyVOut = Bind<Vec::CopyOut<T>, Placeholder::Out2<T>, OpVOutCast>;

  using Outputs = Elems<OpCopyVarOut, OpCopyMOut, OpCopyVOut>;
  using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
  using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace ApplyAdamWOp
#endif  // APPLY_ADAM_W_DAG_H
