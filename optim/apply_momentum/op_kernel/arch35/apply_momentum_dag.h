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
 * \file apply_adagrad_d_dag.h
 * \brief 
 */

#ifndef APPLY_MOMENTUM_DAG_H
#define APPLY_MOMENTUM_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#endif

namespace ApplyMomentumCustomCalc {
    #ifdef __CCE_AICORE__
    using namespace Ops::Base;
    using namespace AscendC;

    constexpr static uint32_t VECTOR_LENGTH = AscendC::GetVecLen();
    constexpr static MicroAPI::CastTrait CAST_B16_TO_B32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                    MicroAPI::MaskMergeMode::ZEROING,
                                                    RoundMode::UNKNOWN};  // bf16/fp16 --> float
    constexpr static MicroAPI::CastTrait CAST_B32_TO_B16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                    MicroAPI::MaskMergeMode::ZEROING,
                                                    RoundMode::CAST_RINT};  // float --> bf16/fp16
    
    template <typename U, typename T>
    __aicore__ inline void LoadOneTensor(MicroAPI::RegTensor<T> &dst, __ubuf__ U *&input, MicroAPI::MaskReg &pregUp,
                                        uint32_t oneRepeat) {
        MicroAPI::RegTensor<U> regTmp;
        MicroAPI::RegTensor<U> regCopyIn;
        if constexpr (IsSameType<T, float>::value && !IsSameType<U, T>::value) {
            MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regCopyIn, input, oneRepeat);
            MicroAPI::UnPack((MicroAPI::RegTensor<int32_t> &)regTmp, (MicroAPI::RegTensor<int16_t> &)regCopyIn);
            MicroAPI::Cast<T, U, CAST_B16_TO_B32>(dst, regTmp, pregUp);
        } else {
            MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, input, oneRepeat);
        }
    }

    template <typename U, typename T>
    __aicore__ inline void StoreOneTensor(__ubuf__ U *&output, MicroAPI::RegTensor<T> &src, MicroAPI::MaskReg &pregUp,
                                        uint32_t oneRepeat) {
        MicroAPI::RegTensor<U> regTmp;
        MicroAPI::RegTensor<U> regCopyOut;
        MicroAPI::MaskReg pregT;

        if constexpr (IsSameType<T, float>::value && !IsSameType<U, T>::value) {
            MicroAPI::Cast<U, T, CAST_B32_TO_B16>(regTmp, src, pregUp);
            MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)regCopyOut, (MicroAPI::RegTensor<uint32_t> &)regTmp);
            MicroAPI::MaskPack(pregT, pregUp);
            MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(output, regCopyOut, oneRepeat, pregT);
        } else {
            MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(output, src, oneRepeat, pregUp);
        }
    }

    template <typename U, typename T>
    __aicore__ inline void CalcAccum(__ubuf__ U* accumAddr, __ubuf__ U* gradAddr, __ubuf__ T* accumResAddr, T momentumUp,
                                        uint32_t repeatTimes, uint32_t sizePerRepeat, uint32_t totalLen) {
        MicroAPI::MaskReg pregUp;
        MicroAPI::RegTensor<T> regAccum;
        MicroAPI::RegTensor<T> regGrad;

        for (uint32_t vfRepeat = 0; vfRepeat < repeatTimes; vfRepeat++) {
            // accum = accum * momentum + grad
            pregUp = MicroAPI::UpdateMask<uint32_t>(totalLen);
            LoadOneTensor<U, T>(regAccum, accumAddr, pregUp, sizePerRepeat);
            LoadOneTensor<U, T>(regGrad, gradAddr, pregUp, sizePerRepeat);

            MicroAPI::Muls(regAccum, regAccum, momentumUp, pregUp);
            MicroAPI::Add(regGrad, regAccum, regGrad, pregUp);

            StoreOneTensor<T, T>(accumResAddr, regGrad, pregUp, sizePerRepeat);
        }
    }

    template <typename U, typename T>
    __aicore__ inline void CalcVar(__ubuf__ U* varAddr, __ubuf__ T* accumNewAddr, __ubuf__ U* varResAddr, T lrUp,
                                        uint32_t repeatTimes, uint32_t sizePerRepeat, uint32_t totalLen) {
        MicroAPI::MaskReg pregUp;
        MicroAPI::RegTensor<T> regVar;
        MicroAPI::RegTensor<T> regAccumNew;
        MicroAPI::RegTensor<T> regVarRes;

        for (uint32_t vfRepeat = 0; vfRepeat < repeatTimes; vfRepeat++) {
            // var = var - lr * accum
            pregUp = MicroAPI::UpdateMask<uint32_t>(totalLen);

            LoadOneTensor<U, T>(regVar, varAddr, pregUp, sizePerRepeat);
            LoadOneTensor<T, T>(regAccumNew, accumNewAddr, pregUp, sizePerRepeat);
            MicroAPI::Muls(regAccumNew, regAccumNew, lrUp, pregUp);
            MicroAPI::Sub(regVarRes, regVar, regAccumNew, pregUp);
            StoreOneTensor<U, T>(varResAddr, regVarRes, pregUp, sizePerRepeat);
        }
    }

    template <typename U, typename T>
    __aicore__ inline void CalcVarNesterov(__ubuf__ U* varAddr, __ubuf__ T* accumNewAddr, __ubuf__ U* gradAddr, __ubuf__ U* varResAddr, T lrUp, T momentumUp,
                                        uint32_t repeatTimes, uint32_t sizePerRepeat, uint32_t totalLen) {
        MicroAPI::MaskReg pregUp;
        MicroAPI::RegTensor<T> regVar;
        MicroAPI::RegTensor<T> regAccumNew;
        MicroAPI::RegTensor<T> regGrad;
        MicroAPI::RegTensor<T> regVarRes;

        for (uint32_t vfRepeat = 0; vfRepeat < repeatTimes; vfRepeat++) {
            // var = var - (grad * lr + accum * momentum * lr)
            pregUp = MicroAPI::UpdateMask<uint32_t>(totalLen);

            LoadOneTensor<U, T>(regVar, varAddr, pregUp, sizePerRepeat);
            LoadOneTensor<T, T>(regAccumNew, accumNewAddr, pregUp, sizePerRepeat);
            LoadOneTensor<U, T>(regGrad, gradAddr, pregUp, sizePerRepeat);
            MicroAPI::Muls(regAccumNew, regAccumNew, momentumUp, pregUp);
            MicroAPI::Muls(regAccumNew, regAccumNew, lrUp, pregUp);
            MicroAPI::Muls(regGrad, regGrad, lrUp, pregUp);
            MicroAPI::Add(regAccumNew, regGrad, regAccumNew, pregUp);
            MicroAPI::Sub(regVarRes, regVar, regAccumNew, pregUp);
            StoreOneTensor<U, T>(varResAddr, regVarRes, pregUp, sizePerRepeat);
        }
    }
    #endif

    template <typename U, typename T>
    struct CalcAccumOp : public Ops::Base::Vec::ElemwiseTernaryOP<T, U, U, U> {
        __aicore__ inline CalcAccumOp(Ops::Base::LocalTensor<T> &accumRes, Ops::Base::LocalTensor<U> &accum, Ops::Base::LocalTensor<U> &grad, U &momentum, int32_t count) {
        #ifdef __CCE_AICORE__
            uint32_t totalLen = count;
            uint32_t sizePerRepeat = VECTOR_LENGTH / sizeof(T);
            uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, sizePerRepeat);
            T momentumUp = 0.0f;
            if constexpr (IsSameType<U, bfloat16_t>::value && IsSameType<T, float>::value) {
                momentumUp = ToFloat(momentum);
            } else if constexpr (IsSameType<U, half>::value && IsSameType<T, float>::value) {
                momentumUp = static_cast<T>(momentum);
            } else {
                momentumUp = momentum;
            }

            __ubuf__ U *accumAddr = (__ubuf__ U *)accum.GetPhyAddr();
            __ubuf__ U *gradAddr = (__ubuf__ U *)grad.GetPhyAddr();
            __ubuf__ T *accumResAddr = (__ubuf__ T *)accumRes.GetPhyAddr();

            VF_CALL<CalcAccum<U, T>>(accumAddr, gradAddr, accumResAddr, momentumUp, repeatTimes, sizePerRepeat, totalLen);
        #endif
        }
    };

    template <typename U, typename T>
    struct CalcVarOp : public Ops::Base::Vec::ElemwiseTernaryOP<U, U, T, U> {
        __aicore__ inline CalcVarOp(Ops::Base::LocalTensor<U> &varRes, Ops::Base::LocalTensor<U> &var, Ops::Base::LocalTensor<T> &accumNew, U &lr, int32_t count) {
        #ifdef __CCE_AICORE__
            uint32_t totalLen = count;
            uint32_t sizePerRepeat = VECTOR_LENGTH / sizeof(T);
            uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, sizePerRepeat);
            T lrUp = 0.0f;
            if constexpr (IsSameType<U, bfloat16_t>::value && IsSameType<T, float>::value) {
                lrUp = ToFloat(lr);
            } else if constexpr (IsSameType<U, half>::value && IsSameType<T, float>::value) {
                lrUp = static_cast<T>(lr);
            } else {
                lrUp = lr;
            }

            __ubuf__ U *varAddr = (__ubuf__ U *)var.GetPhyAddr();
            __ubuf__ T *accumNewAddr = (__ubuf__ T *)accumNew.GetPhyAddr();
            __ubuf__ U *varResAddr = (__ubuf__ U *)varRes.GetPhyAddr();

            VF_CALL<CalcVar<U, T>>(varAddr, accumNewAddr, varResAddr, lrUp, repeatTimes, sizePerRepeat, totalLen);
        #endif
        }
    };

    template <typename U, typename T>
    struct CalcVarNesterovOp : public Ops::Base::Vec::ElemwiseQuinaryOP<U, U, T, U, U, U> {
        __aicore__ inline CalcVarNesterovOp(Ops::Base::LocalTensor<U> &varRes, Ops::Base::LocalTensor<U> &var, Ops::Base::LocalTensor<T> &accumNew, U &lr,
                                            Ops::Base::LocalTensor<U> &grad, U &momentum, int32_t count) {
        #ifdef __CCE_AICORE__
            uint32_t totalLen = count;
            uint32_t sizePerRepeat = VECTOR_LENGTH / sizeof(T);
            uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, sizePerRepeat);
            T lrUp = 0.0f;
            T momentumUp = 0.0f;
            if constexpr (IsSameType<U, bfloat16_t>::value && IsSameType<T, float>::value) {
                lrUp = ToFloat(lr);
                momentumUp = ToFloat(momentum);
            } else if constexpr (IsSameType<U, half>::value && IsSameType<T, float>::value) {
                lrUp = static_cast<T>(lr);
                momentumUp = static_cast<T>(momentum);
            } else {
                lrUp = lr;
                momentumUp = momentum;
            }

            __ubuf__ U *varAddr = (__ubuf__ U *)var.GetPhyAddr();
            __ubuf__ T *accumNewAddr = (__ubuf__ T *)accumNew.GetPhyAddr();
            __ubuf__ U *gradAddr = (__ubuf__ U *)grad.GetPhyAddr();
            __ubuf__ U *varResAddr = (__ubuf__ U *)varRes.GetPhyAddr();

            VF_CALL<CalcVarNesterov<U, T>>(varAddr, accumNewAddr, gradAddr, varResAddr, lrUp, momentumUp, repeatTimes, sizePerRepeat, totalLen);
        #endif
        }
    };
}  // namespace ApplyMomentumCustomCalc

namespace ApplyMomentumOp {
    using namespace Ops::Base;

    template <typename U, typename T = float>
    struct ApplyMomentumDag {
        using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
        using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
        using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;

        using OpAccumNew = Bind<ApplyMomentumCustomCalc::CalcAccumOp<U, T>, OpCopyInAccum, OpCopyInGrad,\
                                Placeholder::In4<U, Placeholder::ScalarAttr<true>>>;
        using OpAccumOutCast = Bind<Vec::Cast<U, T, 1>, OpAccumNew>;
        using OpCopyOutAccum = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpAccumOutCast>; // update input1: accum

        using OpVarOutCast = Bind<ApplyMomentumCustomCalc::CalcVarOp<U, T>, OpCopyInVar, OpAccumNew,\
                                    Placeholder::In2<U, Placeholder::ScalarAttr<true>>>;
        using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>; // output: var

        using Outputs = Elems<OpCopyOutVar, OpCopyOutAccum>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };

    template <typename U, typename T = float>
    struct ApplyNesterovMomentumDag {
        using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
        using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
        using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;

        using OpAccumNew = Bind<ApplyMomentumCustomCalc::CalcAccumOp<U, T>, OpCopyInAccum, OpCopyInGrad,\
                                Placeholder::In4<U, Placeholder::ScalarAttr<true>>>;
        using OpAccumOutCast = Bind<Vec::Cast<U, T, 1>, OpAccumNew>;
        using OpCopyOutAccum = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpAccumOutCast>; // update input1: accum

        using OpVarOutCast = Bind<ApplyMomentumCustomCalc::CalcVarNesterovOp<U, T>, OpCopyInVar, OpAccumNew,\
                                    Placeholder::In2<U, Placeholder::ScalarAttr<true>>, OpCopyInGrad,\
                                    Placeholder::In4<U, Placeholder::ScalarAttr<true>>>;
        using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>; // output: var

        using Outputs = Elems<OpCopyOutVar, OpCopyOutAccum>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };
} // namespace ApplyMomentumOp

#endif  // APPLY_MOMENTUM_DAG_H 