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
 * \file  adam_apply_one_with_decay_dag.h
 * \brief adam_apply_one_with_decay_dag head file
 */

#ifndef ADAM_APPLY_ONE_WITH_DECAY_DAG_H
#define ADAM_APPLY_ONE_WITH_DECAY_DAG_H
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
using AscendC::Reg::RegTensor;
using namespace Ops::Base;

constexpr static Reg::CastTrait CAST_B16_TO_B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                   Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static uint16_t VECTOR_LENGTH = Ops::Base::GetVRegSize();

template <typename T, typename U = float>
__aicore__ inline U ScalarToCompute(T scalar)
{
    if constexpr (IsSameType<T, bfloat16_t>::value && IsSameType<U, float>::value) {
        return ToFloat(scalar);
    } else {
        return scalar;
    }
}

template <typename T, typename U = float>
__aicore__ inline void LoadAsCompute(Reg::RegTensor<U>& dst, __ubuf__ T*& input, Reg::MaskReg& pregUp,
                                     int32_t oneRepeat)
{
    Reg::RegTensor<T> regTmp;
    Reg::RegTensor<T> regCopyIn;
    if constexpr (IsSameType<U, float>::value && !IsSameType<T, U>::value) {
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regCopyIn, input, oneRepeat);
        Reg::UnPack((RegTensor<int32_t>&)regTmp, (RegTensor<int16_t>&)regCopyIn);
        Reg::Cast<U, T, CAST_B16_TO_B32>(dst, regTmp, pregUp);
    } else {
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(dst, input, oneRepeat);
    }
}
#endif

template <typename T, typename U = float>
struct AdamApplyOneWithDecayCalcOutput0 : public Ops::Base::Vec::ElemwiseQuaternaryOP<U, T, T, T, T> {
    __aicore__ inline AdamApplyOneWithDecayCalcOutput0(Ops::Base::LocalTensor<U>& output0,
                                                       Ops::Base::LocalTensor<T>& input0,
                                                       Ops::Base::LocalTensor<T>& input1, T& mul2X, T& mul3X,
                                                       int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U mul2XUp = ScalarToCompute<T, U>(mul2X);
        U mul3XUp = ScalarToCompute<T, U>(mul3X);

        __ubuf__ T* input0Addr = (__ubuf__ T*)input0.GetPhyAddr();
        __ubuf__ T* input1Addr = (__ubuf__ T*)input1.GetPhyAddr();
        __ubuf__ U* output0Addr = (__ubuf__ U*)output0.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg pregUp;
            Reg::RegTensor<U> regInput0;
            Reg::RegTensor<U> regInput1;
            Reg::RegTensor<U> regPower;
            Reg::RegTensor<U> regTmp0;
            Reg::RegTensor<U> regTmp1;
            Reg::RegTensor<U> regOutput0;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = Reg::UpdateMask<U>(totalLen);
                LoadAsCompute<T, U>(regInput0, input0Addr, pregUp, (int32_t)oneRepeat);
                LoadAsCompute<T, U>(regInput1, input1Addr, pregUp, (int32_t)oneRepeat);

                Reg::Mul(regPower, regInput0, regInput0, pregUp);
                Reg::Muls(regTmp0, regPower, mul3XUp, pregUp);
                Reg::Muls(regTmp1, regInput1, mul2XUp, pregUp);
                Reg::Add(regOutput0, regTmp0, regTmp1, pregUp);

                Reg::StoreAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(output0Addr, regOutput0, (int32_t)oneRepeat,
                                                                       pregUp);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct AdamApplyOneWithDecayCalcOutput1 : public Ops::Base::Vec::ElemwiseQuaternaryOP<U, T, T, T, T> {
    __aicore__ inline AdamApplyOneWithDecayCalcOutput1(Ops::Base::LocalTensor<U>& output1,
                                                       Ops::Base::LocalTensor<T>& input2,
                                                       Ops::Base::LocalTensor<T>& input0, T& mul0X, T& mul1X,
                                                       int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U mul0XUp = ScalarToCompute<T, U>(mul0X);
        U mul1XUp = ScalarToCompute<T, U>(mul1X);

        __ubuf__ T* input2Addr = (__ubuf__ T*)input2.GetPhyAddr();
        __ubuf__ T* input0Addr = (__ubuf__ T*)input0.GetPhyAddr();
        __ubuf__ U* output1Addr = (__ubuf__ U*)output1.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg pregUp;
            Reg::RegTensor<U> regInput2;
            Reg::RegTensor<U> regInput0;
            Reg::RegTensor<U> regTmp0;
            Reg::RegTensor<U> regTmp1;
            Reg::RegTensor<U> regOutput1;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = Reg::UpdateMask<U>(totalLen);
                LoadAsCompute<T, U>(regInput2, input2Addr, pregUp, (int32_t)oneRepeat);
                LoadAsCompute<T, U>(regInput0, input0Addr, pregUp, (int32_t)oneRepeat);

                Reg::Muls(regTmp0, regInput2, mul0XUp, pregUp);
                Reg::Muls(regTmp1, regInput0, mul1XUp, pregUp);
                Reg::Add(regOutput1, regTmp0, regTmp1, pregUp);

                Reg::StoreAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(output1Addr, regOutput1, (int32_t)oneRepeat,
                                                                       pregUp);
            }
        }
#endif
    }
};

template <typename T, typename U = float>
struct AdamApplyOneWithDecayCalcOutput2 : public Ops::Base::Vec::Elemwise6OP<U, T, U, U, T, T, T> {
    __aicore__ inline AdamApplyOneWithDecayCalcOutput2(Ops::Base::LocalTensor<U>& output2,
                                                       Ops::Base::LocalTensor<T>& input3,
                                                       Ops::Base::LocalTensor<U>& output0,
                                                       Ops::Base::LocalTensor<U>& output1, T& input4, T& mul4X,
                                                       T& add2Y, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U input4Up = ScalarToCompute<T, U>(input4);
        U mul4XUp = ScalarToCompute<T, U>(mul4X);
        U add2YUp = ScalarToCompute<T, U>(add2Y);

        __ubuf__ T* input3Addr = (__ubuf__ T*)input3.GetPhyAddr();
        __ubuf__ U* output0Addr = (__ubuf__ U*)output0.GetPhyAddr();
        __ubuf__ U* output1Addr = (__ubuf__ U*)output1.GetPhyAddr();
        __ubuf__ U* output2Addr = (__ubuf__ U*)output2.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg pregUp;
            Reg::RegTensor<U> regInput3;
            Reg::RegTensor<U> regOutput0;
            Reg::RegTensor<U> regOutput1;
            Reg::RegTensor<U> regSqrt;
            Reg::RegTensor<U> regDenom;
            Reg::RegTensor<U> regDiv;
            Reg::RegTensor<U> regDecay;
            Reg::RegTensor<U> regTmpAdd;
            Reg::RegTensor<U> regTmpMul;
            Reg::RegTensor<U> regOutput2;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = Reg::UpdateMask<U>(totalLen);
                LoadAsCompute<T, U>(regInput3, input3Addr, pregUp, (int32_t)oneRepeat);
                LoadAsCompute<U, U>(regOutput0, output0Addr, pregUp, (int32_t)oneRepeat);
                LoadAsCompute<U, U>(regOutput1, output1Addr, pregUp, (int32_t)oneRepeat);

                Reg::Sqrt(regSqrt, regOutput0, pregUp);
                Reg::Adds(regDenom, regSqrt, add2YUp, pregUp);
                Reg::Div(regDiv, regOutput1, regDenom, pregUp);
                Reg::Muls(regDecay, regInput3, mul4XUp, pregUp);
                Reg::Add(regTmpAdd, regDiv, regDecay, pregUp);
                Reg::Muls(regTmpMul, regTmpAdd, input4Up, pregUp);
                Reg::Sub(regOutput2, regInput3, regTmpMul, pregUp);

                Reg::StoreAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(output2Addr, regOutput2, (int32_t)oneRepeat,
                                                                       pregUp);
            }
        }
#endif
    }
};

} // namespace Vec
} // namespace AscendC

namespace AdamApplyOneWithDecayOp {
template <typename T, typename U>
struct AdamApplyOneWithDecayCompute {
    using OpInput0 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In0<T>>;
    using OpInput1 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;
    using OpInput2 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In2<T>>;
    using OpInput3 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In3<T>>;

    using OpOutput0 = Ops::Base::Bind<AscendC::Vec::AdamApplyOneWithDecayCalcOutput0<T, U>, OpInput0, OpInput1,
                                      Ops::Base::Placeholder::In7<T, Ops::Base::Placeholder::ScalarAttr<true>>,
                                      Ops::Base::Placeholder::In8<T, Ops::Base::Placeholder::ScalarAttr<true>>>;
    using OpOutput1 = Ops::Base::Bind<AscendC::Vec::AdamApplyOneWithDecayCalcOutput1<T, U>, OpInput2, OpInput0,
                                      Ops::Base::Placeholder::In5<T, Ops::Base::Placeholder::ScalarAttr<true>>,
                                      Ops::Base::Placeholder::In6<T, Ops::Base::Placeholder::ScalarAttr<true>>>;
    using OpOutput2 = Ops::Base::Bind<AscendC::Vec::AdamApplyOneWithDecayCalcOutput2<T, U>, OpInput3, OpOutput0,
                                      OpOutput1,
                                      Ops::Base::Placeholder::In4<T, Ops::Base::Placeholder::ScalarAttr<true>>,
                                      Ops::Base::Placeholder::In9<T, Ops::Base::Placeholder::ScalarAttr<true>>,
                                      Ops::Base::Placeholder::In10<T, Ops::Base::Placeholder::ScalarAttr<true>>>;

    using OpOutput0Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 1>, OpOutput0>;
    using OpOutput1Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 1>, OpOutput1>;
    using OpOutput2Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 1>, OpOutput2>;

    using OpCopyOut0 = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpOutput0Cast>;
    using OpCopyOut1 = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out1<T>, OpOutput1Cast>;
    using OpCopyOut2 = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out2<T>, OpOutput2Cast>;

    using Outputs = Ops::Base::Elems<OpCopyOut0, OpCopyOut1, OpCopyOut2>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};
} // namespace AdamApplyOneWithDecayOp
#endif // ADAM_APPLY_ONE_WITH_DECAY_DAG_H
