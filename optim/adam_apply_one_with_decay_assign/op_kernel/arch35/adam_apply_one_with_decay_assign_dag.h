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
 * \file    adam_apply_one_with_decay_assign_dag.h
 * \brief adam_apply_one_with_decay_assign_dag head file
 */

#ifndef ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_DAG_H
#define ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_DAG_H
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
using namespace Ops::Base;

constexpr static AscendC::Reg::CastTrait ASSIGN_CAST_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE};
constexpr static uint16_t ASSIGN_VECTOR_LENGTH = Ops::Base::GetVRegSize();

template <typename T, typename U = float>
__aicore__ inline U ScalarToComputeAssign(T scalar)
{
    if constexpr (IsSameType<T, bfloat16_t>::value && IsSameType<U, float>::value) {
        return ToFloat(scalar);
    } else {
        return scalar;
    }
}

template <typename T, typename U = float>
__simd_callee__ inline void LoadAsComputeAssign(AscendC::Reg::RegTensor<U>& dst, __ubuf__ T* input, uint32_t offset,
                                                AscendC::Reg::MaskReg& mask)
{
    if constexpr (IsSameType<T, U>::value) {
        AscendC::Reg::LoadAlign<U>(dst, reinterpret_cast<__ubuf__ U*>(input) + offset);
    } else {
        AscendC::Reg::RegTensor<T> regInput;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regInput, input + offset);
        AscendC::Reg::Cast<U, T, ASSIGN_CAST_B16_TO_B32>(dst, regInput, mask);
    }
}

template <typename T, typename U = float>
__simd_vf__ inline void AdamApplyOneWithDecayAssignOutput0VF(__ubuf__ U* output0Addr, __ubuf__ T* input0Addr,
                                                             __ubuf__ T* input1Addr, uint32_t count, uint32_t oneRepeat,
                                                             uint16_t repeatTimes, U mul2X, U mul3X)
{
    AscendC::Reg::RegTensor<U> regInput0;
    AscendC::Reg::RegTensor<U> regInput1;
    AscendC::Reg::RegTensor<U> regPower;
    AscendC::Reg::RegTensor<U> regTmp0;
    AscendC::Reg::RegTensor<U> regTmp1;
    AscendC::Reg::RegTensor<U> regOutput0;
    AscendC::Reg::MaskReg mask;

    uint32_t remain = count;
    for (uint16_t loop = 0; loop < repeatTimes; loop++) {
        uint32_t offset = static_cast<uint32_t>(loop) * oneRepeat;
        mask = AscendC::Reg::UpdateMask<U>(remain);
        LoadAsComputeAssign<T, U>(regInput0, input0Addr, offset, mask);
        LoadAsComputeAssign<T, U>(regInput1, input1Addr, offset, mask);

        AscendC::Reg::Mul<U>(regPower, regInput0, regInput0, mask);
        AscendC::Reg::Muls<U>(regTmp0, regPower, mul3X, mask);
        AscendC::Reg::Muls<U>(regTmp1, regInput1, mul2X, mask);
        AscendC::Reg::Add<U>(regOutput0, regTmp0, regTmp1, mask);

        AscendC::Reg::StoreAlign<U>(output0Addr + offset, regOutput0, mask);
    }
}

template <typename T, typename U = float>
__simd_vf__ inline void AdamApplyOneWithDecayAssignOutput1VF(__ubuf__ U* output1Addr, __ubuf__ T* input2Addr,
                                                             __ubuf__ T* input0Addr, uint32_t count, uint32_t oneRepeat,
                                                             uint16_t repeatTimes, U mul0X, U mul1X)
{
    AscendC::Reg::RegTensor<U> regInput2;
    AscendC::Reg::RegTensor<U> regInput0;
    AscendC::Reg::RegTensor<U> regTmp0;
    AscendC::Reg::RegTensor<U> regTmp1;
    AscendC::Reg::RegTensor<U> regOutput1;
    AscendC::Reg::MaskReg mask;

    uint32_t remain = count;
    for (uint16_t loop = 0; loop < repeatTimes; loop++) {
        uint32_t offset = static_cast<uint32_t>(loop) * oneRepeat;
        mask = AscendC::Reg::UpdateMask<U>(remain);
        LoadAsComputeAssign<T, U>(regInput2, input2Addr, offset, mask);
        LoadAsComputeAssign<T, U>(regInput0, input0Addr, offset, mask);

        AscendC::Reg::Muls<U>(regTmp0, regInput2, mul0X, mask);
        AscendC::Reg::Muls<U>(regTmp1, regInput0, mul1X, mask);
        AscendC::Reg::Add<U>(regOutput1, regTmp0, regTmp1, mask);

        AscendC::Reg::StoreAlign<U>(output1Addr + offset, regOutput1, mask);
    }
}

template <typename T, typename U = float>
__simd_vf__ inline void AdamApplyOneWithDecayAssignOutput2VF(__ubuf__ U* output2Addr, __ubuf__ T* input3Addr,
                                                             __ubuf__ U* output0Addr, __ubuf__ U* output1Addr,
                                                             uint32_t count, uint32_t oneRepeat, uint16_t repeatTimes,
                                                             U input4, U mul4X, U add2Y)
{
    AscendC::Reg::RegTensor<U> regInput3;
    AscendC::Reg::RegTensor<U> regOutput0;
    AscendC::Reg::RegTensor<U> regOutput1;
    AscendC::Reg::RegTensor<U> regSqrt;
    AscendC::Reg::RegTensor<U> regDenom;
    AscendC::Reg::RegTensor<U> regDiv;
    AscendC::Reg::RegTensor<U> regDecay;
    AscendC::Reg::RegTensor<U> regTmpAdd;
    AscendC::Reg::RegTensor<U> regTmpMul;
    AscendC::Reg::RegTensor<U> regOutput2;
    AscendC::Reg::MaskReg mask;

    uint32_t remain = count;
    for (uint16_t loop = 0; loop < repeatTimes; loop++) {
        uint32_t offset = static_cast<uint32_t>(loop) * oneRepeat;
        mask = AscendC::Reg::UpdateMask<U>(remain);
        LoadAsComputeAssign<T, U>(regInput3, input3Addr, offset, mask);
        LoadAsComputeAssign<U, U>(regOutput0, output0Addr, offset, mask);
        LoadAsComputeAssign<U, U>(regOutput1, output1Addr, offset, mask);

        AscendC::Reg::Sqrt<U>(regSqrt, regOutput0, mask);
        AscendC::Reg::Adds<U>(regDenom, regSqrt, add2Y, mask);
        AscendC::Reg::Div<U>(regDiv, regOutput1, regDenom, mask);
        AscendC::Reg::Muls<U>(regDecay, regInput3, mul4X, mask);
        AscendC::Reg::Add<U>(regTmpAdd, regDiv, regDecay, mask);
        AscendC::Reg::Muls<U>(regTmpMul, regTmpAdd, input4, mask);
        AscendC::Reg::Sub<U>(regOutput2, regInput3, regTmpMul, mask);

        AscendC::Reg::StoreAlign<U>(output2Addr + offset, regOutput2, mask);
    }
}
#endif

template <typename T, typename U = float>
struct AdamApplyOneWithDecayAssignCalcOutput0 : public Ops::Base::Vec::ElemwiseQuaternaryOP<U, T, T, T, T> {
    __aicore__ inline AdamApplyOneWithDecayAssignCalcOutput0(Ops::Base::LocalTensor<U>& output0,
                                                             Ops::Base::LocalTensor<T>& input0,
                                                             Ops::Base::LocalTensor<T>& input1, T& mul2X, T& mul3X,
                                                             int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = ASSIGN_VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U mul2XUp = ScalarToComputeAssign<T, U>(mul2X);
        U mul3XUp = ScalarToComputeAssign<T, U>(mul3X);

        __ubuf__ T* input0Addr = (__ubuf__ T*)input0.GetPhyAddr();
        __ubuf__ T* input1Addr = (__ubuf__ T*)input1.GetPhyAddr();
        __ubuf__ U* output0Addr = (__ubuf__ U*)output0.GetPhyAddr();

        asc_vf_call<AdamApplyOneWithDecayAssignOutput0VF<T, U>>(output0Addr, input0Addr, input1Addr, totalLen,
                                                                oneRepeat, (uint16_t)repeatTimes, mul2XUp, mul3XUp);
#endif
    }
};

template <typename T, typename U = float>
struct AdamApplyOneWithDecayAssignCalcOutput1 : public Ops::Base::Vec::ElemwiseQuaternaryOP<U, T, T, T, T> {
    __aicore__ inline AdamApplyOneWithDecayAssignCalcOutput1(Ops::Base::LocalTensor<U>& output1,
                                                             Ops::Base::LocalTensor<T>& input2,
                                                             Ops::Base::LocalTensor<T>& input0, T& mul0X, T& mul1X,
                                                             int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = ASSIGN_VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U mul0XUp = ScalarToComputeAssign<T, U>(mul0X);
        U mul1XUp = ScalarToComputeAssign<T, U>(mul1X);

        __ubuf__ T* input2Addr = (__ubuf__ T*)input2.GetPhyAddr();
        __ubuf__ T* input0Addr = (__ubuf__ T*)input0.GetPhyAddr();
        __ubuf__ U* output1Addr = (__ubuf__ U*)output1.GetPhyAddr();

        asc_vf_call<AdamApplyOneWithDecayAssignOutput1VF<T, U>>(output1Addr, input2Addr, input0Addr, totalLen,
                                                                oneRepeat, (uint16_t)repeatTimes, mul0XUp, mul1XUp);
#endif
    }
};

template <typename T, typename U = float>
struct AdamApplyOneWithDecayAssignCalcOutput2 : public Ops::Base::Vec::Elemwise6OP<U, T, U, U, T, T, T> {
    __aicore__ inline AdamApplyOneWithDecayAssignCalcOutput2(Ops::Base::LocalTensor<U>& output2,
                                                             Ops::Base::LocalTensor<T>& input3,
                                                             Ops::Base::LocalTensor<U>& output0,
                                                             Ops::Base::LocalTensor<U>& output1, T& input4, T& mul4X,
                                                             T& add2Y, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = ASSIGN_VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        U input4Up = ScalarToComputeAssign<T, U>(input4);
        U mul4XUp = ScalarToComputeAssign<T, U>(mul4X);
        U add2YUp = ScalarToComputeAssign<T, U>(add2Y);

        __ubuf__ T* input3Addr = (__ubuf__ T*)input3.GetPhyAddr();
        __ubuf__ U* output0Addr = (__ubuf__ U*)output0.GetPhyAddr();
        __ubuf__ U* output1Addr = (__ubuf__ U*)output1.GetPhyAddr();
        __ubuf__ U* output2Addr = (__ubuf__ U*)output2.GetPhyAddr();

        asc_vf_call<AdamApplyOneWithDecayAssignOutput2VF<T, U>>(output2Addr, input3Addr, output0Addr, output1Addr,
                                                                totalLen, oneRepeat, (uint16_t)repeatTimes, input4Up,
                                                                mul4XUp, add2YUp);
#endif
    }
};

} // namespace Vec
} // namespace AscendC

namespace AdamApplyOneWithDecayAssignOp {
template <typename T, typename U>
struct AdamApplyOneWithDecayAssignCompute {
    using OpInput0 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In0<T>>;
    using OpInput1 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;
    using OpInput2 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In2<T>>;
    using OpInput3 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In3<T>>;

    using OpOutput0 = Ops::Base::Bind<AscendC::Vec::AdamApplyOneWithDecayAssignCalcOutput0<T, U>, OpInput0, OpInput1,
                                      Ops::Base::Placeholder::In7<T, Ops::Base::Placeholder::ScalarAttr<true>>,
                                      Ops::Base::Placeholder::In8<T, Ops::Base::Placeholder::ScalarAttr<true>>>;
    using OpOutput1 = Ops::Base::Bind<AscendC::Vec::AdamApplyOneWithDecayAssignCalcOutput1<T, U>, OpInput2, OpInput0,
                                      Ops::Base::Placeholder::In5<T, Ops::Base::Placeholder::ScalarAttr<true>>,
                                      Ops::Base::Placeholder::In6<T, Ops::Base::Placeholder::ScalarAttr<true>>>;
    using OpOutput2 = Ops::Base::Bind<AscendC::Vec::AdamApplyOneWithDecayAssignCalcOutput2<T, U>, OpInput3, OpOutput0,
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
} // namespace AdamApplyOneWithDecayAssignOp
#endif // ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_DAG_H
