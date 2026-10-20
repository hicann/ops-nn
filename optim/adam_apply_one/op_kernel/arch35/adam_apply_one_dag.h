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
 * \file    adam_apply_one_dag.h
 * \brief adam_apply_one_dag head file
 */

#ifndef ADAM_APPLY_ONE_DAG_H
#define ADAM_APPLY_ONE_DAG_H
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

template <typename U>
__simd_vf__ inline void CalcOutput0VecFull(uint32_t totalLen, uint32_t oneRepeat, uint32_t repeatTimes,
                                           __ubuf__ U* input0Addr, __ubuf__ U* input1Addr, __ubuf__ U* mul2xAddr,
                                           __ubuf__ U* mul3xAddr, __ubuf__ U* output0Addr)
{
    Reg::MaskReg pregUp;
    Reg::RegTensor<U> regInput0;
    Reg::RegTensor<U> regInput1;
    Reg::RegTensor<U> regMul2x;
    Reg::RegTensor<U> regMul3x;
    Reg::RegTensor<U> regPower;
    Reg::RegTensor<U> regTmp1;
    Reg::RegTensor<U> regTmp2;
    Reg::RegTensor<U> regOutput0;

    for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
        pregUp = Reg::UpdateMask<U>(totalLen);

        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regInput0, input0Addr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regInput1, input1Addr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regMul2x, mul2xAddr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regMul3x, mul3xAddr, (int32_t)oneRepeat);

        Reg::Mul(regPower, regInput0, regInput0, pregUp);
        Reg::Mul(regTmp1, regPower, regMul3x, pregUp);
        Reg::Mul(regTmp2, regInput1, regMul2x, pregUp);
        Reg::Add(regOutput0, regTmp1, regTmp2, pregUp);

        Reg::StoreAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(output0Addr, regOutput0, (int32_t)oneRepeat, pregUp);
    }
}

template <typename U>
__simd_vf__ inline void CalcOutput1VecFull(uint32_t totalLen, uint32_t oneRepeat, uint32_t repeatTimes,
                                           __ubuf__ U* input0Addr, __ubuf__ U* input2Addr, __ubuf__ U* mul0xAddr,
                                           __ubuf__ U* mul1xAddr, __ubuf__ U* output1Addr)
{
    Reg::MaskReg pregUp;
    Reg::RegTensor<U> regInput0;
    Reg::RegTensor<U> regInput2;
    Reg::RegTensor<U> regMul0x;
    Reg::RegTensor<U> regMul1x;
    Reg::RegTensor<U> regTmp3;
    Reg::RegTensor<U> regTmp4;
    Reg::RegTensor<U> regOutput1;

    for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
        pregUp = Reg::UpdateMask<U>(totalLen);

        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regInput0, input0Addr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regInput2, input2Addr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regMul0x, mul0xAddr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regMul1x, mul1xAddr, (int32_t)oneRepeat);

        Reg::Mul(regTmp3, regInput2, regMul0x, pregUp);
        Reg::Mul(regTmp4, regInput0, regMul1x, pregUp);
        Reg::Add(regOutput1, regTmp3, regTmp4, pregUp);

        Reg::StoreAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(output1Addr, regOutput1, (int32_t)oneRepeat, pregUp);
    }
}

template <typename U>
__simd_vf__ inline void CalcOutput2VecFull(uint32_t totalLen, uint32_t oneRepeat, uint32_t repeatTimes,
                                           __ubuf__ U* output0Addr, __ubuf__ U* output1Addr, __ubuf__ U* input3Addr,
                                           __ubuf__ U* input4Addr, __ubuf__ U* add2yAddr, __ubuf__ U* output2Addr)
{
    Reg::MaskReg pregUp;
    Reg::RegTensor<U> regOutput0;
    Reg::RegTensor<U> regOutput1;
    Reg::RegTensor<U> regInput3;
    Reg::RegTensor<U> regInput4;
    Reg::RegTensor<U> regAdd2y;
    Reg::RegTensor<U> regSqrt;
    Reg::RegTensor<U> regTmpAdd;
    Reg::RegTensor<U> regTmpDiv;
    Reg::RegTensor<U> regTmpMul;
    Reg::RegTensor<U> regOutput2;

    for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
        pregUp = Reg::UpdateMask<U>(totalLen);

        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regOutput0, output0Addr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regOutput1, output1Addr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regInput3, input3Addr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regInput4, input4Addr, (int32_t)oneRepeat);
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regAdd2y, add2yAddr, (int32_t)oneRepeat);

        Reg::Sqrt(regSqrt, regOutput0, pregUp);
        Reg::Add(regTmpAdd, regSqrt, regAdd2y, pregUp);
        Reg::Div(regTmpDiv, regOutput1, regTmpAdd, pregUp);
        Reg::Mul(regTmpMul, regTmpDiv, regInput4, pregUp);
        Reg::Sub(regOutput2, regInput3, regTmpMul, pregUp);

        Reg::StoreAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(output2Addr, regOutput2, (int32_t)oneRepeat, pregUp);
    }
}
#endif

template <typename T, typename U = float>
struct CalcOutput0Op : public Ops::Base::Vec::ElemwiseQuaternaryOP<U, U, U, U, U> {
    __aicore__ inline CalcOutput0Op(Ops::Base::LocalTensor<U>& output0, Ops::Base::LocalTensor<U>& input0,
                                    Ops::Base::LocalTensor<U>& input1, Ops::Base::LocalTensor<U>& mul2x,
                                    Ops::Base::LocalTensor<U>& mul3x, int32_t count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ U* input0Addr = (__ubuf__ U*)input0.GetPhyAddr();
        __ubuf__ U* input1Addr = (__ubuf__ U*)input1.GetPhyAddr();
        __ubuf__ U* mul2xAddr = (__ubuf__ U*)mul2x.GetPhyAddr();
        __ubuf__ U* mul3xAddr = (__ubuf__ U*)mul3x.GetPhyAddr();
        __ubuf__ U* output0Addr = (__ubuf__ U*)output0.GetPhyAddr();

        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = (uint32_t)count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        CalcOutput0VecFull<U>(totalLen, oneRepeat, repeatTimes, input0Addr, input1Addr, mul2xAddr, mul3xAddr,
                              output0Addr);
#endif
    }
};

template <typename T, typename U = float>
struct CalcOutput1Op : public Ops::Base::Vec::ElemwiseQuaternaryOP<U, U, U, U, U> {
    __aicore__ inline CalcOutput1Op(Ops::Base::LocalTensor<U>& output1, Ops::Base::LocalTensor<U>& input0,
                                    Ops::Base::LocalTensor<U>& input2, Ops::Base::LocalTensor<U>& mul0x,
                                    Ops::Base::LocalTensor<U>& mul1x, int32_t count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ U* input0Addr = (__ubuf__ U*)input0.GetPhyAddr();
        __ubuf__ U* input2Addr = (__ubuf__ U*)input2.GetPhyAddr();
        __ubuf__ U* mul0xAddr = (__ubuf__ U*)mul0x.GetPhyAddr();
        __ubuf__ U* mul1xAddr = (__ubuf__ U*)mul1x.GetPhyAddr();
        __ubuf__ U* output1Addr = (__ubuf__ U*)output1.GetPhyAddr();

        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = (uint32_t)count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        CalcOutput1VecFull<U>(totalLen, oneRepeat, repeatTimes, input0Addr, input2Addr, mul0xAddr, mul1xAddr,
                              output1Addr);
#endif
    }
};

template <typename T, typename U = float>
struct CalcOutput2Op : public Ops::Base::Vec::ElemwiseQuinaryOP<U, U, U, U, U, U> {
    __aicore__ inline CalcOutput2Op(Ops::Base::LocalTensor<U>& output2, Ops::Base::LocalTensor<U>& output0,
                                    Ops::Base::LocalTensor<U>& output1, Ops::Base::LocalTensor<U>& input3,
                                    Ops::Base::LocalTensor<U>& input4, Ops::Base::LocalTensor<U>& add2y, int32_t count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ U* output0Addr = (__ubuf__ U*)output0.GetPhyAddr();
        __ubuf__ U* output1Addr = (__ubuf__ U*)output1.GetPhyAddr();
        __ubuf__ U* input3Addr = (__ubuf__ U*)input3.GetPhyAddr();
        __ubuf__ U* input4Addr = (__ubuf__ U*)input4.GetPhyAddr();
        __ubuf__ U* add2yAddr = (__ubuf__ U*)add2y.GetPhyAddr();
        __ubuf__ U* output2Addr = (__ubuf__ U*)output2.GetPhyAddr();

        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = (uint32_t)count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);
        CalcOutput2VecFull<U>(totalLen, oneRepeat, repeatTimes, output0Addr, output1Addr, input3Addr, input4Addr,
                              add2yAddr, output2Addr);
#endif
    }
};

} // namespace Vec
} // namespace AscendC

namespace AdamApplyOneOp {
template <typename T, typename U>
struct AdamApplyOneCompute {
    using OpInput0 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In0<T>>;
    using OpInput1 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;
    using OpInput2 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In2<T>>;
    using OpInput3 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In3<T>>;
    using OpInput4 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In4<T>>;
    using OpMul0X = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In5<T>>;
    using OpMul1X = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In6<T>>;
    using OpMul2X = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In7<T>>;
    using OpMul3X = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In8<T>>;
    using OpAdd2Y = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In9<T>>;

    using OpInput0Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpInput0>;
    using OpInput1Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpInput1>;
    using OpInput2Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpInput2>;
    using OpInput3Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpInput3>;
    using OpInput4Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpInput4>;
    using OpMul0XCast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpMul0X>;
    using OpMul1XCast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpMul1X>;
    using OpMul2XCast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpMul2X>;
    using OpMul3XCast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpMul3X>;
    using OpAdd2YCast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, 0>, OpAdd2Y>;

    using OpOutput0 = Ops::Base::Bind<AscendC::Vec::CalcOutput0Op<T, U>, OpInput0Cast, OpInput1Cast, OpMul2XCast,
                                      OpMul3XCast>;
    using OpOutput1 = Ops::Base::Bind<AscendC::Vec::CalcOutput1Op<T, U>, OpInput0Cast, OpInput2Cast, OpMul0XCast,
                                      OpMul1XCast>;
    using OpOutput2 = Ops::Base::Bind<AscendC::Vec::CalcOutput2Op<T, U>, OpOutput0, OpOutput1, OpInput3Cast,
                                      OpInput4Cast, OpAdd2YCast>;

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
} // namespace AdamApplyOneOp
#endif // ADAM_APPLY_ONE_DAG_H
