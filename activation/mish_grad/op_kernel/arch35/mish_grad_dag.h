/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_CUSTOM_OPS_MISHGRAD_DAG_H
#define CANN_CUSTOM_OPS_MISHGRAD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace AscendC;
using namespace Ops::Base;

const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;
const float FP32_NEG_TWO = -2.0;
const float FP32_TWO = 2.0;
const float FP32_ONE = 1.0;
const float FP32_NEG_ONE = -1.0;
const float FP32_ZERO = 0.0;

namespace MishGradDag1 {

template <class T>
struct MishGradCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline MishGradCustom(LocalTensor<T>& dst, LocalTensor<T>& src, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInput2;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputExp2;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputSqr;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg cmpMaskReg;
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
                        vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
                        vregInput2, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                    MicroAPI::Mul(vregInputSqr, vregInput2, vregInput2, mask); // vregInput2 = tanh(z)
                    MicroAPI::Adds(vregInputSqr, vregInputSqr, FP32_NEG_ONE, mask);
                    MicroAPI::Mul(vregInputSqr, vregInputSqr, vregInput, mask);

                    MicroAPI::Muls(vregInputExp2, vregInput, FP32_NEG_ONE, mask); // vregInput = x
                    MicroAPI::Exp(vregInputExp2, vregInputExp2, mask);
                    MicroAPI::Adds(vregInputExp2, vregInputExp2, FP32_ONE, mask);

                    MicroAPI::Div(vregInputExp2, vregInputSqr, vregInputExp2, mask);
                    MicroAPI::Sub(vregOutput, vregInput2, vregInputExp2, mask);
                    // OpCopyOut
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }
        }
#endif
    }
};

} // namespace MishGradDag1

namespace MishDag1 {

template <class T>
struct MishCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline MishCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputExp;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputNeg;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputNeg2;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputMid;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg cmpMaskReg;
        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                // OpCopyIn
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
                    vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));

                MicroAPI::Muls(vregInputNeg, vregInput, FP32_NEG_ONE, mask);
                MicroAPI::Muls(vregInputNeg2, vregInput, FP32_NEG_TWO, mask);
                MicroAPI::Exp(vregInputNeg, vregInputNeg, mask);   // x = e^(-x)
                MicroAPI::Exp(vregInputNeg2, vregInputNeg2, mask); // x = e^(-2x)

                MicroAPI::Muls(vregInputNeg, vregInputNeg, FP32_TWO, mask);
                MicroAPI::Adds(vregInputNeg, vregInputNeg, FP32_ONE, mask); // x = 2e^(-x)+1
                MicroAPI::Muls(vregInputNeg2, vregInputNeg2, FP32_TWO, mask);
                MicroAPI::Add(vregInputNeg2, vregInputNeg2, vregInputNeg, mask); 
                MicroAPI::Div(vregOutput, vregInputNeg, vregInputNeg2, mask);

                MicroAPI::Muls(vregInputMid, vregInput, FP32_TWO, mask);
                MicroAPI::Exp(vregInputMid, vregInputMid, mask);
                MicroAPI::Exp(vregInputExp, vregInput, mask);
                MicroAPI::Axpy(vregInputMid, vregInputExp, FP32_TWO, mask);
                MicroAPI::Adds(vregInputExp, vregInputMid, FP32_TWO, mask);
                MicroAPI::Div(vregInputMid, vregInputMid, vregInputExp, mask);

                MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, vregInput, FP32_ZERO, mask);
                MicroAPI::Select(vregOutput, vregInputMid, vregOutput, cmpMaskReg);
                // OpCopyOut
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};
} // namespace MishDag1

template <typename U, typename T = float>
struct MishGradFullDAG {
    using ConstValue = MAKE_CONST(float, 1.0);
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpCopyIn2 = Bind<Vec::CopyIn<U>, Placeholder::In2<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn1>;
    using OpCopyIn2Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn2>;

    using OpResult2 = Bind<MishGradDag1::MishGradCustom<T>, OpCopyIn1Cast, OpCopyIn2Cast>;
    using OpResultMul = Bind<Vec::Mul<T>, OpResult2, OpCopyIn0Cast>;

    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResultMul>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct MishGradDAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn1>;

    using OpResult1 = Bind<MishDag1::MishCustom<float>, OpCopyIn1Cast>;
    using OpResult2 = Bind<MishGradDag1::MishGradCustom<T>, OpCopyIn1Cast, OpResult1>;
    using OpResultMul = Bind<Vec::Mul<T>, OpResult2, OpCopyIn0Cast>;
    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResultMul>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

#endif // CANN_CUSTOM_OPS_MISHGRAD_DAG_H