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
 * \file gelu_grad_dag_v2.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_GELU_GRAD_V2_DAG_H
#define CANN_CUSTOM_OPS_GELU_GRAD_V2_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace GeluGradV2Op {
using namespace Ops::Base;
using namespace AscendC;

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;
constexpr float BETAN = -1.595769121605730711759f;
constexpr float AN = -0.0713548162726002527220f;
constexpr float A3 = 0.2140644488178007f;
constexpr float BETA = 1.595769121605730711759f;

template <class T>
struct ErfFast : public Vec::ElemwiseUnaryOP<T, T, 0, 0, true> {
    __aicore__ inline ErfFast(LocalTensor<T>& dst, LocalTensor<T>& src, int count)
    {
#ifdef __CCE_AICORE__
        AscendC::Erf(dst, src, count);
#endif
    }
};

template <class T>
struct GeluGradV2ErfPost : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline GeluGradV2ErfPost(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1,
                                        LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput0;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput1;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput2;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregCdfMuls;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregCdfRes;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregPdfMul;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregPdfMuls;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregPdfExp;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregPdfRes;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregMulRes;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregAddRes;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput;
        Reg::MaskReg mask;
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::LoadAlign(vregInput0, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                    Reg::LoadAlign(vregInput1, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                    Reg::LoadAlign(vregInput2, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));
                    Reg::Muls(vregCdfMuls, vregInput2, (float)0.5, mask);
                    Reg::Adds(vregCdfRes, vregCdfMuls, (float)0.5, mask);

                    Reg::Mul(vregPdfMul, vregInput1, vregInput1, mask);
                    Reg::Muls(vregPdfMuls, vregPdfMul, (float)-0.5, mask);
                    Reg::Exp(vregPdfExp, vregPdfMuls, mask);
                    Reg::Muls(vregPdfRes, vregPdfExp, (float)0.3989422804, mask); // 1 / sqrt(2 * pi)

                    Reg::Mul(vregMulRes, vregPdfRes, vregInput1, mask);
                    Reg::Add(vregAddRes, vregCdfRes, vregMulRes, mask);
                    Reg::Mul(vregOutput, vregAddRes, vregInput0, mask);
                    // OpCopyOut
                    Reg::StoreAlign((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }
        }
#endif
    }
};

template <class T>
struct GeluGradV2TanhCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline GeluGradV2TanhCustom(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1,
                                           uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;
        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputDy;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputX;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputXSqr;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputPX;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputRes0;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputT;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputDiv;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputOne;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputZero;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputResp;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregSelect;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput;
        Reg::MaskReg mask, cmpMask;

        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                Reg::Duplicate(vregInputOne, (float)1.0);
                Reg::Duplicate(vregInputZero, (float)0.0);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                    Reg::Duplicate(vregInputPX, BETAN);
                    // OpCopyIn
                    Reg::LoadAlign(vregInputDy, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                    Reg::LoadAlign(vregInputX, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                    // compute
                    Reg::Mul(vregInputXSqr, vregInputX, vregInputX, mask);
                    Reg::Axpy(vregInputPX, vregInputXSqr, AN, mask);
                    Reg::Mul(vregInputPX, vregInputPX, vregInputX, mask);
                    Reg::Exp(vregInputPX, vregInputPX, mask);

                    Reg::Duplicate(vregInputRes0, BETA);
                    Reg::Axpy(vregInputRes0, vregInputXSqr, A3, mask);
                    Reg::Mul(vregInputRes0, vregInputRes0, vregInputX, mask);

                    Reg::Adds(vregInputT, vregInputPX, (float)1.0, mask);
                    Reg::Div(vregInputDiv, vregInputOne, vregInputT, mask);

                    Reg::Mul(vregInputResp, vregInputPX, vregInputDiv, mask);
                    Reg::Mul(vregInputResp, vregInputResp, vregInputRes0, mask);
                    Reg::Mul(vregInputResp, vregInputResp, vregInputDiv, mask);
                    Reg::Compare<T, CMPMODE::EQ>(cmpMask, vregInputResp, vregInputResp, mask);
                    Reg::Select<T>(vregSelect, vregInputResp, vregInputZero, cmpMask);
                    Reg::Add(vregInputResp, vregSelect, vregInputDiv, mask);
                    Reg::Mul(vregOutput, vregInputDy, vregInputResp, mask);

                    // OpCopyOut
                    Reg::StoreAlign((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }
        }
#endif
    }
};

template <typename U>
struct GeluGradV2None16DAG {
    using ONE_OVER_SQRT_TWO = MAKE_CONST(float, 0.707106781); // 1/sqrt(2)
    using OpCopyIn0 = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<float, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<float, U, CAST_MODE_NONE>, OpCopyIn1>;

    using OpCdfErfInput = Bind<Vec::Muls<float>, OpCopyIn1Cast, ONE_OVER_SQRT_TWO>;
    using OpCdfErf = Bind<ErfFast<float>, OpCdfErfInput>;
    using OpErfPost = Bind<GeluGradV2ErfPost<float>, OpCopyIn0Cast, OpCopyIn1Cast, OpCdfErf>;
    using OpResultCast = Bind<Vec::Cast<U, float, CAST_MODE_RINT>, OpErfPost>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct GeluGradV2None32DAG {
    using ONE_OVER_SQRT_TWO = MAKE_CONST(float, 0.707106781); // 1/sqrt(2)
    using OpCopyIn0 = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<float, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<float, U, CAST_MODE_NONE>, OpCopyIn1>;

    using OpCdfErfInput = Bind<Vec::Muls<float>, OpCopyIn1Cast, ONE_OVER_SQRT_TWO>;
    using OpCdfErf = Bind<Vec::Erf<float>, OpCdfErfInput>;
    using OpErfPost = Bind<GeluGradV2ErfPost<float>, OpCopyIn0Cast, OpCopyIn1Cast, OpCdfErf>;
    using OpResultCast = Bind<Vec::Cast<U, float, CAST_MODE_RINT>, OpErfPost>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct GeluGradV2TanhDAG {
    using OpCopyIn0 = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<float, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<float, U, CAST_MODE_NONE>, OpCopyIn1>;

    using OpGeluGradV2Result = Bind<GeluGradV2TanhCustom<float>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpResultCast = Bind<Vec::Cast<U, float, CAST_MODE_RINT>, OpGeluGradV2Result>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace GeluGradV2Op
#endif // CANN_CUSTOM_OPS_GELU_GRAD_V2_DAG_H
