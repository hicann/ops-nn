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
 * \file gelu_grad_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_GELU_GRAD_DAG_H
#define CANN_CUSTOM_OPS_GELU_GRAD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
namespace GeluGradOp {
using namespace AscendC;

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;
namespace GeluGradDag1 {
constexpr float BETAN = -1.595769121605730711759f;
constexpr float AN = -0.0713548162726002527220f;
constexpr float A3 = 0.2140644488178007f;
constexpr float BETA = 1.595769121605730711759f;

template <class T>
struct GeluGradCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline GeluGradCustom(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;
        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputDy;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputX;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputXSqr;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputPX;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputRes0;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputT;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputDiv;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputOne;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputZero;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputResp;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregSelect;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask, cmpMask;

        static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
            AscendC::MicroAPI::MaskMergeMode::ZEROING,
            false,
        };
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                MicroAPI::Duplicate(vregInputOne, (float)1.0);
                MicroAPI::Duplicate(vregInputZero, (float)0.0);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                    MicroAPI::Duplicate(vregInputPX, BETAN);
                    // OpCopyIn
                    MicroAPI::DataCopy(vregInputDy, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                    MicroAPI::DataCopy(vregInputX, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                    // compute
                    MicroAPI::Mul(vregInputXSqr, vregInputX, vregInputX, mask);
                    MicroAPI::Axpy(vregInputPX, vregInputXSqr, AN, mask);
                    MicroAPI::Mul(vregInputPX, vregInputPX, vregInputX, mask);
                    MicroAPI::Exp(vregInputPX, vregInputPX, mask);

                    MicroAPI::Duplicate(vregInputRes0, BETA);
                    MicroAPI::Axpy(vregInputRes0, vregInputXSqr, A3, mask);
                    MicroAPI::Mul(vregInputRes0, vregInputRes0, vregInputX, mask);

                    MicroAPI::Adds(vregInputT, vregInputPX, (float)1.0, mask);
                    MicroAPI::Div(vregInputDiv, vregInputOne, vregInputT, mask);

                    MicroAPI::Mul(vregInputResp, vregInputPX, vregInputDiv, mask);
                    MicroAPI::Mul(vregInputResp, vregInputResp, vregInputRes0, mask);
                    MicroAPI::Mul(vregInputResp, vregInputResp, vregInputDiv, mask);
                    MicroAPI::Compare<T, CMPMODE::EQ>(cmpMask, vregInputResp, vregInputResp, mask);
                    MicroAPI::Select<T>(vregSelect, vregInputResp, vregInputZero, cmpMask);
                    MicroAPI::Add(vregInputResp, vregSelect, vregInputDiv, mask);
                    MicroAPI::Mul(vregOutput, vregInputDy, vregInputResp, mask);

                    // OpCopyOut
                    MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }
        }
#endif
    }
};
} // namespace GeluGradDag1

template <typename U>
struct GeluGradDAG {
    using OpCopyIn0 = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<float, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpCopyIn1Cast = Bind<Vec::Cast<float, U, CAST_MODE_NONE>, OpCopyIn1>;

    using OpGeluGradResult = Bind<GeluGradDag1::GeluGradCustom<float>, OpCopyIn0Cast, OpCopyIn1Cast>;
    using OpResultCast = Bind<Vec::Cast<U, float, CAST_MODE_RINT>, OpGeluGradResult>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace GeluGradOp

#endif // CANN_CUSTOM_OPS_GELU_GRAD_DAG_H