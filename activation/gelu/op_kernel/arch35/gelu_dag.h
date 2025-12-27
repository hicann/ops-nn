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
 * \file gelu_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_GELU_DAG_H
#define CANN_CUSTOM_OPS_GELU_DAG_H
 
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
const float TANH_APPROX_FACTOR = 1 / 0.044715;
const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;
// tbe realization: x / (1 + e^(-1.5957691(x + 0.044715 x * x * x)))
// current realization: x / (1 + e^(-1.5957691*0.044715(x/0.044715 + x^3)))
namespace GeluDag1 {
template<class T>
struct GeluCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline GeluCustom(LocalTensor<T> &dst, LocalTensor<T> &src, uint32_t count) {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputSqr;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputCub;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask;
        if constexpr(std::is_same_v<T, float>) {
            __VEC_SCOPE__ {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    MicroAPI::Mul(vregInputSqr, vregInput, vregInput, mask);
                    MicroAPI::Mul(vregInputCub, vregInputSqr, vregInput, mask);
                    MicroAPI::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
                    MicroAPI::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
                    MicroAPI::Exp(vregInputCub, vregInputCub, mask);
                    MicroAPI::Adds(vregInputCub, vregInputCub, (float)1.0, mask);
                    MicroAPI::Div(vregOutput, vregInput, vregInputCub, mask);
                    
                    // OpCopyOut
                    MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }
        }
#endif
    }
};
} // namespace GeluDag1

template <typename U, typename T = float>
struct GeluDAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    
    using OpLogResult = Bind<GeluDag1::GeluCustom<T>, OpCopyIn0Cast>;
    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpLogResult>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

#endif  // CANN_CUSTOM_OPS_GELU_DAG_H