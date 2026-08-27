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
 * \file gelu_v2_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_GELU_DAG_V2_H
#define CANN_CUSTOM_OPS_GELU_DAG_V2_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace GeluV2Op {
using namespace Ops::Base;
const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
const float TANH_APPROX_FACTOR = 1 / 0.044715;
const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;

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
struct GeluV2ErfPost : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline GeluV2ErfPost(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput1;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput2;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputAdds;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputMuls;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput;
        Reg::MaskReg mask;
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::LoadAlign(vregInput1, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                    Reg::LoadAlign(vregInput2, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));
                    Reg::Adds(vregInputAdds, vregInput2, (float)1.0, mask);
                    Reg::Muls(vregInputMuls, vregInput1, (float)0.5, mask);
                    Reg::Mul(vregOutput, vregInputAdds, vregInputMuls, mask);
                    // OpCopyOut
                    Reg::StoreAlign((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }
        }
#endif
    }
};

template <class T>
struct GeluV2Tanh : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline GeluV2Tanh(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputSqr;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputCub;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput;
        Reg::MaskReg mask;
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::LoadAlign(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    Reg::Mul(vregInputSqr, vregInput, vregInput, mask);
                    Reg::Mul(vregInputCub, vregInputSqr, vregInput, mask);
                    Reg::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
                    Reg::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
                    Reg::Exp(vregInputCub, vregInputCub, mask);
                    Reg::Adds(vregInputCub, vregInputCub, (float)1.0, mask);
                    Reg::Div(vregOutput, vregInput, vregInputCub, mask);

                    // OpCopyOut
                    Reg::StoreAlign((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }
        }
#endif
    }
};

template <typename U, typename T = float>
struct GeluV2Erf16BDag {
    using ONE_OVER_SQRT_TWO = MAKE_CONST(float, 0.707106781); // 1/sqrt(2)

    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;

    using OpMulX = Bind<Vec::Muls<T>, OpCopyIn0Cast, ONE_OVER_SQRT_TWO>;
    using OpErf = Bind<ErfFast<T>, OpMulX>;
    using OpErfPost = Bind<GeluV2ErfPost<T>, OpCopyIn0Cast, OpErf>;

    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpErfPost>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct GeluV2Erf32BDag {
    using ONE_OVER_SQRT_TWO = MAKE_CONST(float, 0.707106781); // 1/sqrt(2)

    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;

    using OpMulX = Bind<Vec::Muls<T>, OpCopyIn0, ONE_OVER_SQRT_TWO>;
    using OpErf = Bind<Vec::Erf<T>, OpMulX>;
    using OpErfPost = Bind<GeluV2ErfPost<T>, OpCopyIn0, OpErf>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpErfPost>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct GeluV2TanhDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;

    using OpLogResult = Bind<GeluV2Tanh<T>, OpCopyIn0Cast>;
    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpLogResult>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace GeluV2Op

#endif // CANN_CUSTOM_OPS_GELU_V2_DAG_H
