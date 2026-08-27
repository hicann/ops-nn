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
 * \file p_relu_dag.h
 * \brief prelu kernel dag
 */

#ifndef OP_NN_ACTIVATION_P_RELU_OP_KERNEL_PRELU_DAG_H
#define OP_NN_ACTIVATION_P_RELU_OP_KERNEL_PRELU_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace PreluOp {
using namespace Ops::Base;

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;

template <class T>
struct PreluCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline PreluCustom(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;
        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputX;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputWeight;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputZero;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputProd;
        Reg::MaskReg mask, cmpMask;
        __VEC_SCOPE__
        {
            Reg::Duplicate(vregInputZero, (T)0.0);
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                // OpCopyIn
                Reg::LoadAlign(vregInputX, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                Reg::LoadAlign(vregInputWeight, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));

                // compute
                Reg::Mul(vregInputProd, vregInputX, vregInputWeight, mask);
                Reg::Compare<T, CMPMODE::GT>(cmpMask, vregInputX, vregInputZero, mask);
                Reg::Select<T>(vregOutput, vregInputX, vregInputProd, cmpMask);

                // OpCopyOut
                Reg::StoreAlign((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};

template <typename U>
struct PreluDAG {
    using OpCopyIn0 = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpPreluResult = Bind<PreluCustom<U>, OpCopyIn0, OpCopyIn1>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpPreluResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace PreluOp

#endif // OP_NN_ACTIVATION_P_RELU_OP_KERNEL_PRELU_DAG_H
