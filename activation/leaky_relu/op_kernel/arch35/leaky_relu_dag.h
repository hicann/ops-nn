/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file leaky_relu_dag.h
 * \brief LeakyReLU 算子 DAG 定义及 MicroAPI 自定义 Kernel 实现
 */
#ifndef OPS_NN_ACTIVATION_LEAKY_RELU_OP_KERNEL_ARCH35_LEAKY_RELU_DAG_H
#define OPS_NN_ACTIVATION_LEAKY_RELU_OP_KERNEL_ARCH35_LEAKY_RELU_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LeakyReluOp {
using namespace Ops::Base;
template <class T>
struct LeakyReluCustom : public Vec::ElemwiseBinaryOP<T, T, float> {
    __aicore__ inline LeakyReluCustom(LocalTensor<T>& dst, LocalTensor<T>& src, float negativeSlope, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregNegPart;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregZero;
        MicroAPI::MaskReg mask, cmpMask;

        __VEC_SCOPE__
        {
            MicroAPI::Duplicate(vregZero, (T)0.0);
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));

                MicroAPI::Muls(vregNegPart, vregInput, negativeSlope, mask);
                MicroAPI::Compare<T, CMPMODE::GT>(cmpMask, vregInput, vregZero, mask);
                MicroAPI::Select<T>(vregOutput, vregInput, vregNegPart, cmpMask);

                MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};

template <typename U, typename T = float>
struct LeakyReluDag {
    using OpCopyInX = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpLeakyRelu = Bind<LeakyReluCustom<U>, OpCopyInX, Placeholder::Var<T, 0>>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpLeakyRelu>;
    using Outputs = Elems<OpCopyOut>;
    using OpDag = DAGSch<Outputs>;
};

template <typename U, typename T = float>
struct LeakyReluCastDag {
    using OpCopyInX = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyInXCast = Bind<Vec::Cast<T, U, 0>, OpCopyInX>;
    using OpLeakyRelu = Bind<LeakyReluCustom<T>, OpCopyInXCast, Placeholder::Var<T, 0>>;
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyOutCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpLeakyRelu>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyOutCast>;
    using Outputs = Elems<OpCopyOut>;
    using OpDag = DAGSch<Outputs>;
};
} // namespace LeakyReluOp
#endif // OPS_NN_ACTIVATION_LEAKY_RELU_OP_KERNEL_ARCH35_LEAKY_RELU_DAG_H
