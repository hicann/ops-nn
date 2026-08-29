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
 * \brief LeakyReLU 算子 DAG 定义及 Reg 自定义 Kernel 实现
 */
#ifndef OPS_NN_ACTIVATION_LEAKY_RELU_OP_KERNEL_ARCH35_LEAKY_RELU_DAG_H
#define OPS_NN_ACTIVATION_LEAKY_RELU_OP_KERNEL_ARCH35_LEAKY_RELU_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LeakyReluOp {
using namespace Ops::Base;
#ifdef __CCE_AICORE__
constexpr static AscendC::Reg::CastTrait castTrait0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_RINT};
#endif

template <class T>
struct LeakyReluCustom : public Vec::ElemwiseBinaryOP<T, T, float> {
    __aicore__ inline LeakyReluCustom(LocalTensor<T>& dst, LocalTensor<T>& src, float negativeSlope, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputfloat;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregNegPart;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregOutput;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregZero;
        Reg::MaskReg mask, cmpMask;

        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                Reg::Duplicate(vregZero, 0.0f);
                mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregInputfloat,
                                                                (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    Reg::Muls(vregNegPart, vregInputfloat, negativeSlope, mask);
                    Reg::Compare<float, CMPMODE::GT>(cmpMask, vregInputfloat, vregZero, mask);
                    Reg::Select<float>(vregOutput, vregInputfloat, vregNegPart, cmpMask);
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize),
                                                                      vregOutput, mask);
                }
            }
        } else {
            Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputT;
            Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutputT;
            __VEC_SCOPE__
            {
                Reg::Duplicate(vregZero, 0.0f);
                mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(vregInputT,
                                                                      (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    Reg::Cast<float, T, castTrait0>(vregInputfloat, vregInputT, mask);
                    Reg::Muls(vregNegPart, vregInputfloat, negativeSlope, mask);
                    Reg::Compare<float, CMPMODE::GT>(cmpMask, vregInputfloat, vregZero, mask);
                    Reg::Select<float>(vregOutput, vregInputfloat, vregNegPart, cmpMask);
                    Reg::Cast<T, float, castTrait1>(vregOutputT, vregOutput, mask);
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize),
                                                                      vregOutputT, mask);
                }
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
} // namespace LeakyReluOp
#endif // OPS_NN_ACTIVATION_LEAKY_RELU_OP_KERNEL_ARCH35_LEAKY_RELU_DAG_H
