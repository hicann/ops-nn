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
 * \file swish_grad_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_SWISHGRAD_DAG_H
#define CANN_CUSTOM_OPS_SWISHGRAD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace AscendC;
using namespace Ops::Base;

const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;
// realization: swish_grad = x1*(sigmoid[scale * x ] * (1 + scale * x * (1-sigmoid[ scale * x ])))
#ifdef __CCE_AICORE__
constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::MicroAPI::CastTrait castTrait1 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#endif
namespace SwishGradDag1 {
template <class T>
struct SwishGradCustom : public Ops::Base::Vec::ElemwiseTernaryOP<T, T, T, float> {
    __aicore__ inline SwishGradCustom(
        LocalTensor<T>& dst, LocalTensor<T>& src, LocalTensor<T>& src2, float scale1, uint32_t count)
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
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputMid;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregValue1;
        MicroAPI::MaskReg mask;
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                MicroAPI::Duplicate(vregValue1, (float)1.0);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    MicroAPI::DataCopy(vregInput2, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));
                    MicroAPI::Muls(vregInput, vregInput, scale1, mask);
                    MicroAPI::Neg(vregInputMid, vregInput, mask);
                    MicroAPI::Exp(vregInputMid, vregInputMid, mask);
                    MicroAPI::Adds(vregInputMid, vregInputMid, (float)1.0, mask);
                    MicroAPI::Div(vregInputMid, vregValue1, vregInputMid, mask);

                    MicroAPI::Sub(vregOutput, vregValue1, vregInputMid, mask);
                    MicroAPI::Mul(vregOutput, vregOutput, vregInput, mask);
                    MicroAPI::FusedMulDstAdd(vregOutput, vregInputMid, vregInputMid, mask);
                    MicroAPI::Mul(vregOutput, vregOutput, vregInput2, mask);

                    // OpCopyOut
                    MicroAPI::DataCopy(
                        (__ubuf__ T*)(dstAddr + loopIdx * vlSize), (MicroAPI::RegTensor<T>&)vregOutput, mask);
                }
            }
        } else {
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput16;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput162;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput16;
            __VEC_SCOPE__
            {
                MicroAPI::Duplicate(vregValue1, (float)1.0);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregInput16, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregInput162, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));
                    MicroAPI::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);
                    MicroAPI::Cast<float, T, castTrait0>(vregInput2, vregInput162, mask);

                    MicroAPI::Muls(vregInput, vregInput, scale1, mask);
                    MicroAPI::Neg(vregInputMid, vregInput, mask);
                    MicroAPI::Exp(vregInputMid, vregInputMid, mask);
                    MicroAPI::Adds(vregInputMid, vregInputMid, (float)1.0, mask);
                    MicroAPI::Div(vregInputMid, vregValue1, vregInputMid, mask);

                    MicroAPI::Sub(vregOutput, vregValue1, vregInputMid, mask);
                    MicroAPI::Mul(vregOutput, vregOutput, vregInput, mask);
                    MicroAPI::FusedMulDstAdd(vregOutput, vregInputMid, vregInputMid, mask);
                    MicroAPI::Mul(vregOutput, vregOutput, vregInput2, mask);

                    MicroAPI::Cast<T, float, castTrait1>(vregOutput16, vregOutput, mask);
                    // OpCopyOut
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                        (__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput16, mask);
                }
            }
        }
#endif
    }
};
} // namespace SwishGradDag1

template <typename U>
struct SwishGradDAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpResult1 = Bind<SwishGradDag1::SwishGradCustom<U>, OpCopyIn1, OpCopyIn0, Placeholder::Var<float, 0>>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResult1>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

#endif // CANN_CUSTOM_OPS_SWISHGRAD_DAG_H