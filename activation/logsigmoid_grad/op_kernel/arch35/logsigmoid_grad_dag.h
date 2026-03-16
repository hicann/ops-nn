/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logsigmoid_grad_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_LOGSIGMOID_GRAD_DAG_H
#define CANN_CUSTOM_OPS_LOGSIGMOID_GRAD_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LogSigmoidGradOp {
using namespace AscendC;
using namespace Ops::Base;
#ifdef __CCE_AICORE__
constexpr static MicroAPI::CastTrait castTrait0 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr static MicroAPI::CastTrait castTrait1 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
#endif

template <class T>
struct LogSigmoidGradCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline LogSigmoidGradCustom(
        LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;

        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> DataOne;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> DataZero;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputGradOut;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputSelf;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> SelfAbs;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> SelfAbsNeg;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> SelfAbsNegExp;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> SelfAbsNegExpAdd;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregSelect;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> Answer;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> LastAnswer;
        MicroAPI::MaskReg mask, cmpMask;

        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                MicroAPI::Duplicate(DataOne, (float)1.0);
                MicroAPI::Duplicate(DataZero, (float)0.0);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::LoadAlign(vregInputGradOut, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                    MicroAPI::LoadAlign(vregInputSelf, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                    // compute
                    MicroAPI::Abs(SelfAbs, vregInputSelf, mask);
                    MicroAPI::Muls(SelfAbsNeg, SelfAbs, (float)-1.0, mask);
                    MicroAPI::Exp(SelfAbsNegExp, SelfAbsNeg, mask);
                    MicroAPI::Adds(SelfAbsNegExpAdd, SelfAbsNegExp, (float)1.0, mask);

                    MicroAPI::Compare<float, CMPMODE::LT>(cmpMask, vregInputSelf, DataZero, mask);
                    MicroAPI::Select(vregSelect, DataOne, SelfAbsNegExp, cmpMask);
                    MicroAPI::Div(Answer, vregSelect, SelfAbsNegExpAdd, mask);
                    MicroAPI::Mul(LastAnswer, Answer, vregInputGradOut, mask);

                    // OpCopyOut
                    MicroAPI::StoreAlign((__ubuf__ T*)(dstAddr + loopIdx * vlSize), LastAnswer, mask);
                }
            }
        } else {
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputGradOutT;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInputSelfT;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> LastAnswerT;
            __VEC_SCOPE__
            {
                MicroAPI::Duplicate(DataOne, (float)1.0);
                MicroAPI::Duplicate(DataZero, (float)0.0);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregInputGradOutT, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregInputSelfT, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                    MicroAPI::Cast<float, T, castTrait0>(vregInputGradOut, vregInputGradOutT, mask);
                    MicroAPI::Cast<float, T, castTrait0>(vregInputSelf, vregInputSelfT, mask);
                    // compute
                    MicroAPI::Abs(SelfAbs, vregInputSelf, mask);
                    MicroAPI::Muls(SelfAbsNeg, SelfAbs, (float)-1.0, mask);
                    MicroAPI::Exp(SelfAbsNegExp, SelfAbsNeg, mask);
                    MicroAPI::Adds(SelfAbsNegExpAdd, SelfAbsNegExp, (float)1.0, mask);

                    MicroAPI::Compare<float, CMPMODE::LT>(cmpMask, vregInputSelf, DataZero, mask);
                    MicroAPI::Select(vregSelect, DataOne, SelfAbsNegExp, cmpMask);
                    MicroAPI::Div(Answer, vregSelect, SelfAbsNegExpAdd, mask);
                    MicroAPI::Mul(LastAnswer, Answer, vregInputGradOut, mask);

                    MicroAPI::Cast<T, float, castTrait1>(LastAnswerT, LastAnswer, mask);
                    // OpCopyOut
                    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                        (__ubuf__ T*)(dstAddr + loopIdx * vlSize), LastAnswerT, mask);
                }
            }
        }
#endif
    }
};

template <typename U>
struct LogSigmoidGradDag {
    using grad_out = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyInself = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;

    using Answer = Bind<LogSigmoidGradCustom<U>, grad_out, OpCopyInself>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, Answer>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LogSigmoidGradOp
#endif // LOGSIGMOID_GRAD_DAG_H