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
constexpr static Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::UNKNOWN};
constexpr static Reg::CastTrait castTrait1 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::CAST_RINT};
#endif

template <class T>
struct LogSigmoidGradCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline LogSigmoidGradCustom(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1,
                                           uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;

        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<float, Reg::RegTraitNumOne> DataOne;
        Reg::RegTensor<float, Reg::RegTraitNumOne> DataZero;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputGradOut;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputSelf;
        Reg::RegTensor<float, Reg::RegTraitNumOne> SelfAbs;
        Reg::RegTensor<float, Reg::RegTraitNumOne> SelfAbsNeg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> SelfAbsNegExp;
        Reg::RegTensor<float, Reg::RegTraitNumOne> SelfAbsNegExpAdd;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregSelect;
        Reg::RegTensor<float, Reg::RegTraitNumOne> Answer;
        Reg::RegTensor<float, Reg::RegTraitNumOne> LastAnswer;
        Reg::MaskReg mask, cmpMask;

        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                Reg::Duplicate(DataOne, (float)1.0);
                Reg::Duplicate(DataZero, (float)0.0);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::LoadAlign(vregInputGradOut, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                    Reg::LoadAlign(vregInputSelf, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                    // compute
                    Reg::Abs(SelfAbs, vregInputSelf, mask);
                    Reg::Muls(SelfAbsNeg, SelfAbs, (float)-1.0, mask);
                    Reg::Exp(SelfAbsNegExp, SelfAbsNeg, mask);
                    Reg::Adds(SelfAbsNegExpAdd, SelfAbsNegExp, (float)1.0, mask);

                    Reg::Compare<float, CMPMODE::LT>(cmpMask, vregInputSelf, DataZero, mask);
                    Reg::Select(vregSelect, DataOne, SelfAbsNegExp, cmpMask);
                    Reg::Div(Answer, vregSelect, SelfAbsNegExpAdd, mask);
                    Reg::Mul(LastAnswer, Answer, vregInputGradOut, mask);

                    // OpCopyOut
                    Reg::StoreAlign((__ubuf__ T*)(dstAddr + loopIdx * vlSize), LastAnswer, mask);
                }
            }
        } else {
            Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputGradOutT;
            Reg::RegTensor<T, Reg::RegTraitNumOne> vregInputSelfT;
            Reg::RegTensor<T, Reg::RegTraitNumOne> LastAnswerT;
            __VEC_SCOPE__
            {
                Reg::Duplicate(DataOne, (float)1.0);
                Reg::Duplicate(DataZero, (float)0.0);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(vregInputGradOutT,
                                                                      (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(vregInputSelfT,
                                                                      (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                    Reg::Cast<float, T, castTrait0>(vregInputGradOut, vregInputGradOutT, mask);
                    Reg::Cast<float, T, castTrait0>(vregInputSelf, vregInputSelfT, mask);
                    // compute
                    Reg::Abs(SelfAbs, vregInputSelf, mask);
                    Reg::Muls(SelfAbsNeg, SelfAbs, (float)-1.0, mask);
                    Reg::Exp(SelfAbsNegExp, SelfAbsNeg, mask);
                    Reg::Adds(SelfAbsNegExpAdd, SelfAbsNegExp, (float)1.0, mask);

                    Reg::Compare<float, CMPMODE::LT>(cmpMask, vregInputSelf, DataZero, mask);
                    Reg::Select(vregSelect, DataOne, SelfAbsNegExp, cmpMask);
                    Reg::Div(Answer, vregSelect, SelfAbsNegExpAdd, mask);
                    Reg::Mul(LastAnswer, Answer, vregInputGradOut, mask);

                    Reg::Cast<T, float, castTrait1>(LastAnswerT, LastAnswer, mask);
                    // OpCopyOut
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize),
                                                                      LastAnswerT, mask);
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
