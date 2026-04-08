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
 * \file sigmoid_cross_entropy_with_logits_dag.h
 * \brief sigmoid_cross_entropy_with_logits_dag head file
 */
#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_DAG_H_
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_DAG_H_

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#endif

namespace AscendC {
using namespace Ops::Base;
using namespace AscendC;

#ifdef __CCE_AICORE__
using AscendC::MicroAPI::RegTensor;

constexpr static uint16_t VECTOR_LENGTH = platform::GetVRegSize();

#endif

namespace SigmoidCrossEntropyWithLogitsDag1 {
template<class T>
struct CalcSigmoidCrossEntropyWithLogits : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline CalcSigmoidCrossEntropyWithLogits(LocalTensor<T> &dst, LocalTensor<T> &src0, LocalTensor<T> &src1, uint32_t count) {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vlSize = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vlSize);
        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregPredict;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregTarget;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregMaxPredict;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregAbsPredict;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregNegAbsPredict;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregExpNegAbs;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOneAddExp;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregLog;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregMulPredictTarget;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask;

        __VEC_SCOPE__ {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                
                MicroAPI::DataCopy(vregPredict, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                MicroAPI::DataCopy(vregTarget, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                
                MicroAPI::Duplicate(vregOneAddExp, (T)1.0f, mask);
                
                MicroAPI::Maxs(vregMaxPredict, vregPredict, (T)0.0f, mask);
                
                MicroAPI::Abs(vregAbsPredict, vregPredict, mask);
                
                MicroAPI::Neg(vregNegAbsPredict, vregAbsPredict, mask);
                
                MicroAPI::Exp(vregExpNegAbs, vregNegAbsPredict, mask);
                
                MicroAPI::Add(vregOneAddExp, vregExpNegAbs, vregOneAddExp, mask);
                
                MicroAPI::Log(vregLog, vregOneAddExp, mask);
                
                MicroAPI::Mul(vregMulPredictTarget, vregPredict, vregTarget, mask);
                
                MicroAPI::Sub(vregOutput, vregMaxPredict, vregMulPredictTarget, mask);
                MicroAPI::Add(vregOutput, vregOutput, vregLog, mask);
                
                MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};
} // namespace SigmoidCrossEntropyWithLogitsDag1

} // namespace AscendC

namespace SigmoidCrossEntropyWithLogitsOp {
using namespace AscendC;
using namespace Ops::Base;

template <typename T>
struct SigmoidCrossEntropyWithLogitsDagNoCast {
    using OpCopyInPredict = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyInTarget = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;

    using OpLoss = Bind<SigmoidCrossEntropyWithLogitsDag1::CalcSigmoidCrossEntropyWithLogits<T>, OpCopyInPredict, OpCopyInTarget>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpLoss>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct SigmoidCrossEntropyWithLogitsDagWithCast {
    constexpr static int CAST_MODE_NONE = 0;
    constexpr static int CAST_MODE_RINT = 1;
    
    using OpCopyInPredict = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyInTarget = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;

    using OpCopyInPredictCast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyInPredict>;
    using OpCopyInTargetCast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyInTarget>;

    using OpLoss = Bind<SigmoidCrossEntropyWithLogitsDag1::CalcSigmoidCrossEntropyWithLogits<T>, OpCopyInPredictCast, OpCopyInTargetCast>;

    using OpResCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpLoss>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

}  // namespace SigmoidCrossEntropyWithLogitsOp

#endif  // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_DAG_H_
