/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sigmoid_cross_entropy_with_logits_v2_dag.h
 * \brief sigmoid_cross_entropy_with_logits_v2_dag head file
 */
#ifndef ASCENDC_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_DAG_H_
#define ASCENDC_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_DAG_H_

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
using AscendC::Reg::RegTensor;

constexpr static uint16_t VECTOR_LENGTH = platform::GetVRegSize();

#endif
template <typename T = float, bool HAS_WEIGHT = false, bool HAS_POS_WEIGHT = false>
struct CalcBCEWithLogitsV2 : public Vec::ElemwiseQuaternaryOP<T, T, T, T, T> {
    __aicore__ inline CalcBCEWithLogitsV2(LocalTensor<T>& loss, LocalTensor<T>& predict, LocalTensor<T>& target,
                                          LocalTensor<T>& weight, LocalTensor<T>& pos_weight, int32_t count)
    {
#ifdef __CCE_AICORE__

        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(T);
        uint32_t totalLen = count;
        uint32_t repeatTimes = ops::CeilDiv<uint32_t>(totalLen, oneRepeat);

        __ubuf__ T* xAddr = (__ubuf__ T*)predict.GetPhyAddr();
        __ubuf__ T* yAddr = (__ubuf__ T*)target.GetPhyAddr();
        __ubuf__ T* weightAddr = HAS_WEIGHT ? (__ubuf__ T*)weight.GetPhyAddr() : nullptr;
        __ubuf__ T* posWeightAddr = HAS_POS_WEIGHT ? (__ubuf__ T*)pos_weight.GetPhyAddr() : nullptr;
        __ubuf__ T* lossAddr = (__ubuf__ T*)loss.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg pregUp;
            Reg::RegTensor<T> regX;
            Reg::RegTensor<T> regY;
            Reg::RegTensor<T> regWeight;
            Reg::RegTensor<T> regPosWeight;
            Reg::RegTensor<T> regLoss;

            Reg::RegTensor<T> regMinVal;
            Reg::RegTensor<T> regAbsX;
            Reg::RegTensor<T> regNegAbsX;
            Reg::RegTensor<T> regExpNegAbsX;
            Reg::RegTensor<T> regOneAddExp;
            Reg::RegTensor<T> regLog;
            Reg::RegTensor<T> regLogSigmoid;
            Reg::RegTensor<T> regPWSubOne;
            Reg::RegTensor<T> regPWSubOneMulY;
            Reg::RegTensor<T> regPWSubOneMulYAddOne;
            Reg::RegTensor<T> regOneSubY;
            Reg::RegTensor<T> regOneSubYMulX;

            Reg::RegTensor<T> regOne;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = Reg::UpdateMask<T>(totalLen);
                AscendC::Reg::Duplicate(regOne, (T)1.0f, pregUp);
                Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regX, xAddr, (int32_t)oneRepeat);
                Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regY, yAddr, (int32_t)oneRepeat);
                if constexpr (HAS_WEIGHT) {
                    Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regWeight, weightAddr, (int32_t)oneRepeat);
                }
                if constexpr (HAS_POS_WEIGHT) {
                    Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regPosWeight, posWeightAddr,
                                                                          (int32_t)oneRepeat);
                }

                Reg::Mins(regMinVal, regX, (T)0.0f, pregUp);
                Reg::Abs(regAbsX, regX, pregUp);
                Reg::Neg(regNegAbsX, regAbsX, pregUp);
                Reg::Exp(regExpNegAbsX, regNegAbsX, pregUp);
                Reg::Adds(regOneAddExp, regExpNegAbsX, (T)1.0f, pregUp);
                Reg::Log(regLog, regOneAddExp, pregUp);

                Reg::Sub(regLogSigmoid, regMinVal, regLog, pregUp);

                if constexpr (HAS_POS_WEIGHT) {
                    Reg::Sub(regPWSubOne, regPosWeight, regOne, pregUp);
                    Reg::Mul(regPWSubOneMulY, regPWSubOne, regY, pregUp);
                    Reg::Adds(regPWSubOneMulYAddOne, regPWSubOneMulY, (T)1.0f, pregUp);
                    Reg::Mul(regLogSigmoid, regLogSigmoid, regPWSubOneMulYAddOne, pregUp);
                }

                Reg::Sub(regOneSubY, regOne, regY, pregUp);
                Reg::Mul(regOneSubYMulX, regOneSubY, regX, pregUp);
                Reg::Sub(regLoss, regOneSubYMulX, regLogSigmoid, pregUp);

                if constexpr (HAS_WEIGHT) {
                    Reg::Mul(regLoss, regLoss, regWeight, pregUp);
                }

                Reg::StoreAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(lossAddr, regLoss, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};
} // namespace AscendC

namespace SigmoidCrossEntropyWithLogitsV2 {
using namespace AscendC;
using namespace Ops::Base;

const bool HAS_WEIGHT = true;
const bool NO_WEIGHT = false;

template <typename U, typename R, typename T = float>
struct SigmoidCEWithLogitsV2HasTwoWeight {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyWeight = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyPosWeight = Bind<Vec::CopyInBrc<U>, Placeholder::In3<U>>;

    using OpCopyXCast = Bind<Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Bind<Vec::Cast<T, U, 0>, OpCopyY>;
    using OpCopyWeightCast = Bind<Vec::Cast<T, U, 0>, OpCopyWeight>;
    using OpCopyPosWeightCast = Bind<Vec::Cast<T, U, 0>, OpCopyPosWeight>;

    using OpLoss = Bind<CalcBCEWithLogitsV2<T, true, true>, OpCopyXCast, OpCopyYCast, OpCopyWeightCast,
                        OpCopyPosWeightCast>;

    using OpRes = Bind<Vec::Cast<R, T, CAST_MODE_RINT>, OpLoss>;

    using OpCopyOut = Bind<Vec::CopyOut<R>, Placeholder::Out0<R>, OpRes>;
    using Outputs = Elems<OpCopyOut>;

    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename R, typename T = float>
struct SigmoidCEWithLogitsV2WeightOnly {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyWeight = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;

    using OpCopyXCast = Bind<Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Bind<Vec::Cast<T, U, 0>, OpCopyY>;
    using OpCopyWeightCast = Bind<Vec::Cast<T, U, 0>, OpCopyWeight>;
    // X冗余传入作为占位符
    using OpLoss = Bind<CalcBCEWithLogitsV2<T, true, false>, OpCopyXCast, OpCopyYCast, OpCopyWeightCast, OpCopyXCast>;

    using OpRes = Bind<Vec::Cast<R, T, CAST_MODE_RINT>, OpLoss>;

    using OpCopyOut = Bind<Vec::CopyOut<R>, Placeholder::Out0<R>, OpRes>;
    using Outputs = Elems<OpCopyOut>;

    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename R, typename T = float>
struct SigmoidCEWithLogitsV2PosWeightOnly {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyPosWeight = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;

    using OpCopyXCast = Bind<Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Bind<Vec::Cast<T, U, 0>, OpCopyY>;
    using OpCopyPosWeightCast = Bind<Vec::Cast<T, U, 0>, OpCopyPosWeight>;

    using OpLoss = Bind<CalcBCEWithLogitsV2<T, false, true>, OpCopyXCast, OpCopyYCast, OpCopyXCast,
                        OpCopyPosWeightCast>;

    using OpRes = Bind<Vec::Cast<R, T, CAST_MODE_RINT>, OpLoss>;

    using OpCopyOut = Bind<Vec::CopyOut<R>, Placeholder::Out0<R>, OpRes>;
    using Outputs = Elems<OpCopyOut>;

    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename R, typename T = float>
struct SigmoidCEWithLogitsV2 {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;

    using OpCopyXCast = Bind<Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Bind<Vec::Cast<T, U, 0>, OpCopyY>;

    using OpLoss = Bind<CalcBCEWithLogitsV2<T, false, false>, OpCopyXCast, OpCopyYCast, OpCopyXCast, OpCopyXCast>;

    using OpRes = Bind<Vec::Cast<R, T, CAST_MODE_RINT>, OpLoss>;

    using OpCopyOut = Bind<Vec::CopyOut<R>, Placeholder::Out0<R>, OpRes>;
    using Outputs = Elems<OpCopyOut>;

    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace SigmoidCrossEntropyWithLogitsV2
#endif // ASCENDC_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_DAG_H_
