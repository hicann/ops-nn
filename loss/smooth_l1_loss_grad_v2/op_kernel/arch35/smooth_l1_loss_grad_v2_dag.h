/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file smooth_l1_loss_grad_v2_dag.h
 * \brief RegBase custom VF node for smooth_l1_loss_grad_v2.
 *        Fuses Sub, Abs, Compare+Select×3, Add×2, Mul(dout), Muls(norm) into a single VF_CALL node.
 *        Formula (matches original Compare+Select chain exactly):
 *          diff = predict - label
 *          select_pos  = (diff >= sigma)  ? 1.0  : 0.0
 *          select_neg  = (diff <= -sigma) ? -1.0 : 0.0
 *          select_inner = (|diff| >= sigma) ? 0.0 : diff * invertSigma
 *          gradient    = norm * dout * (select_pos + select_neg + select_inner)
 */

#ifndef CANN_CUSTOM_OPS_SMOOTH_L1_LOSS_GRAD_V2_DAG_H
#define CANN_CUSTOM_OPS_SMOOTH_L1_LOSS_GRAD_V2_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#endif

// ── Scalar indices (must match tiling SetScalar order) ──
constexpr int SCALAR_SIGMA_IDX = 0;
constexpr int SCALAR_NEG_SIGMA_IDX = 1;
constexpr int SCALAR_INVERT_SIGMA_IDX = 2;
constexpr int SCALAR_NORM_IDX = 3;

namespace SmoothL1LossGradV2Op {
using namespace Ops::Base;

#ifdef __CCE_AICORE__
using namespace AscendC;
constexpr static uint32_t SL1_VEC_LEN = AscendC::GetVecLen();

constexpr static Reg::CastTrait SL1_CAST_B16_TO_B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr static Reg::CastTrait SL1_CAST_B32_TO_B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T, typename PromteT>
__aicore__ inline void CalcSmoothL1LossGradV2VF(__ubuf__ T* predictAddr, __ubuf__ T* labelAddr, __ubuf__ T* doutAddr,
                                                __ubuf__ T* outAddr, PromteT sigma, PromteT negSigma,
                                                PromteT invertSigma, PromteT norm, uint16_t repeatTimes,
                                                uint32_t oneRepeat, uint32_t totalLen)
{
    constexpr PromteT oneFloat = static_cast<PromteT>(1.0f);
    constexpr PromteT negOneFloat = static_cast<PromteT>(-1.0f);
    constexpr PromteT zeroFloat = static_cast<PromteT>(0.0f);

    Reg::MaskReg pregUp;
    Reg::MaskReg maskHigh;
    Reg::MaskReg maskLow;
    Reg::MaskReg maskAbs;
    Reg::RegTensor<PromteT> regPredict;
    Reg::RegTensor<PromteT> regLabel;
    Reg::RegTensor<PromteT> regAbs;
    Reg::RegTensor<PromteT> regResult;
    Reg::RegTensor<PromteT> regDout;
    Reg::RegTensor<PromteT> regTmp;
    Reg::RegTensor<PromteT> regZero;
    Reg::RegTensor<PromteT> regOne;
    Reg::RegTensor<PromteT> regNegOne;

    for (uint16_t vfRepeat = 0; vfRepeat < repeatTimes; vfRepeat++) {
        pregUp = Reg::UpdateMask<PromteT>(totalLen);

        // Load predict and label (raw type T, cast to PromteT if needed)
        // regPredict holds predict, regLabel holds label
        if constexpr (IsSameType<PromteT, float>::value && !IsSameType<T, PromteT>::value) {
            Reg::RegTensor<T> regCopyIn;
            Reg::RegTensor<T> regUnpack;
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regCopyIn, predictAddr, (int32_t)oneRepeat);
            Reg::UnPack((Reg::RegTensor<int32_t>&)regUnpack, (Reg::RegTensor<int16_t>&)regCopyIn);
            Reg::Cast<PromteT, T, SL1_CAST_B16_TO_B32>(regPredict, regUnpack, pregUp);
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regCopyIn, labelAddr, (int32_t)oneRepeat);
            Reg::UnPack((Reg::RegTensor<int32_t>&)regUnpack, (Reg::RegTensor<int16_t>&)regCopyIn);
            Reg::Cast<PromteT, T, SL1_CAST_B16_TO_B32>(regLabel, regUnpack, pregUp);
        } else {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regPredict, predictAddr, (int32_t)oneRepeat);
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regLabel, labelAddr, (int32_t)oneRepeat);
        }

        // diff = predict - label, regPredict becomes diff
        Reg::Sub(regPredict, regPredict, regLabel, pregUp);

        // |diff| in regAbs, diff * invertSigma in regLabel (reuse label's register)
        Reg::Abs(regAbs, regPredict, pregUp);
        Reg::Muls(regLabel, regPredict, invertSigma, pregUp);

        // Initialize constants
        Reg::Duplicate(regZero, zeroFloat, pregUp);
        Reg::Duplicate(regOne, oneFloat, pregUp);
        Reg::Duplicate(regNegOne, negOneFloat, pregUp);

        // select_pos = (diff >= sigma) ? 1.0 : 0.0 → regResult
        Reg::Compares<PromteT, CMPMODE::GE>(maskHigh, regPredict, sigma, pregUp);
        Reg::Select<PromteT>(regResult, regOne, regZero, maskHigh);

        // select_neg = (diff <= -sigma) ? -1.0 : 0.0 → regTmp
        Reg::Compares<PromteT, CMPMODE::LE>(maskLow, regPredict, negSigma, pregUp);
        Reg::Select<PromteT>(regTmp, regNegOne, regZero, maskLow);

        // regResult = select_pos + select_neg
        Reg::Add(regResult, regResult, regTmp, pregUp);

        // select_inner = (|diff| >= sigma) ? 0.0 : (diff * invertSigma) → regTmp
        Reg::Compares<PromteT, CMPMODE::GE>(maskAbs, regAbs, sigma, pregUp);
        Reg::Select<PromteT>(regTmp, regZero, regLabel, maskAbs);

        // regResult = (select_pos + select_neg) + select_inner
        Reg::Add(regResult, regResult, regTmp, pregUp);

        // Load dout (raw type T, cast to PromteT if needed)
        if constexpr (IsSameType<PromteT, float>::value && !IsSameType<T, PromteT>::value) {
            Reg::RegTensor<T> regCopyIn;
            Reg::RegTensor<T> regUnpack;
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regCopyIn, doutAddr, (int32_t)oneRepeat);
            Reg::UnPack((Reg::RegTensor<int32_t>&)regUnpack, (Reg::RegTensor<int16_t>&)regCopyIn);
            Reg::Cast<PromteT, T, SL1_CAST_B16_TO_B32>(regDout, regUnpack, pregUp);
        } else {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(regDout, doutAddr, (int32_t)oneRepeat);
        }

        // result = result * dout * norm
        Reg::Mul(regResult, regResult, regDout, pregUp);
        Reg::Muls(regResult, regResult, norm, pregUp);

        // Store: if T != PromteT, Cast+Pack back to bfloat16/half
        if constexpr (IsSameType<PromteT, float>::value && !IsSameType<T, PromteT>::value) {
            Reg::RegTensor<T> regTmp;
            Reg::RegTensor<T> regCopyOut;
            Reg::MaskReg pregT;
            Reg::Cast<T, PromteT, SL1_CAST_B32_TO_B16>(regTmp, regResult, pregUp);
            Reg::Pack((Reg::RegTensor<uint16_t>&)regCopyOut, (Reg::RegTensor<uint32_t>&)regTmp);
            Reg::Pack(pregT, pregUp);
            Reg::StoreAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(outAddr, regCopyOut, (int32_t)oneRepeat, pregT);
        } else {
            Reg::StoreAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(outAddr, regResult, (int32_t)oneRepeat, pregUp);
        }
    }
}
#endif

// Custom op: VF_CALL dispatches to Compare+Select VF kernel
template <typename T, typename PromteT = float>
struct CalcSmoothL1LossGradV2Op : public Vec::Elemwise7OP<T, T, T, T, PromteT, PromteT, PromteT, PromteT> {
    __aicore__ inline CalcSmoothL1LossGradV2Op(LocalTensor<T>& result, LocalTensor<T>& predict, LocalTensor<T>& label,
                                               LocalTensor<T>& dout, PromteT& sigma, PromteT& negSigma,
                                               PromteT& invertSigma, PromteT& norm, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = SL1_VEC_LEN / sizeof(PromteT);
        uint32_t totalLen = count;
        uint16_t repeatTimes = (uint16_t)CeilDiv<uint32_t>(totalLen, oneRepeat);

        VF_CALL<CalcSmoothL1LossGradV2VF<T, PromteT>>(
            (__ubuf__ T*)predict.GetPhyAddr(), (__ubuf__ T*)label.GetPhyAddr(), (__ubuf__ T*)dout.GetPhyAddr(),
            (__ubuf__ T*)result.GetPhyAddr(), sigma, negSigma, invertSigma, norm, repeatTimes, oneRepeat, totalLen);
#endif
    }
};

// ── Tensor dout DAG ──
// 5 nodes: 3 CopyInBrc + CalcSmoothL1LossGradV2Op + CopyOut
template <typename T, typename PromteT = float>
struct SmoothL1LossGradV2OpDag {
    using OpPredict = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpLabel = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpDout = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;

    using OpCalc = Bind<CalcSmoothL1LossGradV2Op<T, PromteT>, OpPredict, OpLabel, OpDout,
                        Placeholder::Var<PromteT, SCALAR_SIGMA_IDX>, Placeholder::Var<PromteT, SCALAR_NEG_SIGMA_IDX>,
                        Placeholder::Var<PromteT, SCALAR_INVERT_SIGMA_IDX>, Placeholder::Var<PromteT, SCALAR_NORM_IDX>>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCalc>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ── Scalar dout DAG ──
// 5 nodes: 2 CopyInBrc + Duplicate + CalcSmoothL1LossGradV2Op + CopyOut
template <typename T, typename PromteT = float>
struct SmoothL1LossGradV2OpScalarDag {
    using OpPredict = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpLabel = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpDout = Bind<Vec::Duplicate<T>, Placeholder::In2<T, Placeholder::ScalarAttr<true>>>;

    using OpCalc = Bind<CalcSmoothL1LossGradV2Op<T, PromteT>, OpPredict, OpLabel, OpDout,
                        Placeholder::Var<PromteT, SCALAR_SIGMA_IDX>, Placeholder::Var<PromteT, SCALAR_NEG_SIGMA_IDX>,
                        Placeholder::Var<PromteT, SCALAR_INVERT_SIGMA_IDX>, Placeholder::Var<PromteT, SCALAR_NORM_IDX>>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCalc>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace SmoothL1LossGradV2Op

#endif // CANN_CUSTOM_OPS_SMOOTH_L1_LOSS_GRAD_V2_DAG_H
