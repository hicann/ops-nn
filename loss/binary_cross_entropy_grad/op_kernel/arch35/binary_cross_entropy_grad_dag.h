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
 * \file binary_cross_entropy_grad_dag.h
 * \brief RegBase custom VF node for binary_cross_entropy_grad.
 *        Fuses Cast+Sub+Subs+Mul+Maxs+Div+Mul+Muls into a single VF_CALL node.
 *        Formula:
 *          result = grad_output * (has_weight ? weight : 1) * (x - y) / max(1e-12, (1 - x) * x)
 *          result *= (has_mean ? reduceMeanCof : 1)
 */
#ifndef ASCENDC_BINARY_CORSS_ENTROPY_GRAD_DAG_H_
#define ASCENDC_BINARY_CORSS_ENTROPY_GRAD_DAG_H_

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#endif

constexpr int BCEG_SCALAR_REDUCEMEAN_IDX = 0;

namespace BinaryCrossEntropyGrad {
using namespace Ops::Base;

#ifdef __CCE_AICORE__
using namespace AscendC;
constexpr static Reg::CastTrait BCEG_CAST_B16_TO_B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr static Reg::CastTrait BCEG_CAST_B32_TO_B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename U, typename T>
__aicore__ inline void LoadOneTensor(Reg::RegTensor<T>& dst, __ubuf__ U*& input, Reg::MaskReg& pregUp,
                                     uint32_t oneRepeat)
{
    if constexpr (IsSameType<T, float>::value && !IsSameType<U, T>::value) {
        Reg::RegTensor<U> regCopyIn;
        Reg::RegTensor<U> regUnpack;
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(regCopyIn, input, (int32_t)oneRepeat);
        Reg::UnPack((Reg::RegTensor<int32_t>&)regUnpack, (Reg::RegTensor<int16_t>&)regCopyIn);
        Reg::Cast<T, U, BCEG_CAST_B16_TO_B32>(dst, regUnpack, pregUp);
    } else {
        Reg::LoadAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(dst, input, (int32_t)oneRepeat);
    }
}

template <typename U, typename T>
__aicore__ inline void StoreOneTensor(__ubuf__ U*& output, Reg::RegTensor<T>& src, Reg::MaskReg& pregUp,
                                      uint32_t oneRepeat)
{
    if constexpr (IsSameType<T, float>::value && !IsSameType<U, T>::value) {
        Reg::RegTensor<U> regTmp;
        Reg::RegTensor<U> regCopyOut;
        Reg::MaskReg pregT;
        Reg::Cast<U, T, BCEG_CAST_B32_TO_B16>(regTmp, src, pregUp);
        Reg::Pack((Reg::RegTensor<uint16_t>&)regCopyOut, (Reg::RegTensor<uint32_t>&)regTmp);
        Reg::Pack(pregT, pregUp);
        Reg::StoreAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(output, regCopyOut, (int32_t)oneRepeat, pregT);
    } else {
        Reg::StoreAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(output, src, (int32_t)oneRepeat, pregUp);
    }
}

// ── Core VF: fuses Cast + Sub + Subs + Mul + Maxs + Div + Mul + (optional Mul weight) + (optional Muls norm) ──
template <typename U, typename T, bool HasWeight, bool HasMean>
__aicore__ inline void CalcBCEGradVF(__ubuf__ U* xAddr, __ubuf__ U* yAddr, __ubuf__ U* gradOutAddr,
                                     __ubuf__ U* weightAddr, __ubuf__ T* outAddr, T normUp, uint16_t repeatTimes,
                                     uint32_t oneRepeat, uint32_t totalLen)
{
    constexpr T one = static_cast<T>(1.0f);
    constexpr T epsilon = static_cast<T>(1e-12);

    Reg::MaskReg pregUp;
    Reg::RegTensor<T> regX;
    Reg::RegTensor<T> regY;
    Reg::RegTensor<T> regGradOut;
    Reg::RegTensor<T> regWeight;
    Reg::RegTensor<T> regTmp;
    Reg::RegTensor<T> regDenom;
    Reg::RegTensor<T> regResult;

    for (uint16_t vfRepeat = 0; vfRepeat < repeatTimes; vfRepeat++) {
        pregUp = Reg::UpdateMask<T>(totalLen);

        LoadOneTensor<U, T>(regX, xAddr, pregUp, oneRepeat);
        LoadOneTensor<U, T>(regY, yAddr, pregUp, oneRepeat);
        LoadOneTensor<U, T>(regGradOut, gradOutAddr, pregUp, oneRepeat);

        // denom = (1 - x) * x, clamped to epsilon
        Reg::Duplicate(regDenom, one, pregUp);          // regDenom = 1 (broadcast scalar)
        Reg::Sub(regDenom, regDenom, regX, pregUp);     // regDenom = 1 - x
        Reg::Mul(regDenom, regDenom, regX, pregUp);     // regDenom = (1-x)*x
        Reg::Maxs(regDenom, regDenom, epsilon, pregUp); // clamp

        // result = grad_output * (x - y) / denom
        Reg::Sub(regResult, regX, regY, pregUp);            // regResult = x - y
        Reg::Mul(regResult, regGradOut, regResult, pregUp); // result = grad * (x-y)
        Reg::Div(regResult, regResult, regDenom, pregUp);   // result = grad*(x-y) / denom

        // optional weight multiply
        if constexpr (HasWeight) {
            LoadOneTensor<U, T>(regWeight, weightAddr, pregUp, oneRepeat);
            Reg::Mul(regResult, regResult, regWeight, pregUp);
        }

        // optional mean multiply
        if constexpr (HasMean) {
            Reg::Muls(regResult, regResult, normUp, pregUp);
        }

        StoreOneTensor<T, T>(outAddr, regResult, pregUp, oneRepeat);
    }
}
#endif

// ── BCEGMeanHasWeight: 4 tensor inputs + norm scalar → VF_CALL ──
template <typename U, typename T = float>
struct CalcBCEGMeanHasWeightOp : public Vec::ElemwiseQuinaryOP<T, U, U, U, U, T> {
    __aicore__ inline CalcBCEGMeanHasWeightOp(LocalTensor<T>& result, LocalTensor<U>& x, LocalTensor<U>& y,
                                              LocalTensor<U>& gradOutput, LocalTensor<U>& weight, T& norm,
                                              int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = AscendC::GetVecLen() / sizeof(T);
        uint32_t totalLen = count;
        uint16_t repeatTimes = (uint16_t)CeilDiv<uint32_t>(totalLen, oneRepeat);

        VF_CALL<CalcBCEGradVF<U, T, true, true>>(
            (__ubuf__ U*)x.GetPhyAddr(), (__ubuf__ U*)y.GetPhyAddr(), (__ubuf__ U*)gradOutput.GetPhyAddr(),
            (__ubuf__ U*)weight.GetPhyAddr(), (__ubuf__ T*)result.GetPhyAddr(), norm, repeatTimes, oneRepeat, totalLen);
#endif
    }
};

// ── BCEGMeanHasNoWeight: 3 tensor inputs + norm scalar → VF_CALL ──
template <typename U, typename T = float>
struct CalcBCEGMeanHasNoWeightOp : public Vec::ElemwiseQuaternaryOP<T, U, U, U, T> {
    __aicore__ inline CalcBCEGMeanHasNoWeightOp(LocalTensor<T>& result, LocalTensor<U>& x, LocalTensor<U>& y,
                                                LocalTensor<U>& gradOutput, T& norm, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = AscendC::GetVecLen() / sizeof(T);
        uint32_t totalLen = count;
        uint16_t repeatTimes = (uint16_t)CeilDiv<uint32_t>(totalLen, oneRepeat);

        VF_CALL<CalcBCEGradVF<U, T, false, true>>(
            (__ubuf__ U*)x.GetPhyAddr(), (__ubuf__ U*)y.GetPhyAddr(), (__ubuf__ U*)gradOutput.GetPhyAddr(),
            (__ubuf__ U*)nullptr, (__ubuf__ T*)result.GetPhyAddr(), norm, repeatTimes, oneRepeat, totalLen);
#endif
    }
};

// ── BCEGSumHasWeight: 4 tensor inputs → VF_CALL ──
template <typename U, typename T = float>
struct CalcBCEGSumHasWeightOp : public Vec::ElemwiseQuaternaryOP<T, U, U, U, U> {
    __aicore__ inline CalcBCEGSumHasWeightOp(LocalTensor<T>& result, LocalTensor<U>& x, LocalTensor<U>& y,
                                             LocalTensor<U>& gradOutput, LocalTensor<U>& weight, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = AscendC::GetVecLen() / sizeof(T);
        uint32_t totalLen = count;
        uint16_t repeatTimes = (uint16_t)CeilDiv<uint32_t>(totalLen, oneRepeat);

        T normDummy = static_cast<T>(1.0f);
        VF_CALL<CalcBCEGradVF<U, T, true, false>>((__ubuf__ U*)x.GetPhyAddr(), (__ubuf__ U*)y.GetPhyAddr(),
                                                  (__ubuf__ U*)gradOutput.GetPhyAddr(),
                                                  (__ubuf__ U*)weight.GetPhyAddr(), (__ubuf__ T*)result.GetPhyAddr(),
                                                  normDummy, repeatTimes, oneRepeat, totalLen);
#endif
    }
};

// ── BCEGSumHasNoWeight: 3 tensor inputs → VF_CALL ──
template <typename U, typename T = float>
struct CalcBCEGSumHasNoWeightOp : public Vec::ElemwiseTernaryOP<T, U, U, U> {
    __aicore__ inline CalcBCEGSumHasNoWeightOp(LocalTensor<T>& result, LocalTensor<U>& x, LocalTensor<U>& y,
                                               LocalTensor<U>& gradOutput, int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = AscendC::GetVecLen() / sizeof(T);
        uint32_t totalLen = count;
        uint16_t repeatTimes = (uint16_t)CeilDiv<uint32_t>(totalLen, oneRepeat);

        T normDummy = static_cast<T>(1.0f);
        VF_CALL<CalcBCEGradVF<U, T, false, false>>(
            (__ubuf__ U*)x.GetPhyAddr(), (__ubuf__ U*)y.GetPhyAddr(), (__ubuf__ U*)gradOutput.GetPhyAddr(),
            (__ubuf__ U*)nullptr, (__ubuf__ T*)result.GetPhyAddr(), normDummy, repeatTimes, oneRepeat, totalLen);
#endif
    }
};

// ── DAG definitions (simplified: CopyInBrc → CustomOp → Cast → CopyOut) ──

template <typename U, typename T = float>
struct BCEGMeanHasWeight {
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyGradOutput = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyWeight = Bind<Vec::CopyInBrc<U>, Placeholder::In3<U>>;

    using OpCalc = Bind<CalcBCEGMeanHasWeightOp<U, T>, OpCopyX, OpCopyY, OpCopyGradOutput, OpCopyWeight,
                        Placeholder::Var<T, BCEG_SCALAR_REDUCEMEAN_IDX>>;
    using OpRes = Bind<Vec::Cast<U, T, 1>, OpCalc>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEGMeanHasNoWeight {
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyGradOutput = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;

    using OpCalc = Bind<CalcBCEGMeanHasNoWeightOp<U, T>, OpCopyX, OpCopyY, OpCopyGradOutput,
                        Placeholder::Var<T, BCEG_SCALAR_REDUCEMEAN_IDX>>;
    using OpRes = Bind<Vec::Cast<U, T, 1>, OpCalc>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEGSumHasWeight {
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyGradOutput = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyWeight = Bind<Vec::CopyInBrc<U>, Placeholder::In3<U>>;

    using OpCalc = Bind<CalcBCEGSumHasWeightOp<U, T>, OpCopyX, OpCopyY, OpCopyGradOutput, OpCopyWeight>;
    using OpRes = Bind<Vec::Cast<U, T, 1>, OpCalc>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEGSumHasNoWeight {
    using OpCopyX = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyY = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyGradOutput = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;

    using OpCalc = Bind<CalcBCEGSumHasNoWeightOp<U, T>, OpCopyX, OpCopyY, OpCopyGradOutput>;
    using OpRes = Bind<Vec::Cast<U, T, 1>, OpCalc>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace BinaryCrossEntropyGrad
#endif // ASCENDC_BINARY_CORSS_ENTROPY_GRAD_DAG_H_
