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
 * \file sigmoid_cross_entropy_with_logits_grad_v2_dag.h
 * \brief
 */

#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DAG_H
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace SigmoidCrossEntropyWithLogitsGradV2Op {
using namespace Ops::Base;
using namespace AscendC;
constexpr uint32_t VEC_LENGTH_B32 = 64;

#ifdef __CCE_AICORE__
using AscendC::Reg::RegTensor;
__aicore__ inline void SigmoidCustomImpl(Reg::RegTensor<float>& yReg, Reg::RegTensor<float>& xReg,
                                         Reg::RegTensor<float>& oneReg, Reg::MaskReg& mask)
{
    Reg::Muls(yReg, xReg, -1.0f, mask);
    Reg::Exp(yReg, yReg, mask);
    Reg::Add(yReg, yReg, oneReg, mask);
    Reg::Div(yReg, oneReg, yReg, mask);
}
#endif

template <uint32_t HAS_WEIGHT, uint32_t HAS_POS_WEIGHT, uint32_t IS_MEAN>
__aicore__ inline void SigmoidCrossEntropyWithLogitsGradV2Impl(LocalTensor<float>& dst, LocalTensor<float>& predict,
                                                               LocalTensor<float>& target, LocalTensor<float>& dout,
                                                               LocalTensor<float>& weight,
                                                               LocalTensor<float>& posWeight, float scale,
                                                               uint32_t count)
{
#ifdef __CCE_AICORE__
    uint16_t loopNum = (count + VEC_LENGTH_B32 - 1) / VEC_LENGTH_B32;
    uint32_t vlSize = VEC_LENGTH_B32;

    __ubuf__ float* predictAddr = (__ubuf__ float*)predict.GetPhyAddr();
    __ubuf__ float* targetAddr = (__ubuf__ float*)target.GetPhyAddr();
    __ubuf__ float* doutAddr = (__ubuf__ float*)dout.GetPhyAddr();
    __ubuf__ float* weightAddr = HAS_WEIGHT ? (__ubuf__ float*)weight.GetPhyAddr() : nullptr;
    __ubuf__ float* posWeightAddr = HAS_POS_WEIGHT ? (__ubuf__ float*)posWeight.GetPhyAddr() : nullptr;
    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();

    __VEC_SCOPE__
    {
        Reg::MaskReg mask0 = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<float> oneReg;
        Reg::Duplicate(oneReg, 1.0f, mask0);

        Reg::RegTensor<float> predictReg;
        Reg::RegTensor<float> targetReg;
        Reg::RegTensor<float> doutReg;
        Reg::RegTensor<float> weightReg;
        Reg::RegTensor<float> posWeightReg;
        Reg::RegTensor<float> gradientReg;
        Reg::RegTensor<float> posWeightTargetReg;
        Reg::RegTensor<float> tmpReg;
        Reg::MaskReg mask;
        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(predictReg, predictAddr, vlSize);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(targetReg, targetAddr, vlSize);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(doutReg, doutAddr, vlSize);
            if constexpr (HAS_WEIGHT) {
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(weightReg, weightAddr, vlSize);
            }
            if constexpr (HAS_POS_WEIGHT) {
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(posWeightReg, posWeightAddr, vlSize);
            }

            SigmoidCustomImpl(gradientReg, predictReg, oneReg, mask);

            if constexpr (HAS_POS_WEIGHT) {
                Reg::Mul(posWeightTargetReg, posWeightReg, targetReg, mask);
                Reg::Adds(tmpReg, posWeightTargetReg, 1.0f, mask);
                Reg::Sub(tmpReg, tmpReg, targetReg, mask);
                Reg::Mul(gradientReg, tmpReg, gradientReg, mask);
                Reg::Sub(gradientReg, gradientReg, posWeightTargetReg, mask);
            } else {
                Reg::Sub(gradientReg, gradientReg, targetReg, mask);
            }

            Reg::Mul(gradientReg, gradientReg, doutReg, mask);

            if constexpr (HAS_WEIGHT) {
                Reg::Mul(gradientReg, gradientReg, weightReg, mask);
            }

            if constexpr (IS_MEAN == 1) {
                Reg::Muls(gradientReg, gradientReg, scale, mask);
            }

            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, gradientReg, vlSize, mask);
        }
    }
#endif
}

template <uint32_t HAS_WEIGHT, uint32_t HAS_POS_WEIGHT, uint32_t IS_MEAN>
struct DagWeightPosWightCompute : public Vec::Elemwise6OP<float, float, float, float, float, float, float> {
    __aicore__ inline DagWeightPosWightCompute(LocalTensor<float>& dst, LocalTensor<float>& predict,
                                               LocalTensor<float>& target, LocalTensor<float>& dout,
                                               LocalTensor<float>& weight, LocalTensor<float>& posWeight, float scale,
                                               uint32_t count)
    {
        SigmoidCrossEntropyWithLogitsGradV2Impl<HAS_WEIGHT, HAS_POS_WEIGHT, IS_MEAN>(dst, predict, target, dout, weight,
                                                                                     posWeight, scale, count);
    }
};

template <uint32_t HAS_WEIGHT, uint32_t HAS_POS_WEIGHT, uint32_t IS_MEAN>
struct DagWeightCompute : public Vec::ElemwiseQuinaryOP<float, float, float, float, float, float> {
    __aicore__ inline DagWeightCompute(LocalTensor<float>& dst, LocalTensor<float>& predict, LocalTensor<float>& target,
                                       LocalTensor<float>& dout, LocalTensor<float>& weight, float scale,
                                       uint32_t count)
    {
        SigmoidCrossEntropyWithLogitsGradV2Impl<HAS_WEIGHT, HAS_POS_WEIGHT, IS_MEAN>(dst, predict, target, dout, weight,
                                                                                     predict, scale, count);
    }
};

template <uint32_t HAS_WEIGHT, uint32_t HAS_POS_WEIGHT, uint32_t IS_MEAN>
struct DagPosWeightCompute : public Vec::ElemwiseQuinaryOP<float, float, float, float, float, float> {
    __aicore__ inline DagPosWeightCompute(LocalTensor<float>& dst, LocalTensor<float>& predict,
                                          LocalTensor<float>& target, LocalTensor<float>& dout,
                                          LocalTensor<float>& posWeight, float scale, uint32_t count)
    {
        SigmoidCrossEntropyWithLogitsGradV2Impl<HAS_WEIGHT, HAS_POS_WEIGHT, IS_MEAN>(dst, predict, target, dout,
                                                                                     predict, posWeight, scale, count);
    }
};

template <uint32_t HAS_WEIGHT, uint32_t HAS_POS_WEIGHT, uint32_t IS_MEAN>
struct DagCompute : public Vec::ElemwiseQuaternaryOP<float, float, float, float, float> {
    __aicore__ inline DagCompute(LocalTensor<float>& dst, LocalTensor<float>& predict, LocalTensor<float>& target,
                                 LocalTensor<float>& dout, float scale, uint32_t count)
    {
        SigmoidCrossEntropyWithLogitsGradV2Impl<HAS_WEIGHT, HAS_POS_WEIGHT, IS_MEAN>(dst, predict, target, dout,
                                                                                     predict, predict, scale, count);
    }
};

template <typename T, uint32_t IS_MEAN>
struct SigmoidCrossEntropyWithLogitsGradV2DagWeightPosWight {
    using InputPredictT = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputTargetT = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputDoutT = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using InputWeightT = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;
    using InputPosWeightT = Bind<Vec::CopyInBrc<T>, Placeholder::In4<T>>;

    // cast
    using InputPredict = Bind<Vec::Cast<float, T, 0>, InputPredictT>;
    using InputTarget = Bind<Vec::Cast<float, T, 0>, InputTargetT>;
    using InputDout = Bind<Vec::Cast<float, T, 0>, InputDoutT>;
    using InputWeight = Bind<Vec::Cast<float, T, 0>, InputWeightT>;
    using InputPosWeight = Bind<Vec::Cast<float, T, 0>, InputPosWeightT>;

    using OpResult = Bind<DagWeightPosWightCompute<1, 1, IS_MEAN>, InputPredict, InputTarget, InputDout, InputWeight,
                          InputPosWeight, Placeholder::Var<float, 0>>;
    using OpCastRes = Bind<Vec::Cast<T, float, 1>, OpResult>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, uint32_t IS_MEAN>
struct SigmoidCrossEntropyWithLogitsGradV2DagWeight {
    using InputPredictT = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputTargetT = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputDoutT = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using InputWeightT = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;

    // cast
    using InputPredict = Bind<Vec::Cast<float, T, 0>, InputPredictT>;
    using InputTarget = Bind<Vec::Cast<float, T, 0>, InputTargetT>;
    using InputDout = Bind<Vec::Cast<float, T, 0>, InputDoutT>;
    using InputWeight = Bind<Vec::Cast<float, T, 0>, InputWeightT>;

    using OpResult = Bind<DagWeightCompute<1, 0, IS_MEAN>, InputPredict, InputTarget, InputDout, InputWeight,
                          Placeholder::Var<float, 0>>;
    using OpCastRes = Bind<Vec::Cast<T, float, 1>, OpResult>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, uint32_t IS_MEAN>
struct SigmoidCrossEntropyWithLogitsGradV2DagPosWeight {
    using InputPredictT = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputTargetT = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputDoutT = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using InputPosWeightT = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;

    // cast
    using InputPredict = Bind<Vec::Cast<float, T, 0>, InputPredictT>;
    using InputTarget = Bind<Vec::Cast<float, T, 0>, InputTargetT>;
    using InputDout = Bind<Vec::Cast<float, T, 0>, InputDoutT>;
    using InputPosWeight = Bind<Vec::Cast<float, T, 0>, InputPosWeightT>;

    using OpResult = Bind<DagPosWeightCompute<0, 1, IS_MEAN>, InputPredict, InputTarget, InputDout, InputPosWeight,
                          Placeholder::Var<float, 0>>;
    using OpCastRes = Bind<Vec::Cast<T, float, 1>, OpResult>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, uint32_t IS_MEAN>
struct SigmoidCrossEntropyWithLogitsGradV2Dag {
    using InputPredictT = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputTargetT = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputDoutT = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;

    // cast
    using InputPredict = Bind<Vec::Cast<float, T, 0>, InputPredictT>;
    using InputTarget = Bind<Vec::Cast<float, T, 0>, InputTargetT>;
    using InputDout = Bind<Vec::Cast<float, T, 0>, InputDoutT>;

    using OpResult = Bind<DagCompute<0, 0, IS_MEAN>, InputPredict, InputTarget, InputDout, Placeholder::Var<float, 0>>;
    using OpCastRes = Bind<Vec::Cast<T, float, 1>, OpResult>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace SigmoidCrossEntropyWithLogitsGradV2Op

#endif // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_DAG_H
