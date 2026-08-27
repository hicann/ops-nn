/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mish_dag.h
 * \brief
 */

#ifndef OPS_NN_ACTIVATION_MISH_KERNEL_DAG_H
#define OPS_NN_ACTIVATION_MISH_KERNEL_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;

const float FP32_ZERO = 0.0;
const float FP32_ONE = 1.0;
const float FP32_TWO = 2.0;
const float FP32_NEG_ONE = -1.0;
const float FP32_NEG_TWO = -2.0;

#ifdef __CCE_AICORE__
constexpr static AscendC::Reg::CastTrait castTrait0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_RINT};
#endif

namespace MishDag1 {
using namespace Ops::Base;

template <class T>
struct MishCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline MishCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInput;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputNegNumerator;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputNegDenominator;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputNumerator;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputDenominator;

        Reg::RegTensor<float, Reg::RegTraitNumOne> vregOutput;
        Reg::MaskReg mask;
        Reg::MaskReg cmpMaskReg;

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput16;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput16;
        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                // OpCopyIn
                if constexpr (std::is_same_v<T, float>) {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                } else {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(vregInput16,
                                                                      (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    Reg::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);
                }
                Reg::Muls(vregInputNegNumerator, vregInput, FP32_NEG_ONE, mask);             // -x
                Reg::Muls(vregInputNegDenominator, vregInput, FP32_NEG_TWO, mask);           // -2x
                Reg::Exp(vregInputNegNumerator, vregInputNegNumerator, mask);                // e^-x
                Reg::Exp(vregInputNegDenominator, vregInputNegDenominator, mask);            // e^-2x
                Reg::Muls(vregInputNegNumerator, vregInputNegNumerator, FP32_TWO, mask);     // 2e^-x
                Reg::Adds(vregInputNegNumerator, vregInputNegNumerator, FP32_ONE, mask);     // 2e^-x + 1
                Reg::Muls(vregInputNegDenominator, vregInputNegDenominator, FP32_TWO, mask); // 2e^-2x
                Reg::Add(vregInputNegDenominator, vregInputNegNumerator, vregInputNegDenominator, mask);
                Reg::Div(vregOutput, vregInputNegNumerator, vregInputNegDenominator, mask);

                Reg::Muls(vregInputNumerator, vregInput, FP32_TWO, mask);            // 2x
                Reg::Exp(vregInputNumerator, vregInputNumerator, mask);              // e^2x
                Reg::Exp(vregInputDenominator, vregInput, mask);                     // e^x
                Reg::Axpy(vregInputNumerator, vregInputDenominator, FP32_TWO, mask); // e^2x + 2e^x
                Reg::Adds(vregInputDenominator, vregInputNumerator, FP32_TWO, mask); // e^2x + 2e^x + 2
                Reg::Div(vregInputNumerator, vregInputNumerator, vregInputDenominator, mask);

                Reg::Compares<float, CMPMODE::LT>(cmpMaskReg, vregInput, FP32_ZERO, mask);
                Reg::Select(vregOutput, vregInputNumerator, vregOutput, cmpMaskReg);
                Reg::Mul(vregOutput, vregOutput, vregInput, mask);

                // OpCopyOut
                if constexpr (std::is_same_v<T, float>) {
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize),
                                                                      vregOutput, mask);
                } else {
                    Reg::Cast<T, float, castTrait1>(vregOutput16, vregOutput, mask);
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize),
                                                                      vregOutput16, mask);
                }
            }
        }
#endif
    }
};

template <typename U, typename T = float>
struct MishDAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpResult1 = Bind<MishDag1::MishCustom<T>, OpCopyIn0Cast>;
    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResult1>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace MishDag1
#endif // OPS_NN_ACTIVATION_MISH_KERNEL_DAG_H
