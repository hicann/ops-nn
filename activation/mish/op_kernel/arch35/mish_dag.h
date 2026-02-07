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

#ifndef CANN_CUSTOM_OPS_MISH_DAG_H
#define CANN_CUSTOM_OPS_MISH_DAG_H
 
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;
 
const float FP32_ZERO = 0.0;
const float FP32_ONE = 1.0;
const float FP32_TWO = 2.0;
const float FP32_NEG_ONE = -1.0;
const float FP32_NEG_TWO = -2.0;
 
#ifdef __CCE_AICORE__
constexpr static AscendC::MicroAPI::CastTrait castTrait0 = { AscendC::MicroAPI::RegLayout::ZERO,
AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
constexpr static AscendC::MicroAPI::CastTrait castTrait1 = { AscendC::MicroAPI::RegLayout::ZERO,
AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
#endif
namespace MishDag1 {
 
template<class T>
struct MishCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline MishCustom(LocalTensor<T> &dst, LocalTensor<T> &src, uint32_t count) {
        #ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
 
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputNegNumerator;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputNegDenominator;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputNumerator;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputDenominator;
 
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg cmpMaskReg;

        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput16;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput16;
        __VEC_SCOPE__ {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                // OpCopyIn
                if constexpr(std::is_same_v<T, float>) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInput16, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    MicroAPI::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);
                }
                MicroAPI::Muls(vregInputNegNumerator, vregInput, FP32_NEG_ONE, mask);// -x
                MicroAPI::Muls(vregInputNegDenominator, vregInput, FP32_NEG_TWO, mask);// -2x
                MicroAPI::Exp(vregInputNegNumerator, vregInputNegNumerator, mask);// e^-x
                MicroAPI::Exp(vregInputNegDenominator, vregInputNegDenominator, mask);// e^-2x
                MicroAPI::Muls(vregInputNegNumerator, vregInputNegNumerator, FP32_TWO, mask);// 2e^-x
                MicroAPI::Adds(vregInputNegNumerator, vregInputNegNumerator, FP32_ONE, mask);// 2e^-x + 1
                MicroAPI::Muls(vregInputNegDenominator, vregInputNegDenominator, FP32_TWO, mask);// 2e^-2x
                MicroAPI::Add(vregInputNegDenominator, vregInputNegNumerator, vregInputNegDenominator, mask);
                MicroAPI::Div(vregOutput, vregInputNegNumerator, vregInputNegDenominator, mask);

                MicroAPI::Muls(vregInputNumerator, vregInput, FP32_TWO, mask);// 2x
                MicroAPI::Exp(vregInputNumerator, vregInputNumerator, mask);// e^2x
                MicroAPI::Exp(vregInputDenominator, vregInput, mask);// e^x
                MicroAPI::Axpy(vregInputNumerator, vregInputDenominator, FP32_TWO, mask);// e^2x + 2e^x
                MicroAPI::Adds(vregInputDenominator, vregInputNumerator, FP32_TWO, mask);// e^2x + 2e^x + 2
                MicroAPI::Div(vregInputNumerator, vregInputNumerator, vregInputDenominator, mask);

                MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, vregInput, FP32_ZERO, mask);
                MicroAPI::Select(vregOutput, vregInputNumerator, vregOutput, cmpMaskReg);
                MicroAPI::Mul(vregOutput, vregOutput, vregInput, mask);

                // OpCopyOut
                if constexpr(std::is_same_v<T, float>) {
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                } else {
                    MicroAPI::Cast<T, float, castTrait1>(vregOutput16, vregOutput, mask);
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput16, mask);
                }
            }
        } 
        #endif
    }
};
}

template <typename U, typename T=float>
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
 #endif  // CANN_CUSTOM_OPS_MISH_DAG_H