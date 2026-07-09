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
 * \file elu_dag.h
 * \brief elu implement
 */

#ifndef CANN_CUSTOM_OPS_ELU_DAG_H
#define CANN_CUSTOM_OPS_ELU_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace EluOp {
using namespace Ops::Base;
using namespace AscendC;
#ifdef __CCE_AICORE__
    constexpr static AscendC::MicroAPI::CastTrait castTrait0 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
    constexpr static AscendC::MicroAPI::CastTrait castTrait1 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
#endif
constexpr int ELU_ATTR_ALPHA_INDEX = 0;
constexpr int ELU_ATTR_SCALE_INDEX = 1;
constexpr int ELU_ATTR_INPUT_SCALE_INDEX = 2;

namespace EluDag1 {
template <class T>
struct EluCustom : public Vec::ElemwiseQuaternaryOP<T, T, float, float, float> {
    __aicore__ inline EluCustom(LocalTensor<T>& dst, LocalTensor<T>& src, float alpha, float scale, float inputScale,
                                uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint32_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;
        float constNegOne = -1;

        constexpr float coeff5 = 1.0f / 120.0f;
        constexpr float coeff4 = 1.0f / 24.0f;
        constexpr float coeff3 = 1.0f / 6.0f;
        constexpr float coeff2 = 1.0f / 2.0f;
        constexpr float coeff1 = 1.0f;

        constexpr float expm1Threshold = 0.1f;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputFloat;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregNeg;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregOutputFloat;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregExp;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregPoly;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregAbsZ;
        MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregT1;
        MicroAPI::MaskReg mask, cmpMask, cmpExpm1;
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < static_cast<uint16_t>(loopNum); loopIdx++) {
                    mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    MicroAPI::Muls(vregNeg, vregInput, inputScale, mask);

                    // Polynomial expm1(z) = z + z^2/2 + z^3/6 + z^4/24 + z^5/120 for |z| < threshold
                    // Horner: T1 = T1 * z + coeff, then poly = T1 * z
                    MicroAPI::Muls(vregT1, vregNeg, coeff5, mask);
                    MicroAPI::Adds(vregT1, vregT1, coeff4, mask);
                    MicroAPI::Mul(vregT1, vregT1, vregNeg, mask);
                    MicroAPI::Adds(vregT1, vregT1, coeff3, mask);
                    MicroAPI::Mul(vregT1, vregT1, vregNeg, mask);
                    MicroAPI::Adds(vregT1, vregT1, coeff2, mask);
                    MicroAPI::Mul(vregT1, vregT1, vregNeg, mask);
                    MicroAPI::Adds(vregT1, vregT1, coeff1, mask);
                    MicroAPI::Mul(vregPoly, vregT1, vregNeg, mask);

                    // exp(z) - 1 for |z| >= threshold
                    MicroAPI::Exp(vregExp, vregNeg, mask);
                    MicroAPI::Adds(vregExp, vregExp, constNegOne, mask);

                    // Select expm1 result based on |z|
                    MicroAPI::Abs(vregAbsZ, vregNeg, mask);
                    MicroAPI::CompareScalar<T, CMPMODE::LT>(cmpExpm1, vregAbsZ, expm1Threshold, mask);
                    MicroAPI::Select<T>(vregNeg, vregPoly, vregExp, cmpExpm1);

                    MicroAPI::Muls(vregNeg, vregNeg, alpha, mask);

                    MicroAPI::CompareScalar<T, CMPMODE::GT>(cmpMask, vregInput, (float)0.0, mask);
                    MicroAPI::Select<T>(vregOutput, vregInput, vregNeg, cmpMask);
                    MicroAPI::Muls(vregOutput, vregOutput, scale, mask);

                    // OpCopyOut
                    MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < static_cast<uint16_t>(loopNum); loopIdx++) {
                    mask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    MicroAPI::Cast<float, T, castTrait0>(vregInputFloat, vregInput, mask);
                    MicroAPI::Muls(vregNeg, vregInputFloat, inputScale, mask);

                    // Polynomial expm1(z) = z + z^2/2 + z^3/6 + z^4/24 for |z| < threshold
                    // Horner: T1 = T1 * z + coeff, then poly = T1 * z
                    MicroAPI::Muls(vregT1, vregNeg, coeff4, mask);
                    MicroAPI::Adds(vregT1, vregT1, coeff3, mask);
                    MicroAPI::Mul(vregT1, vregT1, vregNeg, mask);
                    MicroAPI::Adds(vregT1, vregT1, coeff2, mask);
                    MicroAPI::Mul(vregT1, vregT1, vregNeg, mask);
                    MicroAPI::Adds(vregT1, vregT1, coeff1, mask);
                    MicroAPI::Mul(vregPoly, vregT1, vregNeg, mask);

                    // exp(z) - 1 for |z| >= threshold
                    MicroAPI::Exp(vregExp, vregNeg, mask);
                    MicroAPI::Adds(vregExp, vregExp, constNegOne, mask);

                    // Select expm1 result based on |z|
                    MicroAPI::Abs(vregAbsZ, vregNeg, mask);
                    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpExpm1, vregAbsZ, expm1Threshold, mask);
                    MicroAPI::Select<float>(vregNeg, vregPoly, vregExp, cmpExpm1);

                    MicroAPI::Muls(vregNeg, vregNeg, alpha, mask);

                    MicroAPI::CompareScalar<float, CMPMODE::GT>(cmpMask, vregInputFloat, (float)0.0, mask);
                    MicroAPI::Select<float>(vregOutputFloat, vregInputFloat, vregNeg, cmpMask);
                    MicroAPI::Muls(vregOutputFloat, vregOutputFloat, scale, mask);
                    MicroAPI::Cast<T, float, castTrait1>(vregOutput, vregOutputFloat, mask);
                    // OpCopyOut
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
                }
            }    
        }
#endif
    }
};
} // namespace EluDag1

template <typename U, typename T = float>
struct EluDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpResult = Bind<EluDag1::EluCustom<U>, OpCopyIn0, Placeholder::Var<float, ELU_ATTR_ALPHA_INDEX>,
                          Placeholder::Var<float, ELU_ATTR_SCALE_INDEX>,
                          Placeholder::Var<float, ELU_ATTR_INPUT_SCALE_INDEX>>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace EluOp
#endif // CANN_CUSTOM_OPS_ELU_DAG_H
