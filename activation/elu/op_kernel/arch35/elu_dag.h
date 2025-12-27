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

using namespace Ops::Base;
using namespace AscendC;
constexpr int ELU_ATTR_ALPHA_INDEX = 0;
constexpr int ELU_ATTR_SCALE_INDEX = 1;
constexpr int ELU_ATTR_INPUT_SCALE_INDEX = 2;

namespace EluDag1 {
template <class T>
struct EluCustom : public Vec::ElemwiseQuaternaryOP<T, T, float, float, float> {
    __aicore__ inline EluCustom(
        LocalTensor<T>& dst, LocalTensor<T>& src, float alpha, float scale, float inputScale, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint32_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;
        T constNegOne = -1;

        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregInput;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregNeg;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vregOutput;
        MicroAPI::MaskReg mask, cmpMask;
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < static_cast<uint16_t>(loopNum); loopIdx++) {
                    mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                    // OpCopyIn
                    MicroAPI::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                    MicroAPI::Muls(vregNeg, vregInput, inputScale, mask);
                    MicroAPI::Exp(vregNeg, vregNeg, mask);
                    MicroAPI::Adds(vregNeg, vregNeg, constNegOne, mask);
                    MicroAPI::Muls(vregNeg, vregNeg, alpha, mask);

                    MicroAPI::CompareScalar<T, CMPMODE::GT>(cmpMask, vregInput, (float)0.0, mask);
                    MicroAPI::Select<T>(vregOutput, vregInput, vregNeg, cmpMask);
                    MicroAPI::Muls(vregOutput, vregOutput, scale, mask);

                    // OpCopyOut
                    MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
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
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, 0>, OpCopyIn0>;

    using OpResult = Bind<
        EluDag1::EluCustom<T>, OpCopyIn0Cast, Placeholder::Var<float, ELU_ATTR_ALPHA_INDEX>,
        Placeholder::Var<float, ELU_ATTR_SCALE_INDEX>, Placeholder::Var<float, ELU_ATTR_INPUT_SCALE_INDEX>>;

    using OpResultCast = Bind<Vec::Cast<U, T, 1>, OpResult>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

#endif // CANN_CUSTOM_OPS_ELU_DAG_H