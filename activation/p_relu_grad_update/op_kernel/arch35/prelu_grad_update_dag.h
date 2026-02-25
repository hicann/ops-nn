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
 * \file prelu_grad_update_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_PRELU_GRAD_UPDATE_DAG_H
#define CANN_CUSTOM_OPS_PRELU_GRAD_UPDATE_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

const float FP32_ZERO = 0.0;

namespace PReluGradUpdate {
using namespace AscendC;
using namespace Ops::Base;

template <class T>
struct PReluGradDxCustom : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline PReluGradDxCustom(
        const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const LocalTensor<T>& src2,
        const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vRegTensorInputFeatures;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vRegTensorInputWeights;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vRegTensorInputGradients;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vRegTensorOutputDx;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vRegTensorInputZero;

        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg cmpMaskReg;
        __VEC_SCOPE__
        {
            MicroAPI::Duplicate(vRegTensorInputZero, FP32_ZERO);
            uint32_t sregMask=count;
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(sregMask);
                // OpCopyIn
                MicroAPI::DataCopy(vRegTensorInputGradients, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                MicroAPI::DataCopy(vRegTensorInputFeatures, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                MicroAPI::DataCopy(vRegTensorInputWeights, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                // compute
                MicroAPI::Mul(vRegTensorOutputDx, vRegTensorInputWeights, vRegTensorInputGradients, mask);
                MicroAPI::Compare<T, CMPMODE::GT>(cmpMaskReg, vRegTensorInputFeatures, vRegTensorInputZero, mask);
                MicroAPI::Select<T>(vRegTensorOutputDx, vRegTensorInputGradients, vRegTensorOutputDx, cmpMaskReg);
                // OpCopyOut
                MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vRegTensorOutputDx, mask);
            }
        }
#endif
    }
};

template <class T>
struct PReluGradDaCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline PReluGradDaCustom(
        const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vRegTensorInputFeatures;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vRegTensorInputGradients;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vRegTensorOutputDa;
        MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vRegTensorInputZero;

        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg cmpMaskReg;
        __VEC_SCOPE__
        {
            MicroAPI::Duplicate(vRegTensorInputZero, FP32_ZERO);
            uint32_t sregMask = count;
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(sregMask);
                // OpCopyIn
                MicroAPI::DataCopy(vRegTensorInputGradients, (__ubuf__ T*)(src0Addr + loopIdx * vlSize));
                MicroAPI::DataCopy(vRegTensorInputFeatures, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));

                // compute
                MicroAPI::Mul(vRegTensorOutputDa, vRegTensorInputFeatures, vRegTensorInputGradients, mask);
                MicroAPI::Compare<T, CMPMODE::GT>(cmpMaskReg, vRegTensorInputFeatures, vRegTensorInputZero, mask);
                MicroAPI::Select<T>(vRegTensorOutputDa, vRegTensorInputZero, vRegTensorOutputDa, cmpMaskReg);
                // OpCopyOut
                MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vRegTensorOutputDa, mask);
            }
        }
#endif
    }
};

template <typename U>
struct PReluGradUpdateDAG {
    using OpCopyIn0 = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyIn2 = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpResultDx = Bind<PReluGradDxCustom<U>, OpCopyIn0, OpCopyIn1, OpCopyIn2>;
    using OpResultDa = Bind<PReluGradDaCustom<U>, OpCopyIn0, OpCopyIn1>;

    using OpCopyOutDx = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultDx>;
    using OpCopyOutDa = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpResultDa>;
    using Outputs = Elems<OpCopyOutDx, OpCopyOutDa>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace PReluGradUpdate

#endif // CANN_CUSTOM_OPS_PRELUGRADUPDATE_DAG_H