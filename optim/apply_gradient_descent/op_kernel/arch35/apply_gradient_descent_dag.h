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
 * \file apply_gradient_descent_dag.h
 * \brief apply_gradient_descent_dag
 */

#ifndef CANN_CUSTOM_OPS_APPLY_GRADIENT_DEESCENT_DAG_H
#define CANN_CUSTOM_OPS_APPLY_GRADIENT_DEESCENT_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#ifdef __CCE_AICORE__
#include "../inc/load_store_utils.h"
#endif

namespace ApplyGradientDescentOp {
using namespace AscendC;
using namespace Ops::Base;

template <typename T>
struct CustomDescentOp : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline CustomDescentOp(LocalTensor<T>& dstLocal, LocalTensor<T>& varLocal, T alphaScalar,
                                      LocalTensor<T>& deltaLocal, int count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vlSize = VECTOR_REG_WIDTH / dtypeSize;
        uint32_t totalLen = count;
        uint16_t loopNum = (totalLen + vlSize - 1) / vlSize;

        __ubuf__ T* varAddr = (__ubuf__ T*)varLocal.GetPhyAddr();
        __ubuf__ T* deltaAddr = (__ubuf__ T*)deltaLocal.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
        if constexpr (std::is_same_v<T, float>) {
            __VEC_SCOPE__
            {
                MicroAPI::MaskReg mask;
                MicroAPI::RegTensor<T> varReg;
                MicroAPI::RegTensor<T> deltaReg;
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<T>(totalLen);
                    MicroAPI::LoadAlign(varReg, varAddr + loopIdx * vlSize);
                    MicroAPI::LoadAlign(deltaReg, deltaAddr + loopIdx * vlSize);
                    MicroAPI::Muls(deltaReg, deltaReg, alphaScalar, mask);
                    MicroAPI::Sub(varReg, varReg, deltaReg, mask);
                    MicroAPI::StoreAlign(dstAddr + loopIdx * vlSize, varReg, mask);
                }
            }
        } else {
            float alphaVal = 0.0f;
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                alphaVal = ToFloat(alphaScalar);
            } else {
                alphaVal = float(alphaScalar);
            }
            __VEC_SCOPE__
            {
                MicroAPI::MaskReg mask;
                MicroAPI::RegTensor<float> varRegToFloat;
                MicroAPI::RegTensor<float> deltaRegToFloat;
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = MicroAPI::UpdateMask<float>(totalLen);
                    ops::LoadOneTensorForDtypeT<T>(varAddr, varRegToFloat, mask, loopIdx * vlSize);
                    ops::LoadOneTensorForDtypeT<T>(deltaAddr, deltaRegToFloat, mask, loopIdx * vlSize);
                    MicroAPI::Muls(deltaRegToFloat, deltaRegToFloat, alphaVal, mask);
                    MicroAPI::Sub(varRegToFloat, varRegToFloat, deltaRegToFloat, mask);
                    ops::StoreOneTensorForDtypeT<T>(dstAddr, varRegToFloat, mask, loopIdx * vlSize);
                }
            }
        }
#endif
    }
};

template <typename T>
struct ApplyGradientDescentDAG {
    using OpVar = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpDelta = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using OpAlpha = Placeholder::In1<T, Placeholder::ScalarAttr<true>>;

    using OpOutput = Bind<CustomDescentOp<T>, OpVar, OpAlpha, OpDelta>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpOutput>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ApplyGradientDescentOp
#endif // CANN_CUSTOM_OPS_APPLY_GRADIENT_DEESCENT_DAG_H
