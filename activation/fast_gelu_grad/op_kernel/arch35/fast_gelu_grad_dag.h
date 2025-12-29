/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_CUSTOM_OPS_FAST_GELU_GRAD_DAG_H
#define CANN_CUSTOM_OPS_FAST_GELU_GRAD_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace FastGeluGradDag {
using namespace Ops::Base;

template <class T>
struct FastGeluGradCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline FastGeluGradCustom(
        LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, VL);
        uint32_t vlSize = VL;
        T value1 = 1.702;
        T value2 = -1.702;
        T value3 = 1.0;
        T value4 = -1.0;
        __VEC_SCOPE__
        {
            __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
            __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
            __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> dy;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> x;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> constantOne;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> value1MulsX;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> temp1Reg;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> temp2Reg;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> divRes;
            static constexpr AscendC::MicroAPI::DivSpecificMode mode = {
                AscendC::MicroAPI::MaskMergeMode::ZEROING, true};
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::Duplicate(constantOne, value3);
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = AscendC::MicroAPI::UpdateMask<T, AscendC::MicroAPI::RegTraitNumOne>(count);
                // OpCopyIn0
                AscendC::MicroAPI::DataCopy(x, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));
                // temp1Reg = e^(-1.702x) + 1
                AscendC::MicroAPI::Muls(value1MulsX, x, value2, mask);
                AscendC::MicroAPI::Exp(temp1Reg, value1MulsX, mask);
                AscendC::MicroAPI::Adds(temp1Reg, temp1Reg, value3, mask);
                // temp2Reg = (1/(e^(-1.702x) + 1)) - 1
                AscendC::MicroAPI::Div<T, &mode>(divRes, constantOne, temp1Reg, mask);
                AscendC::MicroAPI::Adds(temp2Reg, divRes, value4, mask);
                // divRes = (temp2Reg * -1.702x + 1) * 1/(e^(-1.702x) * dy
                AscendC::MicroAPI::Mul(temp2Reg, temp2Reg, value1MulsX, mask);
                AscendC::MicroAPI::Adds(temp2Reg, temp2Reg, value3, mask);
                AscendC::MicroAPI::Mul(divRes, temp2Reg, divRes, mask);
                AscendC::MicroAPI::DataCopy(dy, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                AscendC::MicroAPI::Mul(divRes, dy, divRes, mask);

                // OpCopyOut
                AscendC::MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), divRes, mask);
            }
        }
#endif
    }
};

template <typename T>
struct FastGeluGradNeedCast {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>; // dy
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>; // x
    using CastIn0 = Bind<Vec::Cast<float, T, 0>, OpCopyIn0>;
    using CastIn1 = Bind<Vec::Cast<float, T, 0>, OpCopyIn1>;
    using OpFastGeluGrad = Bind<FastGeluGradCustom<float>, CastIn0, CastIn1>;
    using CastOut1 = Bind<Vec::Cast<T, float, 1>, OpFastGeluGrad>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastOut1>; // dx
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using OpDag = DAGSch<Outputs>;
};

template <typename T>
struct FastGeluGradNoCast {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;                   // dy
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;                   // x
    using OpFastGeluGrad = Bind<FastGeluGradCustom<float>, OpCopyIn0, OpCopyIn1>;  // compute
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpFastGeluGrad>; // dx
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using OpDag = DAGSch<Outputs>;
};

};     // namespace FastGeluGradDag
#endif // CANN_CUSTOM_OPS_FAST_GELU_GRAD_DAG_H
