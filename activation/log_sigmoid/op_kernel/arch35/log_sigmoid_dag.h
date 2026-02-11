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
 * \file log_sigmoid_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_LOG_SIGMOID_DAG_H
#define CANN_CUSTOM_OPS_LOG_SIGMOID_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
namespace LogSigmoidDag {

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;

template <class T>
struct LogSigmoidCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline LogSigmoidCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t VL = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, VL);
        uint32_t vlSize = VL;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        T valPosOne = 1.0;
        T valNegOne = -1.0;
        T valZero = 0.0;

        __VEC_SCOPE__
        {
            // init vars
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> x;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> zeroReg;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> xAbs;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> xAbsNeg;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> expRes;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> expResPlusOne;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> expResPlusOneSubOne;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> logExpXPlus1;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> divRes;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> mulRes;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> minRes;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> selectRes;
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> ans;

            MicroAPI::MaskReg mask;
            MicroAPI::MaskReg cmpLog1pPosMaskReg;

            MicroAPI::Duplicate(zeroReg, valZero);

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                // regCopyIn
                mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(count);
                MicroAPI::DataCopy(x, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                MicroAPI::Min(minRes, x, zeroReg, mask);        // x1 = min(x, 0)
                MicroAPI::Abs(xAbs, x, mask);                   // x2 = abs(x)
                MicroAPI::Muls(xAbsNeg, xAbs, valNegOne, mask); // x3 = -x2
                MicroAPI::Exp(expRes, xAbsNeg, mask);           // x4 = e^x3
                // log1p
                MicroAPI::Adds(expResPlusOne, expRes, valPosOne, mask);              // y1 = 1 + x4
                MicroAPI::Adds(expResPlusOneSubOne, expResPlusOne, valNegOne, mask); // y2 = y1 - 1
                MicroAPI::Div(divRes, expRes, expResPlusOneSubOne, mask);            // y3 = x4 / y2
                MicroAPI::Log(logExpXPlus1, expResPlusOne, mask);                    // y4 = log(y1)
                MicroAPI::Mul(mulRes, logExpXPlus1, divRes, mask);                   // y5 = y4 * y3
                MicroAPI::CompareScalar<T, CMPMODE::NE>(cmpLog1pPosMaskReg, expResPlusOne, valPosOne, mask);
                MicroAPI::Select(selectRes, mulRes, expRes, cmpLog1pPosMaskReg); // z1 = select(x4, y5)
                MicroAPI::Sub(ans, minRes, selectRes, mask);                     // z2 = x1 - z1

                // regCopyOut
                MicroAPI::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), ans, mask);
            }
        }
#endif
    }
};

template<typename T> 
struct LogSigmoidNoCast {
    // 通过Compute构造计算图
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpLogSigmoid = Bind<LogSigmoidCustom<float>, OpCopyIn>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpLogSigmoid>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using OpDag = DAGSch<Outputs>;
};

template<typename T> 
 struct LogSigmoidNeedCast {
    // 通过Compute构造计算图
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>; // x
    using CastIn = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, OpCopyIn>;
    using OpLogSigmoid = Bind<LogSigmoidCustom<float>, CastIn>;

    using CastOut = Bind<Vec::Cast<T, float, CAST_MODE_RINT>, OpLogSigmoid>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastOut>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using OpDag = DAGSch<Outputs>;
};

};     // namespace LogSigmoidDag
#endif // CANN_CUSTOM_OPS_LOG_SIGMOID_DAG_H