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
 * \file binary_cross_entropy_dag.h
 * \brief binary_cross_entropy_dag head file
 */
#ifndef ASCENDC_BINARY_CROSS_ENTROPY_DAG_H_
#define ASCENDC_BINARY_CROSS_ENTROPY_DAG_H_
#include <cmath>
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

#ifdef __CCE_AICORE__
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#endif

namespace AscendC {
namespace Vec {
#ifdef __CCE_AICORE__
using AscendC::MicroAPI::RegTensor;

constexpr static uint16_t VECTOR_LENGTH = Ops::Base::GetVRegSize();

template <typename U = float>
__aicore__ inline void CalcYLogX(MicroAPI::RegTensor<U> &regYLogX, MicroAPI::RegTensor<U> &regX, MicroAPI::RegTensor<U> &regY,
                                 MicroAPI::MaskReg &pregUp)
{
    MicroAPI::RegTensor<U> regLogX;
    MicroAPI::RegTensor<U> regClipLogX;
    MicroAPI::Log(regLogX, regX, pregUp);
    MicroAPI::Maxs(regClipLogX, regLogX, (U)-100.0, pregUp);
    MicroAPI::Mul(regYLogX, regClipLogX, regY, pregUp);
}
#endif
template <typename U = float>
struct CalcBinaryYLogX : public Ops::Base::Vec::ElemwiseBinaryOP<U, U, U> {
    __aicore__ inline CalcBinaryYLogX(Ops::Base::LocalTensor<U>& yLogX, Ops::Base::LocalTensor<U>& x, Ops::Base::LocalTensor<U>& y, int32_t count) {
#ifdef __CCE_AICORE__

        uint32_t oneRepeat = VECTOR_LENGTH / sizeof(U);
        uint32_t totalLen = count;
        uint32_t repeatTimes = Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);

        __ubuf__ U * xAddr = (__ubuf__ U *)x.GetPhyAddr();
        __ubuf__ U * yAddr = (__ubuf__ U *)y.GetPhyAddr();
        __ubuf__ U * yLogXTAddr = (__ubuf__ U *)yLogX.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::MaskReg pregUp;
            MicroAPI::RegTensor<U> regX;
            MicroAPI::RegTensor<U> regY;
            MicroAPI::RegTensor<U> regTensorOne;
            MicroAPI::RegTensor<U> regOneSubX;
            MicroAPI::RegTensor<U> regOneSubY;
            MicroAPI::RegTensor<U> regOneSubYLogX;
            MicroAPI::RegTensor<U> regYLogX;
            MicroAPI::RegTensor<U> regDataSum;
            MicroAPI::RegTensor<U> regBinaryYLogX;

            for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
                pregUp = MicroAPI::UpdateMask<U>(totalLen);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regX, xAddr, (int32_t)oneRepeat);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regY, yAddr, (int32_t)oneRepeat);
                
                CalcYLogX<U>(regYLogX, regX, regY, pregUp);
                MicroAPI::Duplicate(regTensorOne, (U)1.0f, pregUp);
                MicroAPI::Sub(regOneSubX, regTensorOne, regX, pregUp);
                MicroAPI::Sub(regOneSubY, regTensorOne, regY, pregUp);
                CalcYLogX<U>(regOneSubYLogX, regOneSubX, regOneSubY, pregUp);
                MicroAPI::Add(regDataSum, regYLogX, regOneSubYLogX, pregUp);
                MicroAPI::Muls(regBinaryYLogX, regDataSum, (U)-1.0f, pregUp);

                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(yLogXTAddr, regBinaryYLogX, (int32_t)oneRepeat, pregUp);
            }
        }
#endif
    }
};
}
}

namespace BinaryCrossEntropy {

template <typename U, typename T = float>
struct BCEHasWeightElewise {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpCopyY = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In1<U>>;
    using OpCopyWeight = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In2<U>>;
    using OpCopyXCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyY>;
    using OpCopyWeightCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyWeight>;
    using OpResNeg = Ops::Base::Bind<AscendC::Vec::CalcBinaryYLogX<T>, OpCopyXCast, OpCopyYCast>;
    // -weight * (y * log(x) + (1 - y) * log(1 - x))
    using OpResCast = Ops::Base::Bind<Ops::Base::Vec::Mul<T>, OpResNeg, OpCopyWeightCast>;
    using OpRes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_RINT>, OpResCast>;

    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpRes>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;

    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEElewise {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpCopyY = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In1<U>>;
    using OpCopyXCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyY>;
    // -(y * log(x) + (1 - y) * log(1 - x))
    using OpResNeg = Ops::Base::Bind<AscendC::Vec::CalcBinaryYLogX<T>, OpCopyXCast, OpCopyYCast>;
    using OpRes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_RINT>, OpResNeg>;

    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpRes>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;

    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCESumDag {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpCopyY = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In1<U>>;
    using OpCopyXCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyY>;
    // -(y * log(x) + (1 - y) * log(1 - x))
    using OpMulRes = Ops::Base::Bind<AscendC::Vec::CalcBinaryYLogX<T>, OpCopyXCast, OpCopyYCast>;
    using ReduceOpRes = Ops::Base::Bind<Ops::Base::Vec::ReduceSumOp<T>, OpMulRes>;
    using OpRes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_RINT>, ReduceOpRes>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpRes>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEMeanDag {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpCopyY = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In1<U>>;
    using OpCopyXCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyY>;
    // -(y * log(x) + (1 - y) * log(1 - x))
    using OpMulRes = Ops::Base::Bind<AscendC::Vec::CalcBinaryYLogX<T>, OpCopyXCast, OpCopyYCast>;
    using OpMeanRes = Ops::Base::Bind<Ops::Base::Vec::Muls<T>, OpMulRes, Ops::Base::Placeholder::Var<T, 0>>;
    using ReduceOpRes = Ops::Base::Bind<Ops::Base::Vec::ReduceSumOp<T>, OpMeanRes>;
    using OpRes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_RINT>, ReduceOpRes>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpRes>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEHasWeightSumDag {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpCopyY = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In1<U>>;
    using OpCopyWeight = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In2<U>>;
    using OpCopyXCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyY>;
    using OpCopyWeightCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyWeight>;
    using OpResNeg = Ops::Base::Bind<AscendC::Vec::CalcBinaryYLogX<T>, OpCopyXCast, OpCopyYCast>;
    // -weight * (y * log(x) + (1 - y) * log(1 - x))
    using OpMulRes = Ops::Base::Bind<Ops::Base::Vec::Mul<T>, OpResNeg, OpCopyWeightCast>;
    using ReduceOpRes = Ops::Base::Bind<Ops::Base::Vec::ReduceSumOp<T>, OpMulRes>;
    using OpRes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_RINT>, ReduceOpRes>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpRes>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct BCEHasWeightMeanDag {
    constexpr static int CAST_MODE_RINT = 1;
    using OpCopyX = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpCopyY = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In1<U>>;
    using OpCopyWeight = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In2<U>>;
    using OpCopyXCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyX>;
    using OpCopyYCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyY>;
    using OpCopyWeightCast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, 0>, OpCopyWeight>;
    using OpResNeg = Ops::Base::Bind<AscendC::Vec::CalcBinaryYLogX<T>, OpCopyXCast, OpCopyYCast>;
    // -weight * (y * log(x) + (1 - y) * log(1 - x))
    using OpMulRes = Ops::Base::Bind<Ops::Base::Vec::Mul<T>, OpResNeg, OpCopyWeightCast>;
    using OpMeanRes = Ops::Base::Bind<Ops::Base::Vec::Muls<T>, OpMulRes, Ops::Base::Placeholder::Var<T, 0>>;
    using ReduceOpRes = Ops::Base::Bind<Ops::Base::Vec::ReduceSumOp<T>, OpMeanRes>;
    using OpRes = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_RINT>, ReduceOpRes>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpRes>;
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};
} // namespace BinaryCrossEntropy
#endif //ASCENDC_BINARY_CROSS_ENTROPY_DAG_H_