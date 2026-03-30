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
 * \file erfinv_dag.h
 * \brief
 */

#ifndef OP_NN_ACTIVATION_ERFINV_DAG_H
#define OP_NN_ACTIVATION_ERFINV_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace ErfinvOp {
using namespace Ops::Base;

const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;
constexpr uint32_t THREAD_NUM = 1024;

#ifdef __CCE_AICORE__
template <typename T>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void ErfinvSimtCompute(__ubuf__ T* x, __ubuf__ T* y, const int64_t totalNum)
{
    for (int64_t i = Simt::GetThreadIdx(); i < totalNum; i += Simt::GetThreadNum()) {
        y[i] = Simt::Erfinv(x[i]);
    }
}
#endif

template <class T>
struct ErfinvCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline ErfinvCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        Simt::VF_CALL<ErfinvSimtCompute<T>>(Simt::Dim3(THREAD_NUM), srcAddr, dstAddr, count);
#endif
    }
};

template <typename U, typename T = float>
struct ErfinvDAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpResult1 = Bind<ErfinvCustom<T>, OpCopyIn0Cast>;
    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResult1>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ErfinvOp
#endif // OP_NN_ACTIVATION_ERFINV_DAG_H