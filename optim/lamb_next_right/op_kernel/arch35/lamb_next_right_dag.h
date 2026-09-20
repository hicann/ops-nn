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
 * \file    lamb_next_right_dag.h
 * \brief lamb_next_right_dag head file
 */

#ifndef LAMB_NEXT_RIGHT_DAG_H
#define LAMB_NEXT_RIGHT_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LambNextRightOp {
using namespace AscendC;
using namespace Ops::Base;

// Vec::Div / Vec::Sqrt 的默认档(Div/SqrtAlgo::INTRINSIC)会把非规格化操作数冲刷成 ±0,
// 两个非规格化数相除因此退化成 0/0 = NaN。此处显式取 PRECISION_0ULP_FTZ_FALSE 不冲刷,
// 与竞品(torch 在 CPU/A100 上全程保留 fp32 非规格化数)及本族手写 regbase 算子一致。
// 就地定义而非抽公共头: 本文件同时被 op_host(源码树)与 kernel(构建树)包含, 两棵树相对路径不同。
template <class T>
struct SqrtFtzFalse : public Ops::Base::Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline SqrtFtzFalse(LocalTensor<T>& dst, LocalTensor<T>& src, int count)
    {
#ifdef __CCE_AICORE__
        static constexpr AscendC::SqrtConfig config = {AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
        AscendC::Sqrt<T, config>(dst, src, count);
#endif
    }
};
// In : 0 input_square(g, full) 1 input_mul2(v, full) 2 mul2_x(b2) 3 mul3_x(1-b2)
//      4 truediv1_recip(1/(1-b2^t)) 5 add2_y(eps)   [2..5 scalar]
// Out: 0 y1(next_v = v*b2 + g^2*(1-b2)) 1 y2(sqrt(next_v/(1-b2^t)) + eps)
// Compute in U (=float): half casts to float (sqrt needs float precision).
template <typename T, typename U>
struct LambNextRightCompute {
    // 全部输入统一走 Vec::CopyInBrc: 这些"系数"输入在 A2 的声明就是可广播的 ND Tensor
    // (IR 注释为 "A ND Tensor"、op_info shape 为 all、infershape 走 Broadcast、impl 全程
    // broadcast_shapes), 故按支持范围对齐 A2 补齐广播语义; 融合场景传 (1,) 时行为不变。
    using InG = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InV = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InB2 = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using InOmB2 = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;
    using InRecip = Bind<Vec::CopyInBrc<T>, Placeholder::In4<T>>;
    using InEps = Bind<Vec::CopyInBrc<T>, Placeholder::In5<T>>;

    using G = Bind<Vec::Cast<U, T, 0>, InG>;
    using V = Bind<Vec::Cast<U, T, 0>, InV>;
    using B2 = Bind<Vec::Cast<U, T, 0>, InB2>;
    using OmB2 = Bind<Vec::Cast<U, T, 0>, InOmB2>;
    using Recip = Bind<Vec::Cast<U, T, 0>, InRecip>;
    using Eps = Bind<Vec::Cast<U, T, 0>, InEps>;

    // next_v = v*b2 + g^2*(1-b2)
    using NextV = Bind<Vec::Add<U>, Bind<Vec::Mul<U>, V, B2>, Bind<Vec::Mul<U>, Bind<Vec::Mul<U>, G, G>, OmB2>>;
    // y2 = sqrt(next_v * recip) + eps
    using Y2 = Bind<Vec::Add<U>, Bind<SqrtFtzFalse<U>, Bind<Vec::Mul<U>, NextV, Recip>>, Eps>;

    using Y1Cast = Bind<Vec::Cast<T, U, 1>, NextV>;
    using Y2Cast = Bind<Vec::Cast<T, U, 1>, Y2>;

    using OpOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y1Cast>;
    using OpOut1 = Bind<Vec::CopyOut<T>, Placeholder::Out1<T>, Y2Cast>;

    using Outputs = Elems<OpOut0, OpOut1>;
    // LEVEL_0 = 交给框架按 32 buffer 预算自动择档(LEVEL_2->LEVEL_1->LEVEL_0 取第一个装得下的),
    // 不手工钉死档位: 本族算子输入个数差异大(5~13), 手拍的档位对大输入算子会触发
    // dag.h 的 (mte2+mte3)*BUF_PING_PONG+tmp <= 32 静态断言而编译失败。
    using MemCfg = MemOptCfg<MemLevel::LEVEL_0>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LambNextRightOp
#endif // LAMB_NEXT_RIGHT_DAG_H
