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
 * \file    lamb_apply_weight_assign_dag.h
 * \brief lamb_apply_weight_assign_dag head file
 */

#ifndef LAMB_APPLY_WEIGHT_ASSIGN_DAG_H
#define LAMB_APPLY_WEIGHT_ASSIGN_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LambApplyWeightAssignOp {
using namespace AscendC;
using namespace Ops::Base;

// Vec::Div / Vec::Sqrt 的默认档(Div/SqrtAlgo::INTRINSIC)会把非规格化操作数冲刷成 ±0,
// 两个非规格化数相除因此退化成 0/0 = NaN。此处显式取 PRECISION_0ULP_FTZ_FALSE 不冲刷,
// 与竞品(torch 在 CPU/A100 上全程保留 fp32 非规格化数)及本族手写 regbase 算子一致。
// 就地定义而非抽公共头: 本文件同时被 op_host(源码树)与 kernel(构建树)包含, 两棵树相对路径不同。
template <class T>
struct DivFtzFalse : public Ops::Base::Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline DivFtzFalse(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count)
    {
#ifdef __CCE_AICORE__
        static constexpr AscendC::DivConfig config = {AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
        AscendC::Div<T, config>(dst, src1, src2, count);
#endif
    }
};
constexpr int COMPARE_MODE_GT = 1;
constexpr int SELECT_MODE_T_T = 2;
// In : 0 input0(w_norm, scalar), 1 input1(g_norm, scalar), 2 input2(lr, scalar),
//      3 input3(update, full), 4 input_param(param, full)
// Out: 0 input_param(next_param, in-place)
// ratio = where(w_norm>0, where(g_norm>0, w_norm/g_norm, 1), 1); next_param = param - ratio*(update*lr).
// Compute in U (=float): half casts to float (div needs float precision); fp32 native.
template <typename T, typename U>
struct LambApplyWeightAssignCompute {
    using CZero = MAKE_CONST(U, 0.0);
    using COne = MAKE_CONST(U, 1.0);

    // 全部输入统一走 Vec::CopyInBrc: 这些"系数"输入在 A2 的声明就是可广播的 ND Tensor
    // (IR 注释为 "A ND Tensor"、op_info shape 为 all、infershape 走 Broadcast、impl 全程
    // broadcast_shapes), 故按支持范围对齐 A2 补齐广播语义; 融合场景传 (1,) 时行为不变。
    using InWNorm = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InGNorm = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InLr = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using InUpdate = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;
    using InParam = Bind<Vec::CopyInBrc<T>, Placeholder::In4<T>>;

    using WNorm = Bind<Vec::Cast<U, T, 0>, InWNorm>;
    using GNorm = Bind<Vec::Cast<U, T, 0>, InGNorm>;
    using Lr = Bind<Vec::Cast<U, T, 0>, InLr>;
    using Update = Bind<Vec::Cast<U, T, 0>, InUpdate>;
    using Param = Bind<Vec::Cast<U, T, 0>, InParam>;

    // trust ratio: where(w_norm>0, where(g_norm>0, w_norm/g_norm, 1), 1)
    using OneTensor = Bind<Vec::Duplicate<U>, COne>;
    using GreaterG = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, GNorm, CZero>;
    using GreaterW = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, WNorm, CZero>;
    using WDivG = Bind<DivFtzFalse<U>, WNorm, GNorm>;
    using Select1 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, GreaterG, WDivG, OneTensor>;
    using Ratio = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, GreaterW, Select1, OneTensor>;

    // next_param = param - ratio * (update * lr)
    using UpdLr = Bind<Vec::Mul<U>, Update, Lr>;
    using RatioUpdLr = Bind<Vec::Mul<U>, Ratio, UpdLr>;
    using NextParam = Bind<Vec::Sub<U>, Param, RatioUpdLr>;
    using NextParamCast = Bind<Vec::Cast<T, U, 1>, NextParam>;

    using OpCopyOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, NextParamCast>;

    using Outputs = Elems<OpCopyOut0>;
    // LEVEL_0 = 交给框架按 32 buffer 预算自动择档(LEVEL_2->LEVEL_1->LEVEL_0 取第一个装得下的),
    // 不手工钉死档位: 本族算子输入个数差异大(5~13), 手拍的档位对大输入算子会触发
    // dag.h 的 (mte2+mte3)*BUF_PING_PONG+tmp <= 32 静态断言而编译失败。
    using MemCfg = MemOptCfg<MemLevel::LEVEL_0>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LambApplyWeightAssignOp
#endif // LAMB_APPLY_WEIGHT_ASSIGN_DAG_H
