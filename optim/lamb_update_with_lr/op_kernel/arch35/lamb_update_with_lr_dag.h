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
 * \file    lamb_update_with_lr_dag.h
 * \brief lamb_update_with_lr_dag head file
 */

#ifndef LAMB_UPDATE_WITH_LR_DAG_H
#define LAMB_UPDATE_WITH_LR_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace LambUpdateWithLrOp {
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
// In : 0 input_greater1 1 input_greater_realdiv 2 input_realdiv 3 input_mul0(lr) 4 input_mul1(update, full)
//      5 input_sub(param, full) 6 greater_y 7 select_e 8 minimum_y   [0,1,2,3,6,7,8 scalar]
// Out: 0 y. ratio = where(in0>gy, in1/in2, se) gated again by where(in1>gy, ., se);
//      clip = max(min(ratio, minimum_y), greater_y); y = param - clip*lr*update. Compute in U (=float).
template <typename T, typename U>
struct LambUpdateWithLrCompute {
    // 全部输入统一走 Vec::CopyInBrc: 这些"系数"输入在 A2 的声明就是可广播的 ND Tensor
    // (IR 注释为 "A ND Tensor"、op_info shape 为 all、infershape 走 Broadcast、impl 全程
    // broadcast_shapes), 故按支持范围对齐 A2 补齐广播语义; 融合场景传 (1,) 时行为不变。
    using InG1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InGRd = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InRd = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using InLr = Bind<Vec::CopyInBrc<T>, Placeholder::In3<T>>;
    using InUpd = Bind<Vec::CopyInBrc<T>, Placeholder::In4<T>>;
    using InParam = Bind<Vec::CopyInBrc<T>, Placeholder::In5<T>>;
    using InGreaterY = Bind<Vec::CopyInBrc<T>, Placeholder::In6<T>>;
    using InSelectE = Bind<Vec::CopyInBrc<T>, Placeholder::In7<T>>;
    using InMinY = Bind<Vec::CopyInBrc<T>, Placeholder::In8<T>>;

    using G1 = Bind<Vec::Cast<U, T, 0>, InG1>;
    using GRd = Bind<Vec::Cast<U, T, 0>, InGRd>;
    using Rd = Bind<Vec::Cast<U, T, 0>, InRd>;
    using Lr = Bind<Vec::Cast<U, T, 0>, InLr>;
    using Upd = Bind<Vec::Cast<U, T, 0>, InUpd>;
    using Param = Bind<Vec::Cast<U, T, 0>, InParam>;
    using GreaterY = Bind<Vec::Cast<U, T, 0>, InGreaterY>;
    using SelectE = Bind<Vec::Cast<U, T, 0>, InSelectE>;
    using MinY = Bind<Vec::Cast<U, T, 0>, InMinY>;

    using Greater0 = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, G1, GreaterY>;
    using Greater1 = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, GRd, GreaterY>;
    using RealDiv0 = Bind<DivFtzFalse<U>, GRd, Rd>;
    using Select0 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, Greater0, RealDiv0, SelectE>;
    using Select1 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_T_T>, Greater1, Select0, SelectE>;
    using Minimum0 = Bind<Vec::Min<U>, Select1, MinY>;
    using Maximum0 = Bind<Vec::Max<U>, Minimum0, GreaterY>;

    using Res = Bind<Vec::Sub<U>, Param, Bind<Vec::Mul<U>, Bind<Vec::Mul<U>, Maximum0, Lr>, Upd>>;
    using YCast = Bind<Vec::Cast<T, U, 1>, Res>;

    using OpOut0 = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, YCast>;

    using Outputs = Elems<OpOut0>;
    // LEVEL_0 = 交给框架按 32 buffer 预算自动择档(LEVEL_2->LEVEL_1->LEVEL_0 取第一个装得下的),
    // 不手工钉死档位: 本族算子输入个数差异大(5~13), 手拍的档位对大输入算子会触发
    // dag.h 的 (mte2+mte3)*BUF_PING_PONG+tmp <= 32 静态断言而编译失败。
    using MemCfg = MemOptCfg<MemLevel::LEVEL_0>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LambUpdateWithLrOp
#endif // LAMB_UPDATE_WITH_LR_DAG_H
