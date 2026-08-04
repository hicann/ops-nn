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
 * \file sgd_dag.h
 * \brief SGD 的 ATVOSS DAG 定义（arch35 / Ascend950 / regbase）
 */

#ifndef SGD_DAG_H
#define SGD_DAG_H

#include <type_traits>
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace SgdOp {
using namespace Ops::Base;

// Var 索引：DAG 中 Var 节点随分支存废，索引必须从 0 起连续无空洞
// （sch.SetVar<U, index> 带 static_assert(index < ElemDag::Vars::Size)）。
constexpr int SGD_VAR_IDX_WEIGHT_DECAY = 0;

/**
 * SGD DAG。
 *
 * 计算语义（逐元素，T = float 域；d = dampening、wd = weight_decay、
 * lr = learning_rate[0]、m = momentum[0]）：
 *   1. grad     = gradient + parameters * wd        仅 hasWeightDecay
 *   2. accum_t  = accum * m + grad                  【无条件】
 *   3. accum_t -= grad * ((1 - stat) * d)           仅 hasDampening
 *   4. parameters_out = nesterov ? p - (grad + accum_t*m) * lr
 *                                : p - accum_t*lr   【无条件写出】
 *   5. accum_out = accum_t、stat_out = 0            仅 doWriteback（即 m != 0）
 *
 * 模板参数 doWriteback 承载 `momentum != 0` 掩码：
 *   - true  —— 回写 DAG，Outputs 三路，sch.Init 输出位 3 个；
 *   - false —— 掩码 DAG，Outputs 只剩 OpCopyOutParam，accum / stat 的 GM
 *              【从不出现在 sch.Init 的输出位】→ 框架 InitOutputArgs 不为其构造
 *              outGm[]、CopyOut 路径根本不存在 → 零写事务 → 逐位保持输入原值
 *              对任意位模式（NaN payload / ±inf / -0.0）平凡成立。
 *
 * ⛔ 掩码分支【不得】删掉 In3(accum) / In4(momentum) / In5(stat)：
 *    accum_t 是 parameters_out 的上游，必须照常计算；且 accum 含 ±inf 时
 *    0 * inf = NaN 须按 IEEE 传播进 parameters_out。四套布局输入 holder 完全相同。
 *
 * bf16 的两处特殊处理（先例 optim/apply_ftrl/op_kernel/arch35/apply_ftrl_dag.h）：
 *    ① 标量输入：非 bf16 走 TensorScalar（Vec::CopyIn + ScalarAttr<true>，不占 UB）；
 *       bf16 回退 Vec::Duplicate（BufferNum +2）。
 *    ② 消费算子随之【成对切换】：TensorScalar 产出的是标量，消费者必须是标量变体
 *       Vec::Muls；Duplicate 产出的是张量，消费者必须是 Vec::Mul。两者不可错配。
 *
 * ⚠️【TensorScalar 单槽位约束】—— 框架限制，决定了 Step 4 nesterov 的写法：
 *    ATVOSS 的 ScalarOp 值容器 ScalarOpType（util/node.h: ScalarOpNodes::Export<VarTypeAux>）
 *    大小 = ScalarOp 节点【个数】，但 elewise_sch_with_scalar.h 用 【FunList 下标】存取
 *    （Set<pos> :200/:475、Get<GetFunOutputPos<InputOp>()> :329）；而 placeholder.h:290
 *    的单元素特化 VarTypeStruct<T>::Get/Set 直接【忽略 offset】，于是所有越界下标静默
 *    塌缩到最后一个槽位 —— 本 DAG 的 lr / momentum 共用同一个物理槽，"最后写入者获胜"。
 *    ⇒ 约束：某个 TensorScalar 的【全部消费者】必须排在下一个 TensorScalar 的 CopyIn
 *      之前（FunList 由 Outputs 反向 DFS、InFuns 从左到右后序生成，顺序是确定的）。
 *    本 DAG 满足该约束的方式：lr 全程【只被消费一次】且位于最末（OpAccTMulLr /
 *      OpNesterovMulLr），momentum 的消费者全在其左子树内。
 *    ⛔ 因此 nesterov【不得】展开成 p - (grad*lr + accum_t*m*lr)：那样 OpGradMulLr 会
 *      把 lr 的 CopyIn 提前到 momentum 之前，末尾的 OpAccTMomMulLr 再读槽位时拿到的是
 *      momentum，device 实测结果退化为 p - (grad*lr + accum_t*m*m)（bf16 走 Duplicate
 *      不入该容器，故只有 fp16/fp32 中招；momentum==0 时 nesterov≡plain 亦不暴露）。
 */
template <typename U, bool useNesterov, bool hasWeightDecay, bool hasDampening, bool doWriteback, typename T = float>
struct SgdDag {
    static constexpr bool IS_BF16 = std::is_same<U, bfloat16_t>::value;
    // hasWeightDecay 为 false 时 Var<T,0> 让给 dampening，避免索引空洞
    static constexpr int VAR_IDX_DAMPENING = hasWeightDecay ? 1 : 0;

    // ── 输入 holder（顺序与图原型一致：parameters / gradient / learning_rate /
    //                 accum / momentum / stat）────────────────────────────────
    using OpCopyInParam = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpCopyInLr = std::conditional_t<IS_BF16,
                                          Bind<Vec::Duplicate<U>, Placeholder::In2<U, Placeholder::ScalarAttr<true>>>,
                                          Bind<Vec::CopyIn<U>, Placeholder::In2<U, Placeholder::ScalarAttr<true>>>>;
    using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;
    using OpCopyInMom = std::conditional_t<IS_BF16,
                                           Bind<Vec::Duplicate<U>, Placeholder::In4<U, Placeholder::ScalarAttr<true>>>,
                                           Bind<Vec::CopyIn<U>, Placeholder::In4<U, Placeholder::ScalarAttr<true>>>>;
    // In5 仅 hasDampening == true 时进入 DAG 闭包；hasDampening == false 时
    // 本 typedef 不被 Outputs 可达，DAGSch 不会收录，输入退化为 5 路（无空洞）。
    using OpCopyInStat = Bind<Vec::CopyIn<U>, Placeholder::In5<U>>;

    // ── 升 float32 域 ───────────────────────────────────────────────────────
    using OpParamF = Bind<Vec::Cast<T, U, 0>, OpCopyInParam>;
    using OpGradInF = Bind<Vec::Cast<T, U, 0>, OpCopyInGrad>;
    using OpLrF = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
    using OpAccumF = Bind<Vec::Cast<T, U, 0>, OpCopyInAccum>;
    using OpMomF = Bind<Vec::Cast<T, U, 0>, OpCopyInMom>;
    using OpStatF = Bind<Vec::Cast<T, U, 0>, OpCopyInStat>;

    // ── Step 1：权重衰减 grad = gradient + parameters * wd ──────────────────
    // wd == 0 时【真正跳过】而不是乘 0（spec numerical_stability.skip_zero_branches；
    // 且 0 * inf = NaN 会污染结果）。
    using VarWeightDecay = Placeholder::Var<T, SGD_VAR_IDX_WEIGHT_DECAY>;
    using OpParamMulWd = Bind<Vec::Muls<T>, OpParamF, VarWeightDecay>;
    using OpGradWithWd = Bind<Vec::Add<T>, OpGradInF, OpParamMulWd>;
    using OpGrad = std::conditional_t<hasWeightDecay, OpGradWithWd, OpGradInF>;

    // ── Step 2：动量累积 accum_t = accum * m + grad（无条件）────────────────
    using OpAccMulMom = std::conditional_t<IS_BF16, Bind<Vec::Mul<T>, OpAccumF, OpMomF>,
                                           Bind<Vec::Muls<T>, OpAccumF, OpMomF>>;
    using OpAccumTBase = Bind<Vec::Add<T>, OpAccMulMom, OpGrad>;

    // ── Step 3：阻尼修正 accum_t -= grad * ((1 - stat) * d) ─────────────────
    using ConstNegOne = MAKE_CONST(T, -1);
    using ConstOne = MAKE_CONST(T, 1);
    using OpStatNeg = Bind<Vec::Muls<T>, OpStatF, ConstNegOne>;
    using OpStatAct = Bind<Vec::Adds<T>, OpStatNeg, ConstOne>; // 1 - stat
    using VarDampening = Placeholder::Var<T, VAR_IDX_DAMPENING>;
    using OpStatActMulD = Bind<Vec::Muls<T>, OpStatAct, VarDampening>;
    using OpDampTerm = Bind<Vec::Mul<T>, OpGrad, OpStatActMulD>;
    using OpAccumTDamped = Bind<Vec::Sub<T>, OpAccumTBase, OpDampTerm>;
    using OpAccumT = std::conditional_t<hasDampening, OpAccumTDamped, OpAccumTBase>;

    // ── Step 4：权重更新（无条件写出）───────────────────────────────────────
    // 非 nesterov：parameters - accum_t * lr
    using OpAccTMulLr = std::conditional_t<IS_BF16, Bind<Vec::Mul<T>, OpAccumT, OpLrF>,
                                           Bind<Vec::Muls<T>, OpAccumT, OpLrF>>;
    using OpParamPlain = Bind<Vec::Sub<T>, OpParamF, OpAccTMulLr>;

    // nesterov：parameters - (grad + accum_t * m) * lr
    // ⚠️【不得】改写成展开式 p - (grad*lr + accum_t*m*lr)：见文件头「TensorScalar 单槽位」约束。
    using OpAccTMulMom = std::conditional_t<IS_BF16, Bind<Vec::Mul<T>, OpAccumT, OpMomF>,
                                            Bind<Vec::Muls<T>, OpAccumT, OpMomF>>;
    using OpNesterovSum = Bind<Vec::Add<T>, OpGrad, OpAccTMulMom>; // grad + accum_t * m
    using OpNesterovMulLr = std::conditional_t<IS_BF16, Bind<Vec::Mul<T>, OpNesterovSum, OpLrF>,
                                               Bind<Vec::Muls<T>, OpNesterovSum, OpLrF>>;
    using OpParamNesterov = Bind<Vec::Sub<T>, OpParamF, OpNesterovMulLr>;

    using OpParamNew = std::conditional_t<useNesterov, OpParamNesterov, OpParamPlain>;
    using OpParamOutCast = Bind<Vec::Cast<U, T, 1>, OpParamNew>; // CAST_RINT：就近偶数舍入
    using OpCopyOutParam = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpParamOutCast>; // output: parameters

    // ── Step 5：momentum != 0 掩码控制的两路回写 ────────────────────────────
    using OpAccumOutCast = Bind<Vec::Cast<U, T, 1>, OpAccumT>;
    using OpCopyOutAccum = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpAccumOutCast>; // 原地回写 input3: accum

    using ConstZero = MAKE_CONST(T, 0);
    using OpZeroTsr = Bind<Vec::Duplicate<T>, ConstZero>;
    using OpStatOutCast = Bind<Vec::Cast<U, T, 1>, OpZeroTsr>;
    using OpCopyOutStat = Bind<Vec::CopyOut<U>, Placeholder::Out2<U>, OpStatOutCast>; // 原地回写 input5: stat

    // Outputs 是两套 DAG 的【唯一】差异行。掩码 DAG 下 OpZeroTsr / OpCopyOutAccum /
    // OpCopyOutStat 随 Elems 收缩被 DAGSch 从 Outputs 反向推导时整体裁掉，
    // BufferNum 比回写 DAG 小 5~6（Host 按回写 DAG 反解 ubFormer，故不会溢出）。
    using Outputs = std::conditional_t<doWriteback, Elems<OpCopyOutParam, OpCopyOutAccum, OpCopyOutStat>,
                                       Elems<OpCopyOutParam>>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace SgdOp

#endif // SGD_DAG_H
