/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// LpNormReduce_package/op_graph/lp_norm_reduce_v1_infer.cpp
// =============================================================================
//
// ROLE: V1 (Runtime 1.0) graph-mode shape/dtype inference & parameter checking
//   for the LpNormReduce operator.
//   本文件提供 V1 通道注册（GEIR 图模式 shape/dtype 推导与校验）：
//   - 背景：本轮 GEIR 实测（tests/geir/logs/negative-source-static.log）证明
//     GE 图编译（AddGraph/RunGraph）对自定义算子的 shape 推导走 V1 通道
//     （op_desc_utils_ex.cc `CallInferFunc:Op op1[LpNormReduce] Call
//     InferShapeFuncV1`），V2 侧 IMPL_OP_INFERSHAPE / IMPL_OP 注册（加载与
//     `registered first` 证据完整）不参与该调用。V1 通道按 REG_OP 声明查
//     InferShapeFuncRegister 全局注册表；本文件把校验 + 推导以老式注册
//     （IMPLEMT_INFERFUNC + INFER_FUNC_REG，canndev op_proto/math_ops.cc 同款
//     机制）挂入该表，使 GEIR 图模式在推导阶段即执行与 V2 同口径的校验。
//   - 校验（与 docs/LpNormReduce/develop/proto.md 约束、
//     design/InferShapeDtype.md、op_host/arch35 tiling 校验链一致）：
//     - 输入 dtype ∈ {DT_FLOAT16, DT_FLOAT, DT_BF16}；
//     - 输入 format 须 ND（proto.md「Format/UnknownShapeFormat 全 ND」；
//        FRACTAL_NZ 等非法排布在推导阶段拒绝，先于 FE 的排布适配）；
//     - rank(x) ∈ [0, 8]；
//     - p 值域 [0, 2147483647] ∪ {-2147483648}（±inf 整型哨兵特例）；
//     - axes 归一：空列表 → 全部维度；负轴 +rank；越界拒绝，重复轴去重；
//     - p=±inf 哨兵且归约轴长度为 0 → 拒绝（动态编译期未知维 -1 跳过，
//        交运行时 tiling 同口径校验）；
//     - 输出 y 已显式声明 dtype 且与推导（= x.dtype）冲突 → 拒绝
//        （spec.yaml outputs.y dtype_rule：y.dtype = x.dtype 无跨 dtype 组合）。
//   - 推导主体（canndev op_proto/runtime/lp_norm.cc InferShape4LpNorm 同构）：
//     y.shape = reduce_shape(x.shape, axes 非空 ? axes : None, keepdims=keepdim)
//     ——keepdim=false 删归约维、true 置 1；动态 -1 未知维照常透传。
//   - V2 侧注册（op_host/lp_norm_reduce_infershape.cpp、
//     op_graph/lp_norm_reduce_graph_infer.cpp）保持不动：exe_graph
//     通路仍走 V2；两通道校验与推导口径一致。
//
// OPERATOR NAME VARIANTS:
//   PascalCase : LpNormReduce（REG_OP / IMPLEMT_INFERFUNC 注册名）
//   snake_case : lp_norm_reduce（文件名）
//
// =============================================================================

#include <set>
#include <string>
#include <vector>

#include "graph/operator.h"
#include "graph/operator_reg.h"     // IMPLEMT_INFERFUNC / INFER_FUNC_REG
#include "graph/utils/type_utils.h" // ge::TypeUtils dtype/format 序列化
#include "op_common/log/log.h"      // OP_CHECK_IF / OP_LOGE_FOR_INVALID_* macros

#include "lp_norm_reduce_proto.h" // op::LpNormReduce（REG_OP 生成 Operator 子类）

namespace ge {

// p 值域哨兵（与 op_host/arch35/lp_norm_reduce_tiling_arch35.cpp、
// op_host/lp_norm_reduce_infershape.cpp 同口径：spec.yaml attributes.p
// machine_constraint int_in_range [0, 2147483647]，-inf 以整型哨兵 -2147483648
// 表达，canndev CONST_INF 口径）
namespace {
constexpr int64_t INF_POS_SENTINEL = 2147483647;
constexpr int64_t INF_NEG_SENTINEL = -2147483648LL;
constexpr int64_t DEFAULT_P = 2;
constexpr int64_t MAX_RANK = 8;
} // namespace

// ---------------------------------------------------------------------------
// InferShapeAndCheckLpNormReduceV1(op) — V1 shape/dtype inference + checks
//
// 上述全部校验（见文件头）+ canndev 同构推导；校验失败返回 GRAPH_FAILED，
// GE 图编译（AddGraph/RunGraph）在推导阶段即拒绝，测试程序以非零退出
// 上抛（TTK EXEC_FAILURE + ERROR: 闭环）。
// ---------------------------------------------------------------------------
IMPLEMT_INFERFUNC(LpNormReduce, InferShapeAndCheckLpNormReduceV1)
{
    // ── 输入 desc（推导阶段为用户原始声明：FE 的 dtype/排布适配发生在推导之后）──
    const TensorDesc xDesc = op.GetInputDesc("x");
    const DataType xDtype = xDesc.GetDataType();
    const Format xFormat = xDesc.GetFormat();
    const Shape xShape = xDesc.GetShape();
    const std::string nodeNameStr = op.GetName();
    const char* nodeName = nodeNameStr.c_str();

    // ── 校验 dtype ∈ {DT_FLOAT16, DT_FLOAT, DT_BF16}（批 b-6 对齐 canndev 三档）──
    OP_CHECK_IF(xDtype != DT_FLOAT16 && xDtype != DT_FLOAT && xDtype != DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", TypeUtils::DataTypeToSerialString(xDtype),
                                                      "only DT_FLOAT16/DT_FLOAT/DT_BF16 are supported on ascend950"),
                return GRAPH_FAILED);

    // ── 校验输入 format 须 ND（proto.md Format/UnknownShapeFormat 全 ND）──
    OP_CHECK_IF(xFormat != FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(nodeName, "x", TypeUtils::FormatToSerialString(xFormat),
                                                       "only FORMAT_ND is supported on ascend950"),
                return GRAPH_FAILED);

    // ── unknown rank 透传（批 b-3，对齐 changwei/canndev 通行写法）：x 为 [-2]
    //    标记时归约轴无法确定，输出同样标记 unknown rank 后直接返回（下游 GE 按
    //    unknown rank 落图，运行时由 tiling 实际校验）──
    if (xShape.GetDimNum() == 1 && xShape.GetDim(0) == -2) {
        TensorDesc yDescUnknown = op.GetOutputDesc("y");
        yDescUnknown.SetShape(Shape(std::vector<int64_t>{-2}));
        OP_CHECK_IF(op.UpdateOutputDesc("y", yDescUnknown) != GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "failed to update y with unknown-rank shape"), return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }

    // ── 校验 rank(x) ∈ [0, 8]（批 b-5：rank-0 标量放行，空 axes = 单元素全归约）──
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(
        rank > MAX_RANK,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "x", std::to_string(rank), "rank of x must be in [0, 8]"),
        return GRAPH_FAILED);

    // ── 校验输出声明（先于推导）：y 已显式声明 dtype 且 ≠ x.dtype → 拒绝 ──
    const TensorDesc yDeclared = op.GetOutputDesc("y");
    const DataType yDeclaredDtype = yDeclared.GetDataType();
    OP_CHECK_IF(
        yDeclaredDtype != DT_UNDEFINED && yDeclaredDtype != xDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "y", TypeUtils::DataTypeToSerialString(yDeclaredDtype),
                                              "output dtype must be identical to input dtype (y.dtype = x.dtype, "
                                              "no cross-dtype promotion)"),
        return GRAPH_FAILED);

    // ── attrs（proto.md 默认值：p=2 / axes={} / keepdim=false；GetAttr 未设置时回落默认）──
    int64_t p = DEFAULT_P;
    std::vector<int64_t> axes;
    bool keepdim = false;
    if (op.GetAttr("p", p) != GRAPH_SUCCESS) {
        p = DEFAULT_P;
    }
    (void)op.GetAttr("axes", axes);
    (void)op.GetAttr("keepdim", keepdim);

    // ── 校验 p 值域（批 b-1 对齐 canndev 公共契约：p >= 0 完整非负 int64；
    //    2147483647 保留 +inf 哨兵、-2147483648 为 -inf 哨兵）──
    OP_CHECK_IF(
        p < 0 && p != INF_NEG_SENTINEL,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "p", std::to_string(p),
                                              "p must be non-negative (full int64) or -2147483648 (-inf sentinel)"),
        return GRAPH_FAILED);

    // ── axes 归一校验：空 → 全部维度；负轴 +rank；越界拒绝；
    //    重复轴首次出现去重（批 b-2 对齐 canndev/changwei，与 V2 infershape 同口径）──
    std::set<int64_t> reduceSet;
    if (axes.empty()) {
        for (int64_t i = 0; i < rank; i++) {
            reduceSet.insert(i);
        }
    } else {
        for (size_t i = 0; i < axes.size(); i++) {
            const int64_t v = axes[i];
            OP_CHECK_IF(
                v < -rank || v >= rank,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "axes", std::to_string(v),
                                                      "axes[" + std::to_string(i) + "] out of range [-" +
                                                          std::to_string(rank) + ", " + std::to_string(rank) + ")"),
                return GRAPH_FAILED);
            const int64_t norm = (v < 0) ? (v + rank) : v;
            (void)reduceSet.insert(norm); // 重复轴静默去重（首次出现生效）
        }
    }

    // ── 校验 p=±inf 哨兵且归约轴长度为 0 → 拒绝（空集 max/min 无定义，
    //    canndev 约束 2）。动态编译期未知维（-1）跳过，交运行时 tiling 拒绝 ──
    if (p == INF_POS_SENTINEL || p == INF_NEG_SENTINEL) {
        for (const int64_t axis : reduceSet) {
            if (xShape.GetDim(static_cast<size_t>(axis)) == 0) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    nodeName, "p", std::to_string(p),
                    "p=+-inf sentinel cannot reduce over the whole (empty) tensor or an empty dim "
                    "(canndev constraint, spec.yaml shape_constraints)");
                return GRAPH_FAILED;
            }
        }
    }

    // ── 推导（canndev InferShape4LpNorm 同构）：keepdim=true 归约维置 1 保留、
    //    false 删除；非归约维照常透传（含动态 -1 未知维）；axes 为空 → 全归约 ──
    std::vector<int64_t> yDims;
    yDims.reserve(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; i++) {
        if (reduceSet.find(i) != reduceSet.end()) {
            if (keepdim) {
                yDims.push_back(1);
            }
        } else {
            yDims.push_back(xShape.GetDim(static_cast<size_t>(i)));
        }
    }

    TensorDesc yDesc = op.GetOutputDesc("y");
    yDesc.SetShape(Shape(yDims));
    yDesc.SetDataType(xDtype);
    yDesc.SetFormat(FORMAT_ND);
    OP_CHECK_IF(op.UpdateOutputDesc("y", yDesc) != GRAPH_SUCCESS, OP_LOGE(nodeName, "update output desc y failed"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

// V1 (Runtime 1.0) InferShapeFuncRegister 静态注册：so 被 GE 加载（dlopen）时
// 构造，op type "LpNormReduce" 的 V1 推导即为本实现。
INFER_FUNC_REG(LpNormReduce, InferShapeAndCheckLpNormReduceV1);

} // namespace ge
