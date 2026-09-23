/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// norm/l2_normalize/op_graph/l2_normalize_graph_infer.cpp
// =============================================================================
//
// ROLE: Graph-level inference registration for the L2Normalize operator.
//   - V1（COMMON_INFER_FUNC_REG，ge::Operator 旧式推导）：GE 图编译期
//     （AddGraph/RunGraph 的 shape/type 推导 pass）实际调用的推导入口。本算子 IR
//     与内置 nn_batch_norm_ops.h 的 L2Normalize 同名，内置
//     .so 亦按同名注册 V1 推导（OneInOneOutDynamicInfer）；本包在 custom 路径
//     先于内置包加载（AddSoToRegistry 顺序），按 insert-if-absent 语义以本实现
//     占据同名 V1 槽位，保证推导与校验语义来自本包（y.shape = x.shape、
//     y.dtype = x.dtype）。
//   - V2（IMPL_OP … InferDataType，gert 新式推导）：动态图执行期 executor 直接
//     调用的 gert kernel 注册（InferShape 在 op_host/l2_normalize_infershape.cpp
//     注册）。两条注册路径规则保持一致（恒等透传 + 同一套参数校验）。
//   - 参数校验（GEIR 图模式）：rank-0 标量拒绝（契约 rank 1–8，无合法
//     axis）；声明输出 shape/dtype 与 x 不一致拒绝（与 tiling 硬门禁保持一致；
//     未声明（rank-0 空 shape / DT_UNDEFINED）与未知维
//     （-1/-2）不校验）；声明 format 仅 ND（FORMAT_RESERVED 未声明跳过）。
//     dtype 合法域仍由 OpDef/REG_OP TensorType 声明承载。
// =============================================================================

#include "register/op_impl_registry.h" // IMPL_OP macro for operator registration
#include "graph/operator_reg.h"        // IMPLEMT_COMMON_INFERFUNC / COMMON_INFER_FUNC_REG
#include "graph/utils/type_utils.h"    // TypeUtils::DataTypeToSerialString
#include "op_common/log/log.h"         // OP_LOGE / OP_LOGE_FOR_INVALID_* macros
#include <string>
#include <vector>

using namespace ge;

namespace {

constexpr size_t INPUT_X_IDX = 0;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t MAX_INPUT_RANK = 8;

std::string ShapeToString(const ge::Shape& shape)
{
    std::string str = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i > 0) {
            str += ",";
        }
        str += std::to_string(shape.GetDim(i));
    }
    str += "]";
    return str;
}

// 声明输出 shape 与 x 一致性：rank 相等且双侧均为已知维（≥0）时逐维相等。
// 任一侧未知维（<0，动态 shape -1/-2）不参与比对。
bool DeclaredShapeCompatible(const ge::Shape& declared, const ge::Shape& xShape)
{
    if (declared.GetDimNum() != xShape.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        const int64_t xDim = xShape.GetDim(i);
        const int64_t yDim = declared.GetDim(i);
        if (xDim >= 0 && yDim >= 0 && xDim != yDim) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace ops {

// ---------------------------------------------------------------------------
// V1 图编译期推导（ge::Operator 旧式签名，ge::Operator& 入参）：
//   0. 未知秩（rank=-2）：GE 未知秩在 ge::Shape 中存为 dims={-2} 且
//      GetDimNum()==0（与真标量取值相同，须以 GetDims() 区分，见 graph/tensor.h
//      ge::Shape 注释）；零元素张量在动态图下即以该形态到达（FR-P1-011
//      L0_empty_027/034：desc/ori_shape=[-2]）。未知秩跳过步骤 1/2 的维数校验，
//      y 恒等复制未知秩；dtype/format 契约校验与维数无关，保留。
//   1. rank-0 标量拒绝（契约 rank 1–8；rank-0 无合法 axis）；
//   2. 声明输出 shape 一致性（y.shape = x.shape；未知维/未声明跳过）；
//   3. 声明输出 dtype 一致性（y.dtype = x.dtype；DT_UNDEFINED 跳过）；
//   4. 声明 format 校验（x/y 仅 ND；FORMAT_RESERVED 未声明跳过）；
//   5. 恒等推导：y ← x（shape + dtype，语义对齐内置 OneInOneOutDynamicInfer）。
// ---------------------------------------------------------------------------
static ge::graphStatus L2NormalizeGraphInferFunc(ge::Operator& op)
{
    const ge::TensorDesc xDesc = op.GetInputDescByName("x");
    const ge::Shape xShape = xDesc.GetShape();
    const ge::DataType xDtype = xDesc.GetDataType();

    ge::TensorDesc yDesc = op.GetOutputDescByName("y");
    const ge::Shape yShape = yDesc.GetShape();
    const ge::DataType yDtype = yDesc.GetDataType();

    // 未知秩标记（GE_UNKNOWN_RANK）：步骤 1/2 的维数相关校验对 rank=-2 无意义，整体跳过
    const bool isUnknownRank = (xShape.GetDims() == ge::UNKNOWN_RANK);

    // 1. rank-0 标量拒绝（契约 rank 1–8；GE 图管线会把 rank-0 规整为 (1) 后才到
    //    tiling，故该门禁须在推导期拦截）。标量判定按 ge::Shape 官方推荐
    //    GetDims().empty()（GetDimNum()==0 与未知秩取值冲突，见步骤 0）。
    if (!isUnknownRank && (xShape.GetDims().empty() || xShape.GetDimNum() > MAX_INPUT_RANK)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("L2Normalize", "rank", std::to_string(xShape.GetDimNum()).c_str(), "[1, 8]");
        return ge::GRAPH_FAILED;
    }

    // 2. 声明输出 shape 一致性（host 侧硬门禁；缺省会落设备写坏 out 缓冲）；
    //    未知秩无维可校，跳过（y 由步骤 5 恒等复制未知秩）
    if (!isUnknownRank && yShape.GetDimNum() != 0U && !DeclaredShapeCompatible(yShape, xShape)) {
        OP_LOGE_FOR_INVALID_SHAPE("L2Normalize", "y", ShapeToString(yShape).c_str(), ShapeToString(xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    // 3. 声明输出 dtype 一致性（y 与 x 恒一致，same_as_first_input）
    if (yDtype != ge::DT_UNDEFINED && yDtype != xDtype) {
        OP_LOGE_FOR_INVALID_DTYPE("L2Normalize", "y", ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                                  ge::TypeUtils::DataTypeToSerialString(xDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    // 4. 声明 format 校验：x / y 仅支持 ND；未声明（FORMAT_RESERVED）不校验。y 侧 format 缺该
    //    门禁时 FRACTAL_NZ 声明会落入图执行（tiling 仅校验 x 侧 format）。
    if (xDesc.GetFormat() != ge::FORMAT_ND && xDesc.GetFormat() != ge::FORMAT_RESERVED) {
        OP_LOGE_FOR_INVALID_FORMAT("L2Normalize", "x", std::to_string(static_cast<int64_t>(xDesc.GetFormat())).c_str(),
                                   "ND");
        return ge::GRAPH_FAILED;
    }
    if (yDesc.GetFormat() != ge::FORMAT_ND && yDesc.GetFormat() != ge::FORMAT_RESERVED) {
        OP_LOGE_FOR_INVALID_FORMAT("L2Normalize", "y", std::to_string(static_cast<int64_t>(yDesc.GetFormat())).c_str(),
                                   "ND");
        return ge::GRAPH_FAILED;
    }

    // 5. 恒等推导：y.shape = x.shape、y.dtype = x.dtype（format 不改写）
    yDesc.SetShape(xShape);
    yDesc.SetDataType(xDtype);
    if (op.UpdateOutputDesc("y", yDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGE("L2Normalize", "Update output desc y failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace ops

// V1 推导注册（与 canndev 内置 nn_batch_norm_ops.cc 的 COMMON_INFER_FUNC_REG 同机制；
// 本包 custom 路径先加载，占据同名 V1 槽位）
COMMON_INFER_FUNC_REG(L2Normalize, ops::L2NormalizeGraphInferFunc);

namespace ops {

// ---------------------------------------------------------------------------
// V2 InferDataType（gert 新式推导，动态图执行期 executor 调用）：
//   y.dtype = x.dtype（same_as_first_input 单输入恒等，无接口级
//   promotion；fp16 内部 fp32 提升不改变输出 dtype，推导不感知）。dtype 合法域
//   （{DT_FLOAT16, DT_FLOAT}）由 OpDef/REG_OP TensorType 声明承载，本函数仅做
//   恒等透传，不做值域校验。
// ---------------------------------------------------------------------------
static ge::graphStatus InferDataTypeForL2Normalize(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_Y_IDX, context->GetInputDataType(INPUT_X_IDX));
    return ge::GRAPH_SUCCESS;
}

// IMPL_OP(L2Normalize).InferDataType(func):
//   Registers InferDataTypeForL2Normalize as the type inference function
//   for the L2Normalize operator type.
IMPL_OP(L2Normalize).InferDataType(InferDataTypeForL2Normalize);
} // namespace ops
