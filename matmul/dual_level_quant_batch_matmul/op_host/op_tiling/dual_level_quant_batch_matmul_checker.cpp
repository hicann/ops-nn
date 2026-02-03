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
 * \file dual_level_quant_batch_matmul_checker.cpp
 * \brief
 */
#include "dual_level_quant_batch_matmul_checker.h"
#include <cstdint>
#include <vector>
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"

using namespace optiling;

constexpr uint64_t MICROSCALING_GROUP_SIZE = 32UL;
constexpr uint64_t DEFAULT_LEVEL0_GROUP_SIZE = 512UL;

namespace Ops::NN::DLQBMMChecker {
ge::graphStatus CheckContext(gert::TilingContext* context, uint64_t tilingDataSize)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(
        attrs == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Function context->GetAttrs() failed!"),
        return ge::GRAPH_FAILED);

    // check the Required input and output desc
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X1_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X2_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X1_LEVEL0_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X1_LEVEL1_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X2_LEVEL0_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X2_LEVEL1_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(OUTPUT_Y_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs->GetAttrPointer<int32_t>(ATTR_DTYPE_INDEX));

    // check Raw TilingData
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context->GetRawTilingData()->GetCapacity() < tilingDataSize,
        CUBE_INNER_ERR_REPORT(
            context, "context tiling data capacity %zu < actual tiling data size %zu.",
            context->GetRawTilingData()->GetCapacity(), tilingDataSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool CheckAttrs(
    [[maybe_unused]] gert::TilingContext* context, [[maybe_unused]] NpuArch npuArch,
    const DualLevelQuantBatchMatmulInfo& inputParams)
{
    OP_TILING_CHECK(
        inputParams.transA != false || inputParams.transB != true,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context,
            "x1 transpose should be false and x2 transpose should be true, "
            "but got x1 transpose: %s, x2 transpose: %s",
            inputParams.transA ? "true" : "false", inputParams.transB ? "true" : "false"),
        return false);
    OP_TILING_CHECK(
        inputParams.level1GroupSize != MICROSCALING_GROUP_SIZE,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context, "Level1 group size is only supported to be %zu, but got %zu", MICROSCALING_GROUP_SIZE,
            inputParams.level1GroupSize),
        return false);
    OP_TILING_CHECK(
        inputParams.level0GroupSize != DEFAULT_LEVEL0_GROUP_SIZE,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context, "Level0 group size is only supported to be %zu, but got %zu", DEFAULT_LEVEL0_GROUP_SIZE,
            inputParams.level0GroupSize),
        return false);
    return true;
}

bool CheckDtypes(
    [[maybe_unused]] gert::TilingContext* context, [[maybe_unused]] NpuArch npuArch,
    const DualLevelQuantBatchMatmulInfo& inputParams)
{
    OP_TILING_CHECK(
        inputParams.x1Dtype != ge::DT_FLOAT4_E2M1 || inputParams.x2Dtype != ge::DT_FLOAT4_E2M1,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context, "Input x1 and x2 dtype is only supported to be float4_e2m1, but x1Dtype: %s x2Dtype: %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams.x1Dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.x2Dtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams.x1Level0ScaleDtype != ge::DT_FLOAT || inputParams.x2Level0ScaleDtype != ge::DT_FLOAT,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context,
            "Input x1Level0Scale and x2Level0Scale dtype is only supported to be float, "
            "but x1Level0ScaleDtype: %s, x2Level0ScaleDtype: %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams.x1Level0ScaleDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.x2Level0ScaleDtype).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams.x1Level1ScaleDtype != ge::DT_FLOAT8_E8M0 || inputParams.x2Level1ScaleDtype != ge::DT_FLOAT8_E8M0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context,
            "Input x1Level1Scale and x2Level1Scale dtype is only supported to be float8_e8m0, "
            "but x1Level1ScaleDtype: %s, x2Level1ScaleDtype: %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams.x1Level1ScaleDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.x2Level1ScaleDtype).c_str()),
        return false);
    if (inputParams.hasBias) {
        OP_TILING_CHECK(
            inputParams.biasDtype != ge::DT_FLOAT,
            VECTOR_INNER_ERR_REPORT_TILIING(
                context,
                "Input bias dtype is only supported to be float, "
                "but got %s",
                ge::TypeUtils::DataTypeToSerialString(inputParams.biasDtype).c_str()),
            return false);
    }
    OP_TILING_CHECK(
        inputParams.yDtype != ge::DT_FLOAT16 && inputParams.yDtype != ge::DT_BF16,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context, "Output y dtype is only supported to be float16 or bfloat16, but got %s",
            ge::TypeUtils::DataTypeToSerialString(inputParams.yDtype).c_str()),
        return false);
    return true;
}

std::string ToShapeString(std::initializer_list<uint64_t> shape)
{
    std::string shapeStr("[");
    const char* sep = "";
    for (auto x : shape) {
        shapeStr.append(sep);
        shapeStr.append(std::to_string(x));
        sep = ", ";
    }
    return shapeStr;
}

bool CheckInputShape(
    gert::TilingContext* context, const char* variableName, const gert::Shape& shape,
    std::initializer_list<uint64_t> expectedShape)
{
    auto shapeLen = shape.GetDimNum();
    OP_TILING_CHECK(
        shapeLen != expectedShape.size(),
        CUBE_INNER_ERR_REPORT(
            context, "input %s deminsion should be %zu, but got %zu", variableName, expectedShape.size(), shapeLen),
        return false);
    size_t i = 0;
    for (auto dim : expectedShape) {
        OP_TILING_CHECK(
            dim != static_cast<uint64_t>(shape.GetDim(i++)),
            VECTOR_INNER_ERR_REPORT_TILIING(
                context, "Check input %s shape failed, expected %s, but got %s", variableName,
                ToShapeString(expectedShape).c_str(), Ops::Base::ToString(shape).c_str()),
            return false);
    }
    return true;
}

bool IsInputsValid(gert::TilingContext* context, const DualLevelQuantBatchMatmulInfo& inputParams)
{
    OP_TILING_CHECK(
        inputParams.mSize == 0 || inputParams.nSize == 0 || inputParams.kSize == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context, "The input M, N and K axes cannot be 0"), return false);
    OP_TILING_CHECK(
        inputParams.x1Format != ge::FORMAT_ND,
        VECTOR_INNER_ERR_REPORT_TILIING(context, "Input x1 input format shoulde be ND"), return false);
    OP_TILING_CHECK(
        inputParams.x2Format != ge::FORMAT_FRACTAL_NZ,
        VECTOR_INNER_ERR_REPORT_TILIING(context, "Input x2 weight format shoulde be FRACTAL_NZ"), return false);
    return true;
}

bool CheckInputs(
    [[maybe_unused]] gert::TilingContext* context, [[maybe_unused]] NpuArch npuArch,
    const DualLevelQuantBatchMatmulInfo& inputParams)
{
    if (!IsInputsValid(context, inputParams)) {
        return false;
    }

    auto& x1Level0ScaleShape = context->GetInputShape(X1_LEVEL0_SCALE_INDEX)->GetOriginShape();
    auto& x1Level1ScaleShape = context->GetInputShape(X1_LEVEL1_SCALE_INDEX)->GetOriginShape();
    auto& x2Level0ScaleShape = context->GetInputShape(X2_LEVEL0_SCALE_INDEX)->GetOriginShape();
    auto& x2Level1ScaleShape = context->GetInputShape(X2_LEVEL1_SCALE_INDEX)->GetOriginShape();
    OP_TILING_CHECK(
        x1Level0ScaleShape.GetShapeSize() == 0 || x1Level1ScaleShape.GetShapeSize() == 0 ||
            x2Level0ScaleShape.GetShapeSize() == 0 || x2Level1ScaleShape.GetShapeSize() == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context, "Not yet support empty tensor"), return false);

    // check input shape
    uint64_t level1ScaleKSize = ops::CeilDiv<uint64_t>(inputParams.kSize, MICROSCALING_GROUP_SIZE * 2UL);
    uint64_t level0ScaleKSize = ops::CeilDiv<uint64_t>(inputParams.kSize, DEFAULT_LEVEL0_GROUP_SIZE);

    if (!CheckInputShape(context, "x1Level1Scale", x1Level1ScaleShape, {inputParams.mSize, level1ScaleKSize, 2UL})) {
        return false;
    }
    if (!CheckInputShape(context, "x1Level0Scale", x1Level0ScaleShape, {inputParams.mSize, level0ScaleKSize})) {
        return false;
    }
    if (!CheckInputShape(context, "x2Level1Scale", x2Level1ScaleShape, {inputParams.nSize, level1ScaleKSize, 2UL})) {
        return false;
    }
    if (!CheckInputShape(context, "x2Level0Scale", x2Level0ScaleShape, {level0ScaleKSize, inputParams.nSize})) {
        return false;
    }

    if (inputParams.hasBias) {
        auto& biasShape = context->GetInputShape(BIAS_INDEX)->GetOriginShape();
        OP_TILING_CHECK(
            biasShape.GetShapeSize() == 0,
            VECTOR_INNER_ERR_REPORT_TILIING(
                context,
                "Optional input bias is not yet support empty tensor, "
                "if the meaning is that no bias is needed, please use a null pointer"),
            return false);
        if (!CheckInputShape(context, "bias", biasShape, {inputParams.nSize})) {
            return false;
        }
    }
    return true;
}

} // Ops::NN::DLQBMMChecker