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
        attrs == nullptr, OP_LOGE(context->GetNodeName(), "Function context->GetAttrs() failed!"),
        return ge::GRAPH_FAILED);

    // check the Required input and output desc
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X1_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X2_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X1_LEVEL0_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X1_LEVEL1_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X2_LEVEL0_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X2_LEVEL1_SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(OUTPUT_Y_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs->GetAttrPointer<int64_t>(ATTR_DTYPE_INDEX));

    // check Raw TilingData
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context->GetRawTilingData()->GetCapacity() < tilingDataSize,
        OP_LOGE(
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
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "transA, transB",
            (std::string(inputParams.transA ? "true" : "false") + ", " +
             (inputParams.transB ? "true" : "false")).c_str(),
            "x1 transpose must be false and x2 transpose must be true"),
        return false);
    OP_TILING_CHECK(
        inputParams.level1GroupSize != MICROSCALING_GROUP_SIZE,
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "level1GroupSize", std::to_string(inputParams.level1GroupSize).c_str(),
            std::to_string(MICROSCALING_GROUP_SIZE).c_str()),
        return false);
    OP_TILING_CHECK(
        inputParams.level0GroupSize != DEFAULT_LEVEL0_GROUP_SIZE,
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "level0GroupSize", std::to_string(inputParams.level0GroupSize).c_str(),
            std::to_string(DEFAULT_LEVEL0_GROUP_SIZE).c_str()),
        return false);
    return true;
}

bool CheckDtypes(
    [[maybe_unused]] gert::TilingContext* context, [[maybe_unused]] NpuArch npuArch,
    const DualLevelQuantBatchMatmulInfo& inputParams)
{
    OP_TILING_CHECK(
        inputParams.x1Dtype != ge::DT_FLOAT4_E2M1 || inputParams.x2Dtype != ge::DT_FLOAT4_E2M1,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context->GetNodeName(), "x1, x2",
            (ge::TypeUtils::DataTypeToSerialString(inputParams.x1Dtype) + ", " +
             ge::TypeUtils::DataTypeToSerialString(inputParams.x2Dtype)).c_str(),
            "dtype is only supported to be float4_e2m1"),
        return false);
    OP_TILING_CHECK(
        inputParams.x1Level0ScaleDtype != ge::DT_FLOAT || inputParams.x2Level0ScaleDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context->GetNodeName(), "x1Level0Scale, x2Level0Scale",
            (ge::TypeUtils::DataTypeToSerialString(inputParams.x1Level0ScaleDtype) + ", " +
             ge::TypeUtils::DataTypeToSerialString(inputParams.x2Level0ScaleDtype)).c_str(),
            "dtype is only supported to be float"),
        return false);
    OP_TILING_CHECK(
        inputParams.x1Level1ScaleDtype != ge::DT_FLOAT8_E8M0 ||
            inputParams.x2Level1ScaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context->GetNodeName(), "x1Level1Scale, x2Level1Scale",
            (ge::TypeUtils::DataTypeToSerialString(inputParams.x1Level1ScaleDtype) + ", " +
             ge::TypeUtils::DataTypeToSerialString(inputParams.x2Level1ScaleDtype)).c_str(),
            "dtype is only supported to be float8_e8m0"),
        return false);
    if (inputParams.hasBias) {
        OP_TILING_CHECK(
            inputParams.biasDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "bias",
                ge::TypeUtils::DataTypeToSerialString(inputParams.biasDtype).c_str(),
                "dtype is only supported to be float"),
            return false);
    }
    OP_TILING_CHECK(
        inputParams.yDtype != ge::DT_FLOAT16 && inputParams.yDtype != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(inputParams.yDtype).c_str(),
            "dtype is only supported to be float16 or bfloat16"),
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
    shapeStr.push_back(']');
    return shapeStr;
}

bool CheckInputShape(
    gert::TilingContext* context, const char* variableName, const gert::Shape& shape,
    std::initializer_list<uint64_t> expectedShape)
{
    auto shapeLen = shape.GetDimNum();
    OP_TILING_CHECK(
        shapeLen != expectedShape.size(),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context->GetNodeName(), variableName,
            (std::to_string(shapeLen) + "D").c_str(),
            (std::to_string(expectedShape.size()) + "D").c_str()),
        return false);
    size_t i = 0;
    for (auto dim : expectedShape) {
        OP_TILING_CHECK(
            dim != static_cast<uint64_t>(shape.GetDim(i++)),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), variableName, Ops::Base::ToString(shape).c_str(),
                ("The shape of " + std::string(variableName) + " must be " + ToShapeString(expectedShape)).c_str()),
            return false);
    }
    return true;
}

bool IsInputsValid(gert::TilingContext* context, const DualLevelQuantBatchMatmulInfo& inputParams)
{
    OP_TILING_CHECK(
        inputParams.mSize == 0 || inputParams.nSize == 0 || inputParams.kSize == 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "mSize, nSize, kSize",
            (std::to_string(inputParams.mSize) + ", " + std::to_string(inputParams.nSize) + ", " +
             std::to_string(inputParams.kSize)).c_str(),
            "The sizes of M, N and K axes can not be 0"),
        return false);
    OP_TILING_CHECK(
        inputParams.x1Format != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(
            context->GetNodeName(), "x1",
            ge::TypeUtils::FormatToAscendString(inputParams.x1Format).GetString(), "ND"),
        return false);
    OP_TILING_CHECK(
        inputParams.x2Format != ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            context->GetNodeName(), "x2",
            ge::TypeUtils::FormatToAscendString(inputParams.x2Format).GetString(), "FRACTAL_NZ"),
        return false);
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
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "x1Level0Scale/x1Level1Scale/x2Level0Scale/x2Level1Scale", "0",
            "The shape size of x1Level0Scale/x1Level1Scale/x2Level0Scale/x2Level1Scale can not be 0"),
        return false);

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
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "bias", "0",
                "The shape size of bias can not be 0, if no bias is needed, please use a null pointer"),
            return false);
        if (!CheckInputShape(context, "bias", biasShape, {inputParams.nSize})) {
            return false;
        }
    }
    return true;
}

} // Ops::NN::DLQBMMChecker