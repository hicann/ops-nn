/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "../../../op_host/rms_norm_quant_v2_tiling.h"

namespace {
constexpr size_t TILING_DATA_CAPACITY = 4096U;
constexpr size_t WORKSPACE_ELEMENT_CAPACITY = 4096U;

class RmsNormQuantV2RegbaseTilingForTest : public optiling::RmsNormQuantV2RegbaseTilingFullLoad {
public:
    explicit RmsNormQuantV2RegbaseTilingForTest(gert::TilingContext* context)
        : optiling::RmsNormQuantV2RegbaseTilingFullLoad(context)
    {}

    ge::graphStatus GetShapeAttrsInfoForTest()
    {
        return optiling::RmsNormQuantV2RegbaseTilingBase::GetShapeAttrsInfo();
    }
};

bool ValidateShapeAttrs(const std::vector<gert::StorageShape*>& inputShapes,
                        const std::vector<gert::StorageShape*>& outputShapes, bool isV3, bool outputRstd)
{
    const std::string opType = isV3 ? "RmsNormQuantV3" : "RmsNormQuantV2";
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
        {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
        {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT8)}};
    if (isV3) {
        attrs.emplace_back("output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(outputRstd));
    }
    auto tilingData = gert::TilingData::CreateCap(TILING_DATA_CAPACITY);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(WORKSPACE_ELEMENT_CAPACITY);
    optiling::RmsNormQuantV2CompileInfo compileInfo;
    char platformInfo = 0;
    if (tilingData == nullptr || workspaceHolder == nullptr) {
        return false;
    }
    auto* workspace = static_cast<gert::ContinuousVector*>(static_cast<void*>(workspaceHolder.get()));
    gert::TilingContextFaker faker;
    faker.SetOpType(opType)
        .NodeIoNum(inputShapes.size(), outputShapes.size())
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compileInfo)
        .PlatformInfo(&platformInfo);
    if (isV3) {
        faker.NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.NodeAttrs(attrs).TilingData(tilingData.get()).Workspace(workspace).Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr) {
        return false;
    }
    RmsNormQuantV2RegbaseTilingForTest tiling(context);
    return tiling.GetShapeAttrsInfoForTest() == ge::GRAPH_SUCCESS;
}
} // namespace

TEST(RmsNormQuantV2EmptyTensorTiling, rejects_empty_input_and_output_shapes)
{
    gert::StorageShape xShape = {{8, 64}, {8, 64}};
    gert::StorageShape gammaShape = {{64}, {64}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape betaShape = {{64}, {64}};
    gert::StorageShape yShape = {{8, 64}, {8, 64}};
    gert::StorageShape emptyShape = {{0}, {0}};
    std::vector<gert::StorageShape*> inputShapes = {&xShape,     &gammaShape, &scaleShape, &scaleShape,
                                                    &scaleShape, &scaleShape, &betaShape};
    std::vector<gert::StorageShape*> outputShapes = {&yShape, &yShape};

    for (size_t index = 0; index < inputShapes.size(); ++index) {
        SCOPED_TRACE("empty input index: " + std::to_string(index));
        auto* originalShape = inputShapes[index];
        inputShapes[index] = &emptyShape;
        EXPECT_FALSE(ValidateShapeAttrs(inputShapes, outputShapes, false, false));
        inputShapes[index] = originalShape;
    }
    for (size_t index = 0; index < outputShapes.size(); ++index) {
        SCOPED_TRACE("empty output index: " + std::to_string(index));
        auto* originalShape = outputShapes[index];
        outputShapes[index] = &emptyShape;
        EXPECT_FALSE(ValidateShapeAttrs(inputShapes, outputShapes, false, false));
        outputShapes[index] = originalShape;
    }
    EXPECT_TRUE(ValidateShapeAttrs(inputShapes, outputShapes, false, false));
}

TEST(RmsNormQuantV2EmptyTensorTiling, validates_rstd_only_when_requested)
{
    gert::StorageShape xShape = {{8, 64}, {8, 64}};
    gert::StorageShape gammaShape = {{64}, {64}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape betaShape = {{64}, {64}};
    gert::StorageShape yShape = {{8, 64}, {8, 64}};
    gert::StorageShape emptyRstdShape = {{8, 0}, {8, 0}};
    std::vector<gert::StorageShape*> inputShapes = {&xShape,     &gammaShape, &scaleShape, &scaleShape,
                                                    &scaleShape, &scaleShape, &betaShape};
    std::vector<gert::StorageShape*> outputShapes = {&yShape, &yShape, &emptyRstdShape};

    EXPECT_FALSE(ValidateShapeAttrs(inputShapes, outputShapes, true, true));
    EXPECT_TRUE(ValidateShapeAttrs(inputShapes, outputShapes, true, false));
}
