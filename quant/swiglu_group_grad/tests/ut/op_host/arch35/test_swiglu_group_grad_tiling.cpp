/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/swiglu_group_grad_tiling_base.h"

using namespace ut_util;

namespace {

struct TilingCase {
    gert::Shape gradYShape = {4, 16};
    gert::Shape xShape = {4, 32};
    ge::DataType dtype = ge::DT_FLOAT;
    float clampLimit = -1.0f;
    bool hasWeight = false;
    bool hasYOrigin = false;
    bool hasGroupIndex = false;
    gert::Shape weightShape = {};
    ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS;
};

class SwigluGroupGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluGroupGradTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SwigluGroupGradTilingTest TearDown" << std::endl; }
};

void ExecuteTilingCase(const TilingCase& testCase)
{
    const std::string compileInfoString = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersions = {
        {"Short_SoC_version", "Ascend950"},
        {"NpuArch", "3510"},
    };
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::SwigluGroupGradCompileInfo compileInfo;

    constexpr char kOpType[] = "SwigluGroupGrad";
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(opImpl, nullptr);
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    auto* parsePlatform = kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo();
    ASSERT_TRUE(parsePlatform->Init());
    parsePlatform->SetPlatformRes("SoCInfo", socInfos);
    parsePlatform->SetPlatformRes("AICoreSpec", aicoreSpec);
    parsePlatform->SetCoreNumByCoreType("AICore");
    parsePlatform->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parsePlatform->SetPlatformRes("version", socVersions);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* workspaceSizes = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(tilingData, nullptr);

    gert::StorageShape gradYStorageShape;
    gradYStorageShape.MutableStorageShape() = testCase.gradYShape;
    gradYStorageShape.MutableOriginShape() = gradYStorageShape.MutableStorageShape();
    gert::StorageShape xStorageShape;
    xStorageShape.MutableStorageShape() = testCase.xShape;
    xStorageShape.MutableOriginShape() = xStorageShape.MutableStorageShape();
    gert::StorageShape weightStorageShape;
    weightStorageShape.MutableStorageShape() = testCase.weightShape;
    if (weightStorageShape.MutableStorageShape().GetDimNum() == 0) {
        weightStorageShape.MutableStorageShape() = testCase.gradYShape;
        weightStorageShape.MutableStorageShape().SetDim(weightStorageShape.MutableStorageShape().GetDimNum() - 1, 1);
    }
    weightStorageShape.MutableOriginShape() = weightStorageShape.MutableStorageShape();
    gert::StorageShape yOriginStorageShape;
    yOriginStorageShape.MutableStorageShape() = testCase.gradYShape;
    yOriginStorageShape.MutableOriginShape() = yOriginStorageShape.MutableStorageShape();
    gert::StorageShape groupIndexStorageShape;
    groupIndexStorageShape.MutableStorageShape() = gert::Shape({2});
    groupIndexStorageShape.MutableOriginShape() = groupIndexStorageShape.MutableStorageShape();
    gert::StorageShape gradXStorageShape;
    gradXStorageShape.MutableStorageShape() = testCase.xShape;
    gradXStorageShape.MutableOriginShape() = gradXStorageShape.MutableStorageShape();
    gert::StorageShape gradWeightStorageShape;
    if (testCase.hasWeight) {
        gradWeightStorageShape.MutableStorageShape() = weightStorageShape.MutableStorageShape();
        gradWeightStorageShape.MutableOriginShape() = gradWeightStorageShape.MutableStorageShape();
    }

    std::vector<uint32_t> inputInstanceNum = {1, 1, 1, 1, 1};
    std::vector<void*> inputShapes = {&gradYStorageShape, &xStorageShape,
                                      testCase.hasWeight ? &weightStorageShape : nullptr,
                                      testCase.hasYOrigin ? &yOriginStorageShape : nullptr,
                                      testCase.hasGroupIndex ? &groupIndexStorageShape : nullptr};

    gert::TilingContextFaker contextFaker;
    contextFaker.SetOpType(kOpType)
        .NodeIoNum(inputShapes.size(), 2)
        .IrInstanceNum(inputInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes({&gradXStorageShape, &gradWeightStorageShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(testCase.clampLimit)}})
        .TilingData(tilingData.get())
        .Workspace(workspaceSizes);
    if (testCase.hasWeight) {
        contextFaker.NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (testCase.hasYOrigin) {
        contextFaker.NodeInputTd(3, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (testCase.hasGroupIndex) {
        contextFaker.NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }

    auto holder = contextFaker.Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersions);

    EXPECT_EQ(tilingFunc(tilingContext), testCase.expectedStatus);
}

TEST_F(SwigluGroupGradTilingTest, tiling_fp32) { ExecuteTilingCase({}); }

TEST_F(SwigluGroupGradTilingTest, tiling_fp16)
{
    TilingCase testCase;
    testCase.dtype = ge::DT_FLOAT16;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_bf16)
{
    TilingCase testCase;
    testCase.dtype = ge::DT_BF16;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_all_optional_inputs)
{
    TilingCase testCase;
    testCase.clampLimit = 3.0f;
    testCase.hasWeight = true;
    testCase.hasYOrigin = true;
    testCase.hasGroupIndex = true;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_group_index_only)
{
    TilingCase testCase;
    testCase.hasGroupIndex = true;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_regbase_path_with_all_optional_inputs)
{
    TilingCase testCase;
    testCase.gradYShape = {64, 128};
    testCase.xShape = {64, 256};
    testCase.clampLimit = 3.0f;
    testCase.hasWeight = true;
    testCase.hasYOrigin = true;
    testCase.hasGroupIndex = true;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_regbase_path_without_optional_inputs)
{
    TilingCase testCase;
    testCase.gradYShape = {64, 128};
    testCase.xShape = {64, 256};
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_regbase_path_with_group_index_only)
{
    TilingCase testCase;
    testCase.gradYShape = {64, 128};
    testCase.xShape = {64, 256};
    testCase.hasGroupIndex = true;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_empty_tensor)
{
    TilingCase testCase;
    testCase.gradYShape = {0, 16};
    testCase.xShape = {0, 32};
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_rejects_zero_hidden_size)
{
    TilingCase testCase;
    testCase.gradYShape = {4, 0};
    testCase.xShape = {4, 0};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_rejects_invalid_dtype)
{
    TilingCase testCase;
    testCase.dtype = ge::DT_INT8;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_accepts_3d_input)
{
    TilingCase testCase;
    testCase.gradYShape = {2, 4, 16};
    testCase.xShape = {2, 4, 32};
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_accepts_3d_input_with_all_optional_inputs)
{
    TilingCase testCase;
    testCase.gradYShape = {2, 4, 16};
    testCase.xShape = {2, 4, 32};
    testCase.hasWeight = true;
    testCase.hasYOrigin = true;
    testCase.hasGroupIndex = true;
    testCase.weightShape = {8};
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_accepts_4d_input)
{
    TilingCase testCase;
    testCase.gradYShape = {2, 4, 16, 32};
    testCase.xShape = {128, 64};
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_rejects_mismatched_outer_numel)
{
    TilingCase testCase;
    testCase.gradYShape = {2, 4, 16, 32};
    testCase.xShape = {2, 8, 32, 64};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_rejects_unpaired_weight)
{
    TilingCase testCase;
    testCase.hasWeight = true;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_rejects_unpaired_y_origin)
{
    TilingCase testCase;
    testCase.hasYOrigin = true;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_rejects_zero_clamp_limit)
{
    TilingCase testCase;
    testCase.clampLimit = 0.0f;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteTilingCase(testCase);
}

TEST_F(SwigluGroupGradTilingTest, tiling_rejects_nan_clamp_limit)
{
    TilingCase testCase;
    testCase.clampLimit = std::numeric_limits<float>::quiet_NaN();
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteTilingCase(testCase);
}

} // namespace
