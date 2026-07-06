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
#include <map>
#include <string>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/swiglu_group_tiling.h"

namespace {
constexpr float DEFAULT_CLAMP_LIMIT = -1.0f;

struct TilingCase {
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_FLOAT16;
    ge::DataType weightDtype = ge::DT_FLOAT;
    ge::DataType groupIndexDtype = ge::DT_INT64;
    gert::StorageShape xShape = {{8, 128, 8192}, {8, 128, 8192}};
    gert::StorageShape weightShape = {{1024}, {1024}};
    gert::StorageShape groupIndexShape = {{2}, {2}};
    gert::StorageShape yShape = {{8, 128, 4096}, {8, 128, 4096}};
    float clampLimit = DEFAULT_CLAMP_LIMIT;
    bool hasWeight = false;
    bool hasGroupIndex = false;
    ge::graphStatus status = ge::GRAPH_SUCCESS;
};

class SwigluGroupTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluGroupTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SwigluGroupTilingTest TearDown" << std::endl; }
};

void ExecuteTilingCase(const TilingCase& tc)
{
    const std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::SwigluGroupCompileInfo compileInfo;

    const std::string opType("SwigluGroup");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

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
    gert::StorageShape xShape = tc.xShape;
    gert::StorageShape weightShape = tc.weightShape;
    gert::StorageShape groupIndexShape = tc.groupIndexShape;
    gert::StorageShape yShape = tc.yShape;

    std::vector<uint32_t> inputInstanceNum = {1, tc.hasWeight ? 1U : 0U, tc.hasGroupIndex ? 1U : 0U};
    std::vector<gert::StorageShape*> inputShapes = {&xShape};
    if (tc.hasWeight || tc.hasGroupIndex) {
        inputShapes.emplace_back(tc.hasWeight ? &weightShape : nullptr);
    }
    if (tc.hasGroupIndex) {
        inputShapes.emplace_back(&groupIndexShape);
    }

    gert::TilingContextFaker contextFaker;
    contextFaker.SetOpType(opType)
        .NodeIoNum(3, 1)
        .IrInstanceNum(inputInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes({&yShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, tc.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, tc.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(tc.clampLimit)}})
        .TilingData(tilingData.get())
        .Workspace(workspaceSizes);
    int32_t inputTdIndex = 1;
    if (tc.hasWeight) {
        contextFaker.NodeInputTd(inputTdIndex++, tc.weightDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (tc.hasGroupIndex) {
        contextFaker.NodeInputTd(inputTdIndex++, tc.groupIndexDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }

    auto holder = contextFaker.Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersions);

    EXPECT_EQ(tilingFunc(tilingContext), tc.status);
}

TEST_F(SwigluGroupTilingTest, tiling_fp16)
{
    TilingCase tc;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_bf16)
{
    TilingCase tc;
    tc.xDtype = ge::DT_BF16;
    tc.yDtype = ge::DT_BF16;
    tc.xShape = {{4, 64, 2048}, {4, 64, 2048}};
    tc.yShape = {{4, 64, 1024}, {4, 64, 1024}};
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_f32)
{
    TilingCase tc;
    tc.xDtype = ge::DT_FLOAT;
    tc.yDtype = ge::DT_FLOAT;
    tc.xShape = {{4, 64, 2048}, {4, 64, 2048}};
    tc.yShape = {{4, 64, 1024}, {4, 64, 1024}};
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_clamp_limit)
{
    TilingCase tc;
    tc.clampLimit = 7.0f;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_weight_valid)
{
    TilingCase tc;
    tc.hasWeight = true;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_group_index_valid)
{
    TilingCase tc;
    tc.hasGroupIndex = true;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_error_invalid_last_dim)
{
    // Last dim must be divisible by 2; an odd last dim is invalid.
    TilingCase tc;
    tc.xShape = {{4, 64, 1023}, {4, 64, 1023}};
    tc.yShape = {{4, 64, 511}, {4, 64, 511}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_even_non256_aligned)
{
    // Last dim only needs to be divisible by 2 (relaxed from 256); 384 is valid.
    TilingCase tc;
    tc.xShape = {{4, 384}, {4, 384}};
    tc.yShape = {{4, 192}, {4, 192}};
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_error_invalid_x_dtype)
{
    // x only supports float16/bfloat16/float32; int32 is invalid.
    TilingCase tc;
    tc.xDtype = ge::DT_INT32;
    tc.yDtype = ge::DT_INT32;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_error_invalid_weight_dtype)
{
    TilingCase tc;
    tc.hasWeight = true;
    tc.weightDtype = ge::DT_FLOAT16;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_error_invalid_weight_shape)
{
    TilingCase tc;
    tc.hasWeight = true;
    tc.weightShape = {{1023}, {1023}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_error_invalid_group_index_dtype)
{
    TilingCase tc;
    tc.hasGroupIndex = true;
    tc.groupIndexDtype = ge::DT_INT32;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupTilingTest, tiling_error_zero_clamp_limit)
{
    TilingCase tc;
    tc.clampLimit = 0.0f;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}
} // namespace
