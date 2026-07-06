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
#include "../../../../op_host/arch35/swiglu_group_quant_tiling.h"

namespace {
constexpr float DEFAULT_CLAMP_LIMIT = -1.0f;

struct TilingCase {
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType scaleDtype = ge::DT_FLOAT;
    ge::DataType yOriginDtype = ge::DT_FLOAT16;
    ge::DataType weightDtype = ge::DT_FLOAT;
    ge::DataType groupIndexDtype = ge::DT_INT64;
    gert::StorageShape xShape = {{8, 128, 8192}, {8, 128, 8192}};
    gert::StorageShape weightShape = {{1024}, {1024}};
    gert::StorageShape groupIndexShape = {{2}, {2}};
    gert::StorageShape yShape = {{8, 128, 4096}, {8, 128, 4096}};
    gert::StorageShape scaleShape = {{8, 128, 32}, {8, 128, 32}};
    gert::StorageShape yOriginShape = {{8, 128, 4096}, {8, 128, 4096}};
    int64_t dstType = ge::DT_FLOAT8_E4M3FN;
    int64_t quantMode = 0;
    int64_t blockSize = 0;
    bool roundScale = false;
    float clampLimit = DEFAULT_CLAMP_LIMIT;
    float dstTypeMax = 448.0f;
    bool outputOrigin = false;
    bool hasWeight = false;
    bool hasGroupIndex = false;
    bool hasScale = false;
    ge::graphStatus status = ge::GRAPH_SUCCESS;
};

class SwigluGroupQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluGroupQuantTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SwigluGroupQuantTilingTest TearDown" << std::endl; }
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
    optiling::SwigluGroupQuantCompileInfo compileInfo;

    const std::string opType("SwigluGroupQuant");
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
    gert::StorageShape scaleShape = tc.scaleShape;
    gert::StorageShape yOriginShape = tc.yOriginShape;

    std::vector<uint32_t> inputInstanceNum = {1, 1, 1, 1};
    std::vector<gert::StorageShape*> inputShapes = {&xShape, tc.hasWeight ? &weightShape : nullptr,
                                                    tc.hasGroupIndex ? &groupIndexShape : nullptr,
                                                    tc.hasScale ? &scaleShape : nullptr};

    gert::TilingContextFaker contextFaker;
    contextFaker.SetOpType(opType)
        .NodeIoNum(4, 3)
        .IrInstanceNum(inputInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes({&yShape, &scaleShape, &yOriginShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, tc.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, tc.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, tc.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, tc.yOriginDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(tc.dstType)},
                    {"quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(tc.quantMode)},
                    {"block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(tc.blockSize)},
                    {"round_scale", Ops::NN::AnyValue::CreateFrom<bool>(tc.roundScale)},
                    {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(tc.clampLimit)},
                    {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(tc.dstTypeMax)},
                    {"output_origin", Ops::NN::AnyValue::CreateFrom<bool>(tc.outputOrigin)}})
        .TilingData(tilingData.get())
        .Workspace(workspaceSizes);
    if (tc.hasWeight) {
        contextFaker.NodeInputTd(1, tc.weightDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (tc.hasGroupIndex) {
        contextFaker.NodeInputTd(2, tc.groupIndexDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (tc.hasScale) {
        contextFaker.NodeInputTd(3, tc.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND);
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

TEST_F(SwigluGroupQuantTilingTest, tiling_block_fp8)
{
    TilingCase tc;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_block_fp8_y_origin)
{
    TilingCase tc;
    tc.outputOrigin = true;
    tc.roundScale = true;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_mx_fp8)
{
    TilingCase tc;
    tc.xDtype = ge::DT_BF16;
    tc.yDtype = ge::DT_FLOAT8_E5M2;
    tc.scaleDtype = ge::DT_FLOAT8_E8M0;
    tc.yOriginDtype = ge::DT_BF16;
    tc.xShape = {{4, 64, 2048}, {4, 64, 2048}};
    tc.yShape = {{4, 64, 1024}, {4, 64, 1024}};
    tc.scaleShape = {{4, 64, 16, 2}, {4, 64, 16, 2}};
    tc.yOriginShape = {{4, 64, 1024}, {4, 64, 1024}};
    tc.dstType = ge::DT_FLOAT8_E5M2;
    tc.quantMode = 1;
    tc.roundScale = true;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_mx_fp8_y_origin)
{
    TilingCase tc;
    tc.scaleDtype = ge::DT_FLOAT8_E8M0;
    tc.scaleShape = {{8, 128, 64, 2}, {8, 128, 64, 2}};
    tc.quantMode = 1;
    tc.roundScale = true;
    tc.outputOrigin = true;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_mx_fp4)
{
    TilingCase tc;
    tc.yDtype = ge::DT_FLOAT4_E2M1;
    tc.scaleDtype = ge::DT_FLOAT8_E8M0;
    tc.xShape = {{2, 8, 1024}, {2, 8, 1024}};
    tc.yShape = {{2, 8, 256}, {2, 8, 256}};
    tc.scaleShape = {{2, 8, 8, 2}, {2, 8, 8, 2}};
    tc.yOriginShape = {{2, 8, 512}, {2, 8, 512}};
    tc.dstType = ge::DT_FLOAT4_E2M1;
    tc.quantMode = 1;
    tc.roundScale = true;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_error_mx_without_round_scale)
{
    TilingCase tc;
    tc.scaleDtype = ge::DT_FLOAT8_E8M0;
    tc.scaleShape = {{8, 128, 64, 2}, {8, 128, 64, 2}};
    tc.quantMode = 1;
    tc.roundScale = false;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_error_invalid_block_size)
{
    TilingCase tc;
    tc.blockSize = 32;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_error_fp4_block_quant)
{
    TilingCase tc;
    tc.yDtype = ge::DT_FLOAT4_E1M2;
    tc.dstType = ge::DT_FLOAT4_E1M2;
    tc.quantMode = 0;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_error_invalid_last_dim)
{
    TilingCase tc;
    tc.xShape = {{4, 64, 1024 + 128}, {4, 64, 1024 + 128}};
    tc.yShape = {{4, 64, 576}, {4, 64, 576}};
    tc.yOriginShape = {{4, 64, 576}, {4, 64, 576}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_error_invalid_x_dtype)
{
    TilingCase tc;
    tc.xDtype = ge::DT_FLOAT;
    tc.yOriginDtype = ge::DT_FLOAT;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_weight_valid)
{
    TilingCase tc;
    tc.hasWeight = true;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_error_invalid_weight_dtype)
{
    TilingCase tc;
    tc.hasWeight = true;
    tc.weightDtype = ge::DT_FLOAT16;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_error_invalid_weight_shape)
{
    TilingCase tc;
    tc.hasWeight = true;
    tc.weightShape = {{1023}, {1023}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_error_invalid_group_index_dtype)
{
    TilingCase tc;
    tc.hasGroupIndex = true;
    tc.groupIndexDtype = ge::DT_INT32;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SwigluGroupQuantTilingTest, tiling_error_zero_clamp_limit)
{
    TilingCase tc;
    tc.clampLimit = 0.0f;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}
} // namespace
