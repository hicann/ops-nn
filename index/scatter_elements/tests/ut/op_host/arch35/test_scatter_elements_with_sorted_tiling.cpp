/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <string>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/scatter_elements_tiling.h"

namespace {
constexpr size_t ASCENDC_TOOLS_WORKSPACE = 16777216;

struct WithSortedTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    uint64_t indicesTotalNum = 0;
    uint64_t keySize = 0;
    uint64_t permSize = 0;
    int32_t countMode = -1;
    int32_t shapeMode = -1;
    uint64_t wsSrcPosOff = 0;
};

void RunWithSortedTilingCase(ge::DataType inputDtype, ge::DataType indicesDtype, gert::StorageShape& inputShape,
                             gert::StorageShape& indicesShape, int64_t axis, const std::string& reduction,
                             int32_t deterministic, WithSortedTilingResult& result)
{
    const std::string opType("ScatterElements");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    ASSERT_NE(opImpl, nullptr);
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    const std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> npuArch = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::ScatterElementsCompileInfo compileInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    auto parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    ASSERT_TRUE(parseContext->GetPlatformInfo()->Init());
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parseContext->GetPlatformInfo()->SetPlatformRes("version", npuArch);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    ASSERT_NE(tilingData, nullptr);
    gert::StorageShape updatesShape = indicesShape;
    gert::StorageShape outputShape = inputShape;
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &indicesShape, &updatesShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .DeterministicInfo(deterministic)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto tilingContext = holder.GetContext<gert::TilingContext>();
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", npuArch);

    result.status = tilingFunc(tilingContext);
    if (result.status != ge::GRAPH_SUCCESS) {
        return;
    }
    result.tilingKey = tilingContext->GetTilingKey();
    result.blockDim = tilingContext->GetBlockDim();
    result.workspaceSize = *tilingContext->GetWorkspaceSizes(0);

    auto raw = tilingContext->GetRawTilingData();
    ASSERT_NE(raw, nullptr);
    ASSERT_NE(raw->GetData(), nullptr);
    const auto* p64 = reinterpret_cast<const int64_t*>(raw->GetData());
    const auto* p32 = reinterpret_cast<const uint32_t*>(raw->GetData());
    result.indicesTotalNum = static_cast<uint64_t>(p64[35]);
    result.keySize = static_cast<uint64_t>(p64[36]);
    result.permSize = static_cast<uint64_t>(p64[37]);
    result.countMode = static_cast<int32_t>(p32[76]);
    result.shapeMode = static_cast<int32_t>(p32[77]);
    result.wsSrcPosOff = static_cast<uint64_t>(p64[46]);
}
} // namespace

TEST(ScatterElementsWithSortedTiling, DeterministicAddUsesSortedRoute)
{
    gert::StorageShape dataShape = {{130000}, {130000}};
    gert::StorageShape indicesShape = {{100000}, {100000}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, dataShape, indicesShape, 0, "add", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 2000100UL);
    EXPECT_GT(result.blockDim, 1U);
    EXPECT_GT(result.workspaceSize, ASCENDC_TOOLS_WORKSPACE);
    EXPECT_EQ(result.indicesTotalNum, 100000UL);
    EXPECT_EQ(result.keySize, 4UL);
    EXPECT_EQ(result.permSize, 4UL);
    EXPECT_EQ(result.shapeMode, 1);
    EXPECT_GT(result.wsSrcPosOff, 0UL);
}

TEST(ScatterElementsWithSortedTiling, Int64IndicesUseSortedRoute)
{
    gert::StorageShape dataShape = {{130000}, {130000}};
    gert::StorageShape indicesShape = {{100000}, {100000}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_FLOAT16, ge::DT_INT64, dataShape, indicesShape, 0, "add", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 2001101UL);
}

TEST(ScatterElementsWithSortedTiling, LargeCountUses64BitPermutation)
{
    constexpr int64_t largeCount = (1LL << 30) + 1;
    gert::StorageShape dataShape = {{largeCount}, {largeCount}};
    gert::StorageShape indicesShape = {{largeCount}, {largeCount}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, dataShape, indicesShape, 0, "add", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.countMode, 1);
    EXPECT_EQ(result.permSize, 8UL);
}

TEST(ScatterElementsWithSortedTiling, DeterministicNoneUsesSortedRoute)
{
    gert::StorageShape dataShape = {{100000}, {100000}};
    gert::StorageShape indicesShape = {{100000}, {100000}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_UINT8, ge::DT_INT32, dataShape, indicesShape, 0, "none", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 2000001UL);
    EXPECT_EQ(result.shapeMode, 0);
    EXPECT_EQ(result.wsSrcPosOff, 0UL);
}

TEST(ScatterElementsWithSortedTiling, NonDominantScatterAxisKeepsOriginalRoute)
{
    gert::StorageShape dataShape = {{64, 64}, {64, 64}};
    gert::StorageShape indicesShape = {{64, 64}, {64, 64}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, dataShape, indicesShape, 0, "add", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1000100UL);
    EXPECT_EQ(result.workspaceSize, ASCENDC_TOOLS_WORKSPACE);
}

TEST(ScatterElementsWithSortedTiling, WellParallelizedFloatRouteKeepsOriginalRoute)
{
    gert::StorageShape dataShape = {{128, 6400}, {128, 6400}};
    gert::StorageShape indicesShape = {{128, 6400}, {128, 6400}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, dataShape, indicesShape, 1, "add", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1000100UL);
    EXPECT_EQ(result.workspaceSize, ASCENDC_TOOLS_WORKSPACE);
}

TEST(ScatterElementsWithSortedTiling, MultiDimensionalInt32KeepsOriginalRoute)
{
    gert::StorageShape dataShape = {{1, 100000}, {1, 100000}};
    gert::StorageShape indicesShape = {{1, 100000}, {1, 100000}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_INT32, ge::DT_INT32, dataShape, indicesShape, 1, "add", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1000103UL);
    EXPECT_EQ(result.workspaceSize, ASCENDC_TOOLS_WORKSPACE);
}

TEST(ScatterElementsWithSortedTiling, EqualIndexAndAxisParallelismKeepsOriginalRoute)
{
    gert::StorageShape dataShape = {{1024}, {1024}};
    gert::StorageShape indicesShape = {{1024}, {1024}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, dataShape, indicesShape, 0, "add", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1000100UL);
    EXPECT_EQ(result.workspaceSize, ASCENDC_TOOLS_WORKSPACE);
}

TEST(ScatterElementsWithSortedTiling, FloatNoneKeepsOriginalRoute)
{
    gert::StorageShape dataShape = {{100000}, {100000}};
    gert::StorageShape indicesShape = {{100000}, {100000}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, dataShape, indicesShape, 0, "none", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1000004UL);
    EXPECT_EQ(result.workspaceSize, ASCENDC_TOOLS_WORKSPACE);
}

TEST(ScatterElementsWithSortedTiling, MoreIndexParallelismUsesSortedRoute)
{
    gert::StorageShape dataShape = {{2048}, {2048}};
    gert::StorageShape indicesShape = {{2048}, {2048}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, dataShape, indicesShape, 0, "add", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 2000100UL);
    EXPECT_GT(result.workspaceSize, ASCENDC_TOOLS_WORKSPACE);
}

TEST(ScatterElementsWithSortedTiling, EightDimensionalUint8UsesSortedRoute)
{
    gert::StorageShape dataShape = {{1, 1, 1, 1, 1, 1, 1, 16384}, {1, 1, 1, 1, 1, 1, 1, 16384}};
    gert::StorageShape indicesShape = {{1, 1, 1, 1, 1, 1, 1, 8192}, {1, 1, 1, 1, 1, 1, 1, 8192}};
    WithSortedTilingResult result;
    RunWithSortedTilingCase(ge::DT_UINT8, ge::DT_INT32, dataShape, indicesShape, 7, "none", 1, result);

    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 2000001UL);
    EXPECT_GT(result.workspaceSize, ASCENDC_TOOLS_WORKSPACE);
}

TEST(ScatterElementsWithSortedTiling, UnsupportedModesKeepOriginalRoute)
{
    gert::StorageShape dataShape = {{130000}, {130000}};
    gert::StorageShape indicesShape = {{100000}, {100000}};
    WithSortedTilingResult nonDeterm;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, dataShape, indicesShape, 0, "add", 0, nonDeterm);
    EXPECT_EQ(nonDeterm.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(nonDeterm.tilingKey, 1000100UL);

    WithSortedTilingResult mul;
    RunWithSortedTilingCase(ge::DT_FLOAT, ge::DT_INT32, dataShape, indicesShape, 0, "mul", 1, mul);
    EXPECT_EQ(mul.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(mul.tilingKey, 1000200UL);
}
