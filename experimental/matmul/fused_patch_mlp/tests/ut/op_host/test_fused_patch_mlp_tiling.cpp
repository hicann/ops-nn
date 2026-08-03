/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <map>
#include <string>

#include <gtest/gtest.h>

#include "../../../op_host/fused_patch_mlp_tiling.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"

using namespace ut_util;

namespace {

constexpr uint64_t SYS_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint32_t GELU_MODE_ROW = 0U;
constexpr uint32_t GELU_MODE_FLAT = 1U;

struct TilingPayload {
    uint32_t totalN;
    uint32_t inFeatures;
    uint32_t hiddenSize;
    uint32_t geluTileSize;
    uint32_t geluMode;
    uint32_t numLayers;
    TCubeTiling mm0Tiling;
    TCubeTiling mmHTiling;
};

class FusedPatchMlpTilingTest : public testing::Test {};

void RunTilingCase(ge::DataType dtype, int64_t expectedKey, size_t rows, uint32_t expectedGeluMode, size_t hidden = 256,
                   size_t numLayers = 3, size_t inFeatures = 64, uint32_t expectedHiddenSingleM = 0,
                   uint32_t expectedHiddenSingleN = 0, uint32_t expectedGeluTileSize = 0,
                   uint32_t expectedHiddenStepK = 0, uint32_t expectedHiddenDepth = 0)
{
    const size_t totalWeights = inFeatures * hidden + (numLayers - 1) * hidden * hidden;
    const size_t totalBias = numLayers * hidden;

    gert::StorageShape xShape = {{rows, inFeatures}, {rows, inFeatures}};
    gert::StorageShape weightsShape = {{totalWeights}, {totalWeights}};
    gert::StorageShape biasesShape = {{totalBias}, {totalBias}};
    gert::StorageShape yShape = {{rows, hidden}, {rows, hidden}};

    const std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown",
          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false,
          "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true,
          "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288,
          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
          "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B"}
        })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::FusedPatchMlpCompileInfo compileInfo;

    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedPatchMlp");
    ASSERT_NE(opImpl, nullptr);
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    auto* parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    ASSERT_TRUE(parseContext->GetPlatformInfo()->Init());
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(opImpl->tiling_parse(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    ASSERT_NE(tilingData, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &weightsShape, &biasesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeAttrs({{"num_layers", Ops::NN::AnyValue::CreateFrom<int64_t>(numLayers)}})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype == ge::DT_BF16 ? ge::DT_FLOAT : dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context->GetPlatformInfo(), nullptr);
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(opImpl->tiling(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetTilingKey(), expectedKey);

    const auto* payload = reinterpret_cast<const TilingPayload*>(context->GetRawTilingData()->GetData());
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->totalN, rows);
    EXPECT_EQ(payload->hiddenSize, hidden);
    EXPECT_EQ(payload->geluMode, expectedGeluMode);
    EXPECT_GT(payload->geluTileSize, 0U);
    if (expectedGeluTileSize != 0U) {
        EXPECT_EQ(payload->geluTileSize, expectedGeluTileSize);
    }
    if (expectedGeluMode == GELU_MODE_ROW) {
        EXPECT_EQ(payload->geluTileSize, hidden);
    } else {
        const uint32_t elemsPerBlock = dtype == ge::DT_FLOAT ? 8U : 16U;
        EXPECT_EQ(payload->geluTileSize % elemsPerBlock, 0U);
        EXPECT_LT(payload->geluTileSize, rows * hidden);
    }
    if (expectedHiddenSingleM != 0U) {
        EXPECT_EQ(payload->mmHTiling.singleCoreM, expectedHiddenSingleM);
    }
    if (expectedHiddenSingleN != 0U) {
        EXPECT_EQ(payload->mmHTiling.singleCoreN, expectedHiddenSingleN);
    }
    if (expectedHiddenStepK != 0U) {
        EXPECT_EQ(payload->mmHTiling.stepKa, expectedHiddenStepK);
        EXPECT_EQ(payload->mmHTiling.stepKb, expectedHiddenStepK);
    }
    if (expectedHiddenDepth != 0U) {
        EXPECT_EQ(payload->mmHTiling.depthA1, expectedHiddenDepth);
        EXPECT_EQ(payload->mmHTiling.depthB1, expectedHiddenDepth);
    }

    const uint64_t dtypeSize = dtype == ge::DT_FLOAT ? 4UL : 2UL;
    const uint64_t expectedWorkspace = numLayers == 1 ? 0UL : SYS_WORKSPACE + 2UL * rows * hidden * dtypeSize;
    EXPECT_EQ(context->GetWorkspaceSizes(1)[0], expectedWorkspace);
}

TEST_F(FusedPatchMlpTilingTest, SmallShapeKeepsRowWiseFp16Path) { RunTilingCase(ge::DT_FLOAT16, 1, 4, GELU_MODE_ROW); }

TEST_F(FusedPatchMlpTilingTest, DtypesShareExecutionPathKey)
{
    RunTilingCase(ge::DT_BF16, 1, 4, GELU_MODE_ROW);
    RunTilingCase(ge::DT_FLOAT, 1, 4, GELU_MODE_ROW);
}

TEST_F(FusedPatchMlpTilingTest, LargeShapeUsesFlatUbTiling)
{
    // 192 KiB UB: the three-FP32-buffer half path uses an 8192-element tile; FP32 remains at 4096.
    RunTilingCase(ge::DT_FLOAT16, 1, 4096, GELU_MODE_FLAT, 256, 3, 64, 0, 0, 8192);
    RunTilingCase(ge::DT_FLOAT, 1, 4096, GELU_MODE_FLAT, 256, 3, 64, 0, 0, 4096);
}

TEST_F(FusedPatchMlpTilingTest, LargeHiddenUsesMdlMatmulPath)
{
    RunTilingCase(ge::DT_FLOAT16, 31, 4096, GELU_MODE_FLAT, 5120, 3, 16, 512, 256, 0, 4, 8);
    RunTilingCase(ge::DT_BF16, 31, 4096, GELU_MODE_FLAT, 5120, 3, 16, 512, 256, 0, 4, 8);
}

TEST_F(FusedPatchMlpTilingTest, TwoLayersKeepStableNonPipelinedMdlPath)
{
    RunTilingCase(ge::DT_FLOAT16, 11, 4096, GELU_MODE_FLAT, 5120, 2, 16, 512, 256, 0, 4, 8);
}

TEST_F(FusedPatchMlpTilingTest, Hidden4096UsesLargeMWeightReuseGroup)
{
    RunTilingCase(ge::DT_FLOAT16, 11, 4096, GELU_MODE_FLAT, 4096, 3, 16, 512, 256, 0, 4, 8);
}

TEST_F(FusedPatchMlpTilingTest, SingleLayerUsesAicOnlyWithoutWorkspace)
{
    RunTilingCase(ge::DT_FLOAT16, 21, 4096, GELU_MODE_FLAT, 5120, 1, 16);
}

} // namespace
