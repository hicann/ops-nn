/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/hard_sigmoid_tiling_data.h"

using namespace ut_util;

namespace optiling {
struct HardSigmoidCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling

namespace {
constexpr const char* COMPILE_INFO = R"({
  "hardware_info": {
    "BT_SIZE": 0,
    "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false,
    "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true,
    "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 245760,
    "L2_SIZE": 33554432,
    "L1_SIZE": 524288,
    "L0A_SIZE": 65536,
    "L0B_SIZE": 65536,
    "L0C_SIZE": 131072,
    "CORE_NUM": 64
  }
})";

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint32_t blockDim = 0;
    HardSigmoidTilingData tilingData{};
};

TilingResult RunTilingCase(const std::vector<int64_t>& dims, ge::DataType dtype, float alpha = 1.0f / 6.0f,
                           float beta = 0.5f)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> version = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::HardSigmoidCompileInfo compileInfo;
    TilingResult result;

    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("HardSigmoid");
    if (impl == nullptr || impl->tiling_parse == nullptr || impl->tiling == nullptr) {
        ADD_FAILURE() << "HardSigmoid tiling callbacks are not registered";
        return result;
    }
    auto parseContextHolder = gert::KernelRunContextFaker()
                                  .KernelIONum(1, 1)
                                  .Inputs({const_cast<char*>(COMPILE_INFO), reinterpret_cast<void*>(&platformInfo)})
                                  .Outputs({&compileInfo})
                                  .Build();
    auto* parseContext = parseContextHolder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr || parseContext->GetPlatformInfo() == nullptr ||
        !parseContext->GetPlatformInfo()->Init()) {
        ADD_FAILURE() << "Failed to create the tiling parse context";
        return result;
    }
    auto* parsePlatformInfo = parseContext->GetPlatformInfo();
    parsePlatformInfo->SetPlatformRes("SoCInfo", socInfos);
    parsePlatformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    parsePlatformInfo->SetCoreNumByCoreType("AICore");
    parsePlatformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parsePlatformInfo->SetPlatformRes("version", version);
    if (impl->tiling_parse(parseContextHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        ADD_FAILURE() << "HardSigmoid tiling parse failed";
        return result;
    }

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    if (tilingData == nullptr || workspaceHolder == nullptr) {
        ADD_FAILURE() << "Failed to allocate tiling test buffers";
        return result;
    }
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto contextHolder = gert::TilingContextFaker()
                             .SetOpType("HardSigmoid")
                             .NodeIoNum(1, 1)
                             .IrInstanceNum({1})
                             .InputShapes({&shape})
                             .OutputShapes({&shape})
                             .CompileInfo(&compileInfo)
                             .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                             .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(alpha)},
                                         {"beta", Ops::NN::AnyValue::CreateFrom<float>(beta)}})
                             .TilingData(tilingData.get())
                             .Workspace(workspace)
                             .Build();
    auto* context = contextHolder.GetContext<gert::TilingContext>();
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        ADD_FAILURE() << "Failed to create the tiling context";
        return result;
    }
    auto* tilingPlatformInfo = context->GetPlatformInfo();
    tilingPlatformInfo->SetPlatformRes("SoCInfo", socInfos);
    tilingPlatformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingPlatformInfo->SetCoreNumByCoreType("AICore");
    tilingPlatformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    result.status = impl->tiling(context);
    result.blockDim = context->GetBlockDim();
    auto* rawTilingData = context->GetRawTilingData();
    if (result.status == ge::GRAPH_SUCCESS && rawTilingData != nullptr &&
        rawTilingData->GetDataSize() >= sizeof(HardSigmoidTilingData)) {
        std::memcpy(&result.tilingData, rawTilingData->GetData(), sizeof(HardSigmoidTilingData));
    }
    return result;
}
} // namespace

TEST(HardSigmoidTilingTest, Float32UsesIndependentCoreAndUbSplits)
{
    const auto result = RunTilingCase({262144}, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 64U);
    EXPECT_EQ(result.tilingData.totalElements, 262144);
    EXPECT_EQ(result.tilingData.blockFactor, 4096);
    EXPECT_EQ(result.tilingData.ubFactor, 14848);
}

TEST(HardSigmoidTilingTest, Float16)
{
    const auto result = RunTilingCase({3, 257}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 1U);
    EXPECT_EQ(result.tilingData.blockFactor, 771);
    EXPECT_EQ(result.tilingData.ubFactor, 19712);
}

TEST(HardSigmoidTilingTest, Bfloat16)
{
    const auto result = RunTilingCase({2, 7, 65}, ge::DT_BF16);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 1U);
    EXPECT_EQ(result.tilingData.blockFactor, 910);
    EXPECT_EQ(result.tilingData.ubFactor, 19712);
}

TEST(HardSigmoidTilingTest, Int32)
{
    const auto result = RunTilingCase({129}, ge::DT_INT32);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 1U);
    EXPECT_EQ(result.tilingData.blockFactor, 129);
    EXPECT_EQ(result.tilingData.ubFactor, 11840);
}

TEST(HardSigmoidTilingTest, CustomAttributes)
{
    const auto result = RunTilingCase({1024}, ge::DT_FLOAT, 0.2f, 0.4f);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 1U);
    EXPECT_FLOAT_EQ(result.tilingData.alpha, 0.2f);
    EXPECT_FLOAT_EQ(result.tilingData.beta, 0.4f);
}

TEST(HardSigmoidTilingTest, EmptyTensor)
{
    const auto result = RunTilingCase({0, 4}, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 1U);
    EXPECT_EQ(result.tilingData.totalElements, 0);
    EXPECT_EQ(result.tilingData.blockFactor, 0);
    EXPECT_EQ(result.tilingData.ubFactor, 0);
}

TEST(HardSigmoidTilingTest, UnsupportedDtype) { EXPECT_EQ(RunTilingCase({128}, ge::DT_INT8).status, ge::GRAPH_FAILED); }

TEST(HardSigmoidTilingTest, TwoMinimumCopyChunksUseTwoCores)
{
    const auto result = RunTilingCase({8192}, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 2U);
    EXPECT_EQ(result.tilingData.blockFactor, 4096);
}

TEST(HardSigmoidTilingTest, JustBelowTwoMinimumCopyChunksUsesOneCore)
{
    const auto result = RunTilingCase({8191}, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 1U);
    EXPECT_EQ(result.tilingData.blockFactor, 8191);
}

TEST(HardSigmoidTilingTest, RejectsNegativeDimensions)
{
    EXPECT_EQ(RunTilingCase({-1, -1}, ge::DT_FLOAT).status, ge::GRAPH_FAILED);
}
