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
#include <vector>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "../../../../op_host/arch35/group_norm_tiling_arch35.h"

namespace {
constexpr int32_t CORE_NUM = 64;
constexpr int64_t UB_SIZE = 245760;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t VECTOR_LENGTH = 256;

struct GroupNormTilingDebugPrefix {
    int64_t numGroups;
    int64_t hwNum;
    int64_t elemNum;
    int64_t shapeC;
    int64_t shapeD;
    int64_t realCoreNum;
    int64_t numPerCore;
    int64_t numLastCore;
    int64_t processSize;
    int64_t loopNum;
    int64_t loopTail;
    int64_t innerLoopNum;
    int64_t innerLoopTail;
    int64_t tilingKey;
};

void InitPlatform(fe::PlatFormInfos& platformInfo, std::map<std::string, std::string>& socInfos,
                  std::map<std::string, std::string>& aicoreSpec, std::map<std::string, std::string>& intrinsics,
                  std::map<std::string, std::string>& socVersion)
{
    const std::string compileInfo = R"({
        "hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
        "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(compileInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    platformInfo.Init();
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const auto dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

ge::graphStatus RunTiling(const std::vector<int64_t>& xDims, int64_t numGroups, ge::DataType xDtype,
                          ge::DataType weightDtype, uint64_t* tilingKey = nullptr, int64_t ubSize = UB_SIZE,
                          uint32_t* blockDim = nullptr, int64_t* tilingDataRealCoreNum = nullptr,
                          int64_t* tilingDataKey = nullptr, float epsilon = 1e-4F)
{
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNorm");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    fe::PlatFormInfos platformInfo;
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatform(platformInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    optiling::GroupNormCompileInfo compileInfo{CORE_NUM, ubSize, BLOCK_SIZE, VECTOR_LENGTH};
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    int64_t channel = xDims.size() > 1 ? xDims[1] : 1;
    int64_t batch = xDims.empty() ? 1 : xDims[0];
    gert::StorageShape x = MakeStorageShape(xDims);
    gert::StorageShape gamma = MakeStorageShape({channel});
    gert::StorageShape beta = MakeStorageShape({channel});
    gert::StorageShape y = MakeStorageShape(xDims);
    gert::StorageShape mean = MakeStorageShape({batch, numGroups});
    gert::StorageShape variance = MakeStorageShape({batch, numGroups});

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &gamma, &beta})
                      .OutputShapes({&y, &mean, &variance})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(numGroups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(epsilon)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    ge::graphStatus status = opImpl->tiling(context);
    if (status == ge::GRAPH_SUCCESS && tilingKey != nullptr) {
        *tilingKey = context->GetTilingKey();
    }
    if (status == ge::GRAPH_SUCCESS && blockDim != nullptr) {
        *blockDim = context->GetBlockDim();
    }
    if (status == ge::GRAPH_SUCCESS && (tilingDataRealCoreNum != nullptr || tilingDataKey != nullptr)) {
        auto rawTilingData = context->GetRawTilingData();
        if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr ||
            rawTilingData->GetDataSize() < sizeof(GroupNormTilingDebugPrefix)) {
            return ge::GRAPH_FAILED;
        }
        const auto* debugData = reinterpret_cast<const GroupNormTilingDebugPrefix*>(rawTilingData->GetData());
        if (tilingDataRealCoreNum != nullptr) {
            *tilingDataRealCoreNum = debugData->realCoreNum;
        }
        if (tilingDataKey != nullptr) {
            *tilingDataKey = debugData->tilingKey;
        }
    }
    return status;
}
} // namespace

class GroupNormTiling : public testing::Test {};

TEST_F(GroupNormTiling, SelectsTwoPassPerformance)
{
    uint64_t key = 0;
    uint32_t blockDim = 0;
    int64_t tilingDataRealCoreNum = 0;
    int64_t tilingDataKey = 0;
    EXPECT_EQ(RunTiling({36, 48, 24, 1}, 6, ge::DT_FLOAT16, ge::DT_FLOAT16, &key, UB_SIZE, &blockDim,
                        &tilingDataRealCoreNum, &tilingDataKey),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1110);
    EXPECT_EQ(tilingDataKey, static_cast<int64_t>(key));
    EXPECT_EQ(tilingDataRealCoreNum, static_cast<int64_t>(blockDim));
}

TEST_F(GroupNormTiling, SelectsWelfordPerformance)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({1, 128, 128, 128}, 32, ge::DT_FLOAT, ge::DT_FLOAT, &key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1100);
}

TEST_F(GroupNormTiling, SelectsWelfordGeneralized)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({1, 5120, 32, 32}, 32, ge::DT_FLOAT, ge::DT_FLOAT, &key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1120);
}

TEST_F(GroupNormTiling, RejectsInsufficientUbForWelford)
{
    EXPECT_EQ(RunTiling({1, 128, 128, 128}, 32, ge::DT_FLOAT, ge::DT_FLOAT, nullptr, 256), ge::GRAPH_FAILED);
}

TEST_F(GroupNormTiling, SelectsTwoPassGeneralized)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({1, 32768, 8, 8}, 32768, ge::DT_FLOAT, ge::DT_FLOAT, &key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1130);
}

TEST_F(GroupNormTiling, RejectsEmptyReduction)
{
    EXPECT_EQ(RunTiling({2, 16, 0, 8}, 4, ge::DT_FLOAT16, ge::DT_FLOAT16), ge::GRAPH_FAILED);
}

TEST_F(GroupNormTiling, SupportsEmptyBatch)
{
    uint64_t key = 0;
    uint32_t blockDim = 1;
    int64_t tilingDataRealCoreNum = 1;
    int64_t tilingDataKey = 0;
    EXPECT_EQ(RunTiling({0, 16, 8, 8}, 4, ge::DT_FLOAT, ge::DT_FLOAT, &key, UB_SIZE, &blockDim, &tilingDataRealCoreNum,
                        &tilingDataKey),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1100);
    EXPECT_EQ(blockDim, 0U);
    EXPECT_EQ(tilingDataKey, static_cast<int64_t>(key));
    EXPECT_EQ(tilingDataRealCoreNum, 0);
}

TEST_F(GroupNormTiling, RejectsEmptyChannel)
{
    EXPECT_EQ(RunTiling({2, 0, 8, 8}, 4, ge::DT_FLOAT, ge::DT_FLOAT), ge::GRAPH_FAILED);
}

TEST_F(GroupNormTiling, RejectsMixedDtype)
{
    EXPECT_EQ(RunTiling({1, 16, 8, 8}, 4, ge::DT_FLOAT16, ge::DT_FLOAT), ge::GRAPH_FAILED);
}

TEST_F(GroupNormTiling, RejectsInvalidNumGroups)
{
    EXPECT_EQ(RunTiling({1, 16, 8, 8}, 0, ge::DT_FLOAT, ge::DT_FLOAT), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({1, 15, 8, 8}, 4, ge::DT_FLOAT, ge::DT_FLOAT), ge::GRAPH_FAILED);
}

TEST_F(GroupNormTiling, RejectsNonPositiveEpsilon)
{
    EXPECT_EQ(
        RunTiling({1, 16, 8, 8}, 4, ge::DT_FLOAT, ge::DT_FLOAT, nullptr, UB_SIZE, nullptr, nullptr, nullptr, 0.0F),
        ge::GRAPH_FAILED);
    EXPECT_EQ(
        RunTiling({1, 16, 8, 8}, 4, ge::DT_FLOAT, ge::DT_FLOAT, nullptr, UB_SIZE, nullptr, nullptr, nullptr, -1e-4F),
        ge::GRAPH_FAILED);
}

TEST_F(GroupNormTiling, RejectsRankOneInput)
{
    EXPECT_EQ(RunTiling({16}, 4, ge::DT_FLOAT, ge::DT_FLOAT), ge::GRAPH_FAILED);
}
