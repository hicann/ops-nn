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
 * \file test_cosine_embedding_loss_tiling.cpp
 * \brief CosineEmbeddingLoss arch35 tiling UT.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/cosine_embedding_loss_tiling.h"
#include "../../../../op_kernel/arch35/cosine_embedding_loss_tiling_data.h"

using namespace ge;
using namespace ut_util;

namespace {
constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t REDUCTION_SUM = 1;
constexpr uint32_t REDUCTION_MEAN = 2;
constexpr int64_t ASCEND950_VECTOR_CORE_NUM = 64;
constexpr size_t SYSTEM_WORKSPACE_BYTES = 16UL * 1024UL * 1024UL;

void InitAscend950Platform(fe::PlatFormInfos& platformInfo, std::map<std::string, std::string>& socInfos,
                           std::map<std::string, std::string>& aicoreSpec,
                           std::map<std::string, std::string>& intrinsics,
                           std::map<std::string, std::string>& socVersion)
{
    std::string compileInfoString = R"({
        "hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "CORE_NUM": 64, "socVersion": "Ascend950"}
    })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    platformInfo.Init();
}

void AppendShape(gert::StorageShape& shape, const std::vector<int64_t>& dims)
{
    for (auto dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
}

void RunTilingCase(const std::vector<int64_t>& x1Dims, const std::vector<int64_t>& x2Dims,
                   const std::vector<int64_t>& targetDims, const std::vector<int64_t>& yDims,
                   const std::vector<int64_t>& tilingOutputDims, ge::DataType x1Dtype, ge::DataType x2Dtype,
                   ge::DataType targetDtype, float margin, const std::string& reduction, bool expectSuccess,
                   uint32_t expectReduction, int64_t expectN = 0, int64_t expectD = 0,
                   uint32_t expectFastPath = COSINE_EMBEDDING_LOSS_GENERIC_PATH)
{
    fe::PlatFormInfos platformInfo;
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersion = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    InitAscend950Platform(platformInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("CosineEmbeddingLoss");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    gert::StorageShape xShape;
    gert::StorageShape x2Shape;
    gert::StorageShape targetShape;
    gert::StorageShape yShape;
    AppendShape(xShape, x1Dims);
    AppendShape(x2Shape, x2Dims);
    AppendShape(targetShape, targetDims);
    AppendShape(yShape, yDims);

    optiling::CosineEmbeddingLossCompileInfo compileInfo;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &x2Shape, &targetShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, targetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"margin", Ops::NN::AnyValue::CreateFrom<float>(margin)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    if (!expectSuccess) {
        EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
        return;
    }

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingContext->GetTilingKey(), 0);
    auto* rawTiling = tilingContext->GetRawTilingData();
    ASSERT_NE(rawTiling, nullptr);
    const auto* td = reinterpret_cast<const CosineEmbeddingLossTilingData*>(rawTiling->GetData());
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->n, expectN);
    EXPECT_EQ(td->d, expectD);
    EXPECT_EQ(td->dAlign, ((expectD + 15) / 16) * 16);
    EXPECT_GT(td->x1Num, 0);
    EXPECT_GT(td->x2Num, 0);
    EXPECT_GT(td->targetNum, 0);
    EXPECT_GT(td->ubTileRows, 0);
    EXPECT_EQ(td->reduction, expectReduction);
    ASSERT_EQ(td->outputRank, tilingOutputDims.size());
    for (size_t i = 0; i < tilingOutputDims.size(); ++i) {
        EXPECT_EQ(td->outputShape[i], tilingOutputDims[i]);
    }
    EXPECT_FLOAT_EQ(td->margin, margin);
    int64_t rowsPerCore = (expectN + ASCEND950_VECTOR_CORE_NUM - 1) / ASCEND950_VECTOR_CORE_NUM;
    int64_t expectedUsedCoreNum = (expectN + rowsPerCore - 1) / rowsPerCore;
    if (expectFastPath == COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH && expectReduction == REDUCTION_NONE &&
        expectedUsedCoreNum > 1) {
        rowsPerCore = ((rowsPerCore + 7) / 8) * 8;
        expectedUsedCoreNum = (expectN + rowsPerCore - 1) / rowsPerCore;
    }
    EXPECT_EQ(td->rowsPerCore, rowsPerCore);
    EXPECT_EQ(td->usedCoreNum, expectedUsedCoreNum);
    EXPECT_EQ(tilingContext->GetBlockDim(), static_cast<uint32_t>(expectedUsedCoreNum));
    EXPECT_EQ(tilingContext->GetScheduleMode(), expectReduction == REDUCTION_NONE ? 0U : 1U);
    auto* workspaceSizes = tilingContext->GetWorkspaceSizes(1);
    ASSERT_NE(workspaceSizes, nullptr);
    EXPECT_EQ(workspaceSizes[0], SYSTEM_WORKSPACE_BYTES + static_cast<size_t>(expectedUsedCoreNum) * 32UL);
    EXPECT_EQ(td->fastPath, expectFastPath);
    if (expectFastPath == COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH) {
        EXPECT_GE(td->featureTile, 64);
        EXPECT_EQ(td->featureTile % 64, 0);
        EXPECT_EQ(td->reduceTmpBytes, 32);
    } else {
        EXPECT_EQ(td->featureTile, 0);
        EXPECT_EQ(td->reduceTmpBytes, 0);
    }
    EXPECT_EQ(td->ubTileRows, std::min(rowsPerCore, COSINE_EMBEDDING_LOSS_MAX_UB_TILE_ROWS));
    if (expectReduction == REDUCTION_MEAN) {
        EXPECT_NEAR(td->meanCoef, 1.0f / static_cast<float>(expectN), 1e-9f);
    }
    if (expectReduction == REDUCTION_SUM) {
        EXPECT_EQ(td->meanCoef, 1.0f);
    }
}
} // namespace

class CosineEmbeddingLossArch35Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CosineEmbeddingLossArch35Tiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "CosineEmbeddingLossArch35Tiling TearDown" << std::endl; }
};

TEST_F(CosineEmbeddingLossArch35Tiling, none_fp32)
{
    RunTilingCase({2, 3, 4}, {2, 3, 4}, {2, 4}, {2, 4}, {2, 4}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f, "none",
                  true, REDUCTION_NONE, 8, 3);
}

TEST_F(CosineEmbeddingLossArch35Tiling, mean_fp16_broadcast_x)
{
    RunTilingCase({1, 3, 4}, {2, 3, 1}, {4}, {1}, {2, 4}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, 0.5f, "mean",
                  true, REDUCTION_MEAN, 8, 3);
}

TEST_F(CosineEmbeddingLossArch35Tiling, sum_int32_target_adds_leading_broadcast_dim)
{
    RunTilingCase({3}, {2, 3}, {2, 2}, {1}, {2, 2}, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, -0.1f, "sum", true,
                  REDUCTION_SUM, 4, 3);
}

TEST_F(CosineEmbeddingLossArch35Tiling, mixed_target_dtype)
{
    RunTilingCase({2, 3}, {2, 3}, {2}, {2}, {2}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, 0.0f, "none", true,
                  REDUCTION_NONE, 2, 3, COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH);
}

TEST_F(CosineEmbeddingLossArch35Tiling, none_fast_path_uses_32b_multicore_boundaries)
{
    RunTilingCase({130, 3}, {130, 3}, {130}, {130}, {130}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, "none", true,
                  REDUCTION_NONE, 130, 3, COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH);
}

TEST_F(CosineEmbeddingLossArch35Tiling, invalid_mixed_x_dtype)
{
    RunTilingCase({2, 3}, {2, 3}, {2}, {}, {}, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, 0.0f, "none", false,
                  REDUCTION_NONE);
}

TEST_F(CosineEmbeddingLossArch35Tiling, invalid_target_shape)
{
    RunTilingCase({2, 3, 4}, {2, 3, 4}, {3, 4}, {}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, "none", false,
                  REDUCTION_NONE);
}

TEST_F(CosineEmbeddingLossArch35Tiling, invalid_x_broadcast)
{
    RunTilingCase({2, 3, 4}, {2, 4, 4}, {2, 4}, {}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, "none", false,
                  REDUCTION_NONE);
}

TEST_F(CosineEmbeddingLossArch35Tiling, invalid_broadcast_rank)
{
    RunTilingCase({3}, {3}, {1}, {}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, "none", false, REDUCTION_NONE);
}

TEST_F(CosineEmbeddingLossArch35Tiling, invalid_reduction)
{
    RunTilingCase({2, 3}, {2, 3}, {2}, {}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, "bogus", false,
                  REDUCTION_NONE);
}

TEST_F(CosineEmbeddingLossArch35Tiling, invalid_rank_zero_target)
{
    RunTilingCase({2, 3}, {2, 3}, {}, {}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, "none", false,
                  REDUCTION_NONE);
}

TEST_F(CosineEmbeddingLossArch35Tiling, invalid_zero_dimension)
{
    RunTilingCase({2, 0}, {2, 1}, {2}, {}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, "none", false,
                  REDUCTION_NONE);
}

TEST_F(CosineEmbeddingLossArch35Tiling, invalid_feature_alignment_overflow)
{
    RunTilingCase({1, std::numeric_limits<int64_t>::max()}, {1, std::numeric_limits<int64_t>::max()}, {1}, {}, {},
                  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, "none", false, REDUCTION_NONE);
}

TEST_F(CosineEmbeddingLossArch35Tiling, large_n_uses_row_tile_split)
{
    RunTilingCase({8, 9206, 9, 1923}, {6, 4, 6, 5, 1, 9206, 9, 1}, {1, 9, 1923}, {1}, {6, 6, 5, 8, 9206, 9, 1923},
                  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, -0.7960737922501269f, "sum", true, REDUCTION_SUM,
                  229432668480LL, 4, COSINE_EMBEDDING_LOSS_FEATURE_BROADCAST_REDUCTION_PATH);
}
