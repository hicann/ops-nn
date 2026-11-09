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
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_kernel/arch35/lp_norm_update_v2_tiling_data.h"

using namespace std;

namespace {
struct LpNormUpdateV2CompileInfo {};

const char* const kCompileInfo = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                      "Intrinsic_fix_pipe_l0c2out": false,
                      "Intrinsic_data_move_l12ub": true,
                      "Intrinsic_data_move_l0c2ub": true,
                      "Intrinsic_data_move_out2l1_nd2nz": false,
                      "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                      "CORE_NUM": 64}})";

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    int64_t tilingKey = -1;
    uint32_t blockDim = 0;
    LpNormUpdateV2TilingData data;
};

TilingResult RunTiling(const std::vector<int64_t>& shape, ge::DataType dtype, float p, float epsilon)
{
    TilingResult result;

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    LpNormUpdateV2CompileInfo compileInfo;
    auto param = gert::TilingData::CreateCap(4096);
    EXPECT_NE(param, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    gert::Shape shapeVec;
    shapeVec.SetDimNum(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        shapeVec.SetDim(i, shape[i]);
    }
    gert::StorageShape xShape;
    xShape.MutableOriginShape() = shapeVec;
    xShape.MutableStorageShape() = shapeVec;
    gert::StorageShape yShape;
    yShape.MutableOriginShape() = shapeVec;
    yShape.MutableStorageShape() = shapeVec;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"p", Ops::NN::AnyValue::CreateFrom<float>(p)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)}})
                      .TilingData(param.get())
                      .Workspace(workspace)
                      .Build();

    auto* context = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(context, nullptr);
    if (context == nullptr) {
        return result;
    }
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("LpNormUpdateV2");
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return result;
    }

    result.status = opImpl->tiling(context);
    result.tilingKey = context->GetTilingKey();
    result.blockDim = context->GetBlockDim();
    if (result.status == ge::GRAPH_SUCCESS) {
        auto* tilingData = context->GetTilingData<LpNormUpdateV2TilingData>();
        if (tilingData != nullptr) {
            result.data = *tilingData;
        }
    }
    return result;
}

constexpr float kIntSentinelMax = static_cast<float>(std::numeric_limits<int32_t>::max());
constexpr float kIntSentinelMin = static_cast<float>(std::numeric_limits<int32_t>::min());
constexpr float kFltMax = std::numeric_limits<float>::max();
} // namespace

class LpNormUpdateV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LpNormUpdateV2TilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "LpNormUpdateV2TilingTest TearDown" << std::endl; }
};

TEST_F(LpNormUpdateV2TilingTest, identity_int_sentinel_and_inf)
{
    for (float p : {kIntSentinelMax, kIntSentinelMin, std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(), 0.0f, 1.0f}) {
        auto result = RunTiling({4}, ge::DT_FLOAT, p, 0.0f);
        EXPECT_EQ(result.status, ge::GRAPH_SUCCESS) << "p=" << p;
        EXPECT_EQ(result.tilingKey, 0) << "p=" << p;
        EXPECT_EQ(result.data.epsilonEff, 0.0f) << "p=" << p;
    }
}

TEST_F(LpNormUpdateV2TilingTest, pow_for_nearby_and_large_finite_p)
{
    for (float p : {2147483520.0f, 1e30f, 3.0f, kFltMax, -kFltMax}) {
        auto result = RunTiling({4}, ge::DT_FLOAT, p, 0.0f);
        EXPECT_EQ(result.status, ge::GRAPH_SUCCESS) << "p=" << p;
        EXPECT_EQ(result.tilingKey, 2) << "p=" << p;
    }
}

TEST_F(LpNormUpdateV2TilingTest, sqrt_branch_for_p_two)
{
    auto result = RunTiling({4}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 1);
}

TEST_F(LpNormUpdateV2TilingTest, epsilon_promotion_fp16)
{
    auto zero = RunTiling({4}, ge::DT_FLOAT16, 2.0f, 0.0f);
    EXPECT_EQ(zero.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(zero.data.epsilonEff, 0.0f);

    auto subFloor = RunTiling({4}, ge::DT_FLOAT16, 2.0f, 1e-8f);
    EXPECT_EQ(subFloor.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(subFloor.data.epsilonEff, 1e-7f);

    auto atFloor = RunTiling({4}, ge::DT_FLOAT16, 2.0f, 1e-7f);
    EXPECT_EQ(atFloor.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(atFloor.data.epsilonEff, 1e-7f);

    auto aboveFloor = RunTiling({4}, ge::DT_FLOAT16, 2.0f, 1e-6f);
    EXPECT_EQ(aboveFloor.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(aboveFloor.data.epsilonEff, 1e-6f);

    auto negative = RunTiling({4}, ge::DT_FLOAT16, 2.0f, -1.0f);
    EXPECT_EQ(negative.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(negative.data.epsilonEff, -1.0f);

    auto fp32Zero = RunTiling({4}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(fp32Zero.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(fp32Zero.data.epsilonEff, 0.0f);
}

TEST_F(LpNormUpdateV2TilingTest, invalid_shapes_rejected)
{
    auto negativeDim = RunTiling({-1, 4}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(negativeDim.status, ge::GRAPH_FAILED);

    auto overflow = RunTiling({std::numeric_limits<int64_t>::max(), 2}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(overflow.status, ge::GRAPH_FAILED);

    auto rankNine = RunTiling({1, 1, 1, 1, 1, 1, 1, 1, 1}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(rankNine.status, ge::GRAPH_FAILED);

    auto unknownRank = RunTiling({-2}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(unknownRank.status, ge::GRAPH_FAILED);
}

TEST_F(LpNormUpdateV2TilingTest, empty_and_dynamic_shapes)
{
    auto empty = RunTiling({0, 4}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(empty.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(empty.blockDim, 1u);

    auto resolvedDynamic = RunTiling({1}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(resolvedDynamic.status, ge::GRAPH_SUCCESS);

    auto unresolvedDynamic = RunTiling({-1}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(unresolvedDynamic.status, ge::GRAPH_FAILED);
}

TEST_F(LpNormUpdateV2TilingTest, large_dimension_split_stays_positive)
{
    auto result = RunTiling({std::numeric_limits<int64_t>::max()}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.data.coreNum, 0);
    EXPECT_GT(result.data.blockFormer, 0);
    EXPECT_GT(result.data.blockNum, 0);
    EXPECT_GT(result.data.ubFormer, 0);
}

TEST_F(LpNormUpdateV2TilingTest, rank_zero_is_scalar)
{
    auto result = RunTiling({}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.data.dim0, 1);
}

TEST_F(LpNormUpdateV2TilingTest, capacity_exceeds_uint32_elements)
{
    auto result = RunTiling({65536, 65536}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.data.dim0, 4294967296LL);
    EXPECT_GT(result.data.coreNum, 0);
    EXPECT_GT(result.data.blockFormer, 0);
    EXPECT_GT(result.data.blockNum, 0);
}

TEST_F(LpNormUpdateV2TilingTest, zero_dim_after_large_prefix_is_empty)
{
    auto result = RunTiling({4294967296LL, 4294967296LL, 0}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.data.dim0, 0);
}

TEST_F(LpNormUpdateV2TilingTest, sqrt_window_matches_baseline)
{
    auto exact = RunTiling({4}, ge::DT_FLOAT, 2.0f, 0.0f);
    EXPECT_EQ(exact.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(exact.tilingKey, 1);

    auto outside = RunTiling({4}, ge::DT_FLOAT, 2.00001f, 0.0f);
    EXPECT_EQ(outside.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(outside.tilingKey, 2);
}

TEST_F(LpNormUpdateV2TilingTest, nan_p_uses_pow_branch)
{
    auto result = RunTiling({4}, ge::DT_FLOAT, std::numeric_limits<float>::quiet_NaN(), 0.0f);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 2);
    EXPECT_TRUE(std::isnan(result.data.invP));
}
