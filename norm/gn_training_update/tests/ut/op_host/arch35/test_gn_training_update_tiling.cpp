/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_gn_training_update_tiling.cpp
 * @brief GNTrainingUpdate host tiling（arch35/Ascend950）单元测试
 *
 * 覆盖：NCHW/NHWC 正向（fp16/fp32 × 有无仿射）、空 batch 短路、tilingKey 路由、
 * 关键校验拒绝（dtype/format/rank/G∤C/num_groups/epsilon/统计量 shape 不符）。
 */

#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_kernel/arch35/gn_training_update_tiling_struct.h"
#include "../../../../op_host/arch35/gn_training_update_tiling_arch35.h"

namespace {

constexpr uint64_t CORE_NUM = 24;    // Ascend950PR AIV 核数
constexpr uint64_t UB_SIZE = 253952; // Ascend950PR 单核 UB 字节数

void InitPlatform(fe::PlatFormInfos& platformInfo, std::map<std::string, std::string>& socInfos,
                  std::map<std::string, std::string>& aicoreSpec, std::map<std::string, std::string>& intrinsics,
                  std::map<std::string, std::string>& socVersion)
{
    const std::string compileInfo = R"({
        "hardware_info": {"UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
        "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(compileInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    socInfos["ai_core_cnt"] = "24";
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

struct TilingResult {
    ge::graphStatus status;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    GnTrainingUpdateTilingData<4> data{};
};

// xNchw: [N,C,H,W]；nhwc=true 时按 [N,H,W,C] 传 xDims、统计量 [N,1,1,G,1]
// hasAffine: 是否带 scale/offset；xDtype: x/y 的 dtype
TilingResult RunTiling(const std::vector<int64_t>& xDims, const std::vector<int64_t>& statDims,
                       const std::vector<int64_t>& affineDims, int64_t numGroups, bool hasAffine,
                       ge::DataType xDtype = ge::DT_FLOAT, ge::Format xFmt = ge::FORMAT_ND, float epsilon = 1e-4F,
                       ge::DataType yDtype = ge::DT_UNDEFINED)
{
    TilingResult res{ge::GRAPH_FAILED, 0, 0, {}};
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GNTrainingUpdate");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return res;
    }

    fe::PlatFormInfos platformInfo;
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatform(platformInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    optiling::GnTrainingUpdateCompileInfo compileInfo{CORE_NUM, UB_SIZE};
    auto tilingData = gert::TilingData::CreateCap(sizeof(res.data) + 64);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    gert::StorageShape x = MakeStorageShape(xDims);
    gert::StorageShape sum = MakeStorageShape(statDims);
    gert::StorageShape squareSum = MakeStorageShape(statDims);
    gert::StorageShape scale = MakeStorageShape(affineDims);
    gert::StorageShape offset = MakeStorageShape(affineDims);
    gert::StorageShape y = MakeStorageShape(xDims);
    gert::StorageShape batchMean = MakeStorageShape(statDims);
    gert::StorageShape batchVariance = MakeStorageShape(statDims);

    // 输入槽位：x/sum/square_sum 必选，scale/offset 按 hasAffine，mean/variance 缺席
    // （IrInstanceNum 按 7 个 IR 槽位给 0/1；InputShapes 只列实际实例化的张量）
    std::vector<uint32_t> irInst = {1, 1, 1, hasAffine ? 1U : 0U, hasAffine ? 1U : 0U, 0, 0};
    std::vector<void*> inShapes;
    if (hasAffine) {
        inShapes = {&x, &sum, &squareSum, &scale, &offset};
    } else {
        inShapes = {&x, &sum, &squareSum};
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType("GNTrainingUpdate")
                      .NodeIoNum(hasAffine ? 5 : 3, 3)
                      .IrInstanceNum(irInst)
                      .InputShapes(inShapes)
                      .OutputShapes({&y, &batchMean, &batchVariance})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, xFmt, xFmt)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype == ge::DT_UNDEFINED ? xDtype : yDtype, xFmt, xFmt)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(numGroups)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto context = holder.GetContext<gert::TilingContext>();
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    res.status = opImpl->tiling(context);
    if (res.status == ge::GRAPH_SUCCESS) {
        res.tilingKey = context->GetTilingKey();
        res.blockDim = context->GetBlockDim();
        auto raw = context->GetRawTilingData();
        if (raw != nullptr && raw->GetData() != nullptr &&
            raw->GetDataSize() >= sizeof(GnTrainingUpdateTilingData<4>)) {
            res.data = *reinterpret_cast<const GnTrainingUpdateTilingData<4>*>(raw->GetData());
        }
    }
    return res;
}

class GnTrainingUpdateTilingTest : public testing::Test {};

TEST_F(GnTrainingUpdateTilingTest, nchw_fp32_affine_success)
{
    auto r = RunTiling({2, 4, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 0U); // RANK_4 分支
    EXPECT_GE(r.blockDim, 1U);
    EXPECT_EQ(r.data.hasAffine, 1);
    EXPECT_EQ(r.data.hasOffset, 1);
    EXPECT_FLOAT_EQ(r.data.invM, 1.0f / 12.0f); // M=(4/2)*2*3=12
    EXPECT_GE(r.data.multicore.totalTiles, 1);  // 非空输入至少 1 tile
    EXPECT_LE(r.data.multicore.totalTiles, 4);  // 不超过 N*G*aO 上界
    EXPECT_GE(r.data.multicore.numCores, 1);
    EXPECT_GT(r.data.perBufBytes, 0);
    EXPECT_EQ(r.data.perBufBytes % 32, 0); // 32B 对齐
}

// NCHW + 无仿射 + fp16
TEST_F(GnTrainingUpdateTilingTest, nchw_fp16_no_affine_success)
{
    auto r = RunTiling({2, 4, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, false, ge::DT_FLOAT16);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 0U);
    EXPECT_EQ(r.data.hasAffine, 0);
    EXPECT_EQ(r.data.hasOffset, 0);
}

// NHWC：x[2,2,3,4]（C=4 在末维），统计量 [2,1,1,2,1]，仿射 [1,1,1,2,1]
TEST_F(GnTrainingUpdateTilingTest, nhwc_fp32_affine_success)
{
    auto r = RunTiling({2, 2, 3, 4}, {2, 1, 1, 2, 1}, {1, 1, 1, 2, 1}, 2, true, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.data.hasAffine, 1);
    EXPECT_GE(r.data.multicore.totalTiles, 1);
    EXPECT_LE(r.data.multicore.totalTiles, 4);
    EXPECT_FLOAT_EQ(r.data.invM, 1.0f / 12.0f); // M=(4/2)*2*3=12
}

// 显式 NCHW format 标签：x/y origin format = NCHW
TEST_F(GnTrainingUpdateTilingTest, nchw_explicit_format_tag_success)
{
    auto r = RunTiling({2, 4, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true, ge::DT_FLOAT, ge::FORMAT_NCHW);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.tilingKey, 0U);
}

// 显式 NHWC 标签 + NHWC 形态统计量
TEST_F(GnTrainingUpdateTilingTest, nhwc_explicit_format_tag_success)
{
    auto r = RunTiling({2, 2, 3, 4}, {2, 1, 1, 2, 1}, {1, 1, 1, 2, 1}, 2, true, ge::DT_FLOAT, ge::FORMAT_NHWC);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
}

// 标签冲突：x 标 NCHW 但统计量组维在倒数第 2 位（NHWC 形态）→ 拒绝
TEST_F(GnTrainingUpdateTilingTest, mixed_layout_tag_rejected)
{
    auto r = RunTiling({2, 2, 3, 4}, {2, 1, 1, 2, 1}, {1, 1, 1, 2, 1}, 2, true, ge::DT_FLOAT, ge::FORMAT_NCHW);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// 空 batch：N=0 短路，totalTiles=0
TEST_F(GnTrainingUpdateTilingTest, empty_batch_success)
{
    auto r = RunTiling({0, 4, 2, 3}, {0, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.data.multicore.totalTiles, 0);
    EXPECT_EQ(r.blockDim, 1U);
}

// C 不整除 G：C=3, G=2 → 拒绝
TEST_F(GnTrainingUpdateTilingTest, group_not_divisible_rejected)
{
    auto r = RunTiling({2, 3, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// num_groups 越界（<1）→ 拒绝
TEST_F(GnTrainingUpdateTilingTest, num_groups_zero_rejected)
{
    auto r = RunTiling({2, 4, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 0, true, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// x dtype 非法（int32）→ 拒绝
TEST_F(GnTrainingUpdateTilingTest, bad_dtype_rejected)
{
    auto r = RunTiling({2, 4, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true, ge::DT_INT32);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// x 非法 format（FRACTAL_NZ）→ 拒绝
TEST_F(GnTrainingUpdateTilingTest, bad_format_rejected)
{
    auto r = RunTiling({2, 4, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// 统计量 shape 与推导不符（G 维两处都不是 G）→ 拒绝
TEST_F(GnTrainingUpdateTilingTest, bad_stat_shape_rejected)
{
    auto r = RunTiling({2, 4, 2, 3}, {2, 3, 1, 3, 1}, {1, 2, 1, 1, 1}, 2, true, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

} // namespace

// x rank≠4 拒收（CheckMaxDimensions 第一分支）
TEST_F(GnTrainingUpdateTilingTest, x_rank_not4_rejected)
{
    TilingResult r = RunTiling({2, 4, 2}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// 统计量 rank≠5 拒收（CheckMaxDimensions 输入循环分支）
TEST_F(GnTrainingUpdateTilingTest, stats_rank_not5_rejected)
{
    TilingResult r = RunTiling({2, 4, 2, 3}, {2, 2, 1, 1}, {1, 2, 1, 1, 1}, 2, true);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// epsilon=0 拒收（CheckAttrValueRange epsilon 分支；下界开区间）
TEST_F(GnTrainingUpdateTilingTest, epsilon_zero_rejected)
{
    TilingResult r = RunTiling({2, 4, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true, ge::DT_FLOAT, ge::FORMAT_ND,
                               0.0F);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// H=0 拒收（CheckDimValues：C/H/W 必须 > 0；仅 N 轴可为 0）
TEST_F(GnTrainingUpdateTilingTest, hw_zero_rejected)
{
    TilingResult r = RunTiling({2, 4, 0, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// y dtype 与 x 不一致拒收（CheckDtypeSupportAndCombination 输出分支）
TEST_F(GnTrainingUpdateTilingTest, y_dtype_mismatch_rejected)
{
    TilingResult r = RunTiling({2, 4, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, true, ge::DT_FLOAT, ge::FORMAT_ND,
                               1e-4F, ge::DT_FLOAT16);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}
