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
 * \file test_bn3_d_training_update_tiling.cpp
 * \brief Tiling UT for BN3DTrainingUpdate operator.
 *
 * Schema (aligned to op_host/bn3_d_training_update_def.cpp):
 *   Inputs  (7): x, sum, square_sum, scale, offset, mean, variance
 *     - x: FLOAT16/FLOAT/BF16, rank 4 (NCHW) or rank 5 (NCDHW)
 *     - sum / square_sum / scale / offset / mean / variance: FLOAT32, shape (C,)
 *   Outputs (5): y, mean, variance, batch_mean, batch_variance
 *     - y follows x; stats outputs follow sum
 *   Attrs   (2, required): factor, epsilon
 *
 * TilingKey contract (DESIGN §5.1, rank fork):
 *   key 0 = RANK_4 (BN3_D_TRAINING_UPDATE_RANK_4)
 *   key 1 = RANK_5 (BN3_D_TRAINING_UPDATE_RANK_5)
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/bn3_d_training_update_tiling_struct.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
const std::string OP_TYPE = "BN3DTrainingUpdate";
constexpr int kInputNum = 7;
constexpr int kOutputNum = 5;

// Ascend950 platform compile info (UB=245760, AIV core num 64).
const std::string COMPILE_INFO_950 = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
} // namespace

class BN3DTrainingUpdateTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BN3DTrainingUpdateTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "BN3DTrainingUpdateTilingTest TearDown" << std::endl; }
};

// 通用 RunTiling：经 OpImplRegistry 取注册的 tiling 回调，用 faker 构造上下文执行。
static ge::graphStatus RunTiling(const std::string& compileInfoJson, const vector<gert::StorageShape*>& inputShapes,
                                 const vector<gert::StorageShape*>& outputShapes, ge::DataType xDtype,
                                 ge::Format xFormat, const vector<pair<string, Ops::NN::AnyValue>>& attrs,
                                 uint32_t* outputTilingKey = nullptr, uint32_t* outputBlockDim = nullptr)
{
    map<string, string> socInfos, aicoreSpec, intrinsics;
    GetPlatFormInfos(compileInfoJson.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // The faker packs runtime inputs (compileInfo/platformInfo/tilingData/workspace)
    // into positional slots; CompileInfo must be set (even if unused by the tiling
    // func) so the platformInfo lands at the slot GetPlatformInfo() reads.
    int32_t dummyCompileInfo = 0;

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr)
        return ge::GRAPH_FAILED;
    auto tilingFunc = opImpl->tiling;

    auto tilingData = gert::TilingData::CreateCap(4096);
    EXPECT_NE(tilingData, nullptr);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    EXPECT_NE(wsSize, nullptr);

    gert::TilingContextFaker faker;
    faker.SetOpType(OP_TYPE)
        .NodeIoNum(kInputNum, kOutputNum)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1})
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&dummyCompileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, xDtype, xFormat, xFormat)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, xDtype, xFormat, xFormat)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs(attrs)
        .TilingData(tilingData.get())
        .Workspace(wsSize);

    auto holder = faker.Build();
    auto* tilingCtx = holder.GetContext<gert::TilingContext>();
    fe::PlatFormInfos* pfi = tilingCtx->GetPlatformInfo();
    EXPECT_NE(pfi, nullptr);
    if (pfi == nullptr) {
        return ge::GRAPH_FAILED;
    }
    pfi->SetPlatformRes("SoCInfo", socInfos);
    pfi->SetPlatformRes("AICoreSpec", aicoreSpec);
    pfi->SetCoreNumByCoreType("AICore");
    pfi->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ge::graphStatus status = tilingFunc(tilingCtx);
    if (outputTilingKey != nullptr)
        *outputTilingKey = tilingCtx->GetTilingKey();
    if (outputBlockDim != nullptr)
        *outputBlockDim = tilingCtx->GetBlockDim();
    return status;
}

// ========================================================================
// RANK=4: x {2,3,4,5} NCHW, C=3, FP16
// ========================================================================
TEST_F(BN3DTrainingUpdateTilingTest, rank4_nchw_fp16)
{
    gert::StorageShape x = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape stat = {{3}, {3}};
    gert::StorageShape y = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    uint32_t key = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_950, {&x, &stat, &stat, &stat, &stat, &stat, &stat},
                        {&y, &stat, &stat, &stat, &stat}, DT_FLOAT16, ge::FORMAT_NCHW,
                        {{"factor", Ops::NN::AnyValue::CreateFrom<float>(0.1)},
                         {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1.0e-5f)}},
                        &key, &blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0u);      // RANK_4
    EXPECT_GE(blockDim, 1u); // at least one AIV core
}

// ========================================================================
// RANK=5: x {2,3,4,5,6} NCDHW, C=3, BF16
// ========================================================================
TEST_F(BN3DTrainingUpdateTilingTest, rank5_ncdhw_bf16)
{
    gert::StorageShape x = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape stat = {{3}, {3}};
    gert::StorageShape y = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    uint32_t key = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_950, {&x, &stat, &stat, &stat, &stat, &stat, &stat},
                        {&y, &stat, &stat, &stat, &stat}, DT_BF16, ge::FORMAT_NCDHW,
                        {{"factor", Ops::NN::AnyValue::CreateFrom<float>(0.1)},
                         {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1.0e-5f)}},
                        &key, &blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u); // RANK_5
    EXPECT_GE(blockDim, 1u);
}

// ========================================================================
// RANK=4 FLOAT32 + attr defaults (attrs not passed -> 0.1 / 1e-5)
// ========================================================================
TEST_F(BN3DTrainingUpdateTilingTest, rank4_default_attrs)
{
    gert::StorageShape x = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    gert::StorageShape stat = {{3}, {3}};
    gert::StorageShape y = {{1, 3, 8, 8}, {1, 3, 8, 8}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_950, {&x, &stat, &stat, &stat, &stat, &stat, &stat},
                        {&y, &stat, &stat, &stat, &stat}, DT_FLOAT, ge::FORMAT_NCHW,
                        {{"factor", Ops::NN::AnyValue::CreateFrom<float>(0.1)},
                         {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1.0e-5f)}},
                        &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0u);
}

// ========================================================================
// RANK=5 FLOAT32: batch size 1, num = x.size / C = 40
// ========================================================================
TEST_F(BN3DTrainingUpdateTilingTest, rank5_float32)
{
    gert::StorageShape x = {{1, 4, 2, 2, 5}, {1, 4, 2, 2, 5}};
    gert::StorageShape stat = {{4}, {4}};
    gert::StorageShape y = {{1, 4, 2, 2, 5}, {1, 4, 2, 2, 5}};
    uint32_t key = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_950, {&x, &stat, &stat, &stat, &stat, &stat, &stat},
                        {&y, &stat, &stat, &stat, &stat}, DT_FLOAT, ge::FORMAT_NCDHW,
                        {{"factor", Ops::NN::AnyValue::CreateFrom<float>(0.2)},
                         {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1.0e-6f)}},
                        &key, &blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u);
    EXPECT_GE(blockDim, 1u);
}

// ========================================================================
// Negative / DFX: illegal inputs must be cleanly rejected (GRAPH_FAILED,
// structured OP_LOGE_FOR_INVALID_* reporting) — never crash. Each case aims
// at one tiling gate: dtype (:175) / format (:186) / rank (:297) /
// channel-dim broadcast compatibility (:203).
// ========================================================================
static const std::vector<pair<string, Ops::NN::AnyValue>> kAttrs = {
    {"factor", Ops::NN::AnyValue::CreateFrom<float>(0.1)}, {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1.0e-5f)}};

TEST_F(BN3DTrainingUpdateTilingTest, negative_dtype_int32_rejected)
{
    gert::StorageShape x = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape stat = {{3}, {3}};
    gert::StorageShape y = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    EXPECT_EQ(RunTiling(COMPILE_INFO_950, {&x, &stat, &stat, &stat, &stat, &stat, &stat},
                        {&y, &stat, &stat, &stat, &stat}, ge::DT_INT32, ge::FORMAT_NCHW, kAttrs),
              ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_format_fractal_nz_rejected)
{
    gert::StorageShape x = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape stat = {{3}, {3}};
    gert::StorageShape y = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    EXPECT_EQ(RunTiling(COMPILE_INFO_950, {&x, &stat, &stat, &stat, &stat, &stat, &stat},
                        {&y, &stat, &stat, &stat, &stat}, DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, kAttrs),
              ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_rank3_rejected)
{
    gert::StorageShape x = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape stat = {{3}, {3}};
    gert::StorageShape y = {{2, 3, 4}, {2, 3, 4}};
    EXPECT_EQ(RunTiling(COMPILE_INFO_950, {&x, &stat, &stat, &stat, &stat, &stat, &stat},
                        {&y, &stat, &stat, &stat, &stat}, DT_FLOAT, ge::FORMAT_NCHW, kAttrs),
              ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_rank6_rejected)
{
    gert::StorageShape x = {{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}};
    gert::StorageShape stat = {{3}, {3}};
    gert::StorageShape y = {{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}};
    EXPECT_EQ(RunTiling(COMPILE_INFO_950, {&x, &stat, &stat, &stat, &stat, &stat, &stat},
                        {&y, &stat, &stat, &stat, &stat}, DT_FLOAT, ge::FORMAT_NCDHW, kAttrs),
              ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_channel_mismatch_rejected)
{
    // x says C=3 (NCHW axis 1) but the stats tensors carry C=7.
    gert::StorageShape x = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape stat = {{7}, {7}};
    gert::StorageShape y = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    EXPECT_EQ(RunTiling(COMPILE_INFO_950, {&x, &stat, &stat, &stat, &stat, &stat, &stat},
                        {&y, &stat, &stat, &stat, &stat}, DT_FLOAT, ge::FORMAT_NCHW, kAttrs),
              ge::GRAPH_FAILED);
}
