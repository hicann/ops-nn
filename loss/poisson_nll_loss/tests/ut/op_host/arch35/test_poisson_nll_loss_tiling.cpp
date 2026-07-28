/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_poisson_nll_loss_tiling.cpp
 * \brief PoissonNllLoss arch35 tiling UT. Covers reduction none/sum/mean x fp16/fp32,
 *        the log_input/full attr plumbing, empty-tensor acceptance, and dtype tiling-key dispatch.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../op_host/arch35/poisson_nll_loss_tiling.h"
#include "../../../../op_kernel/arch35/poisson_nll_loss_tiling_def.h"

using namespace std;
using namespace ge;
using namespace ut_util;

namespace {
constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t REDUCTION_SUM = 1;
constexpr uint32_t REDUCTION_MEAN = 2;
// Tiling key = doubleBufferKey * 256 + dtypeIdx, where dtypeIdx is fp32=0 / fp16=1 and
// doubleBufferKey is 1 when totalNum > MIN_SPLIT_THRESHOLD (double buffer), else 0.
// Confirmed by smoke (all >1024 shapes: fp32=256, fp16=257) and by the small-shape UT below.
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static int64_t ExpectedKey(ge::DataType dtype, int64_t totalNum)
{
    int64_t dtypeIdx = (dtype == ge::DT_FLOAT16) ? 1 : 0;
    int64_t bufferKey = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    return bufferKey * 256 + dtypeIdx;
}
} // namespace

class PoissonNllLossTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PoissonNllLossTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "PoissonNllLossTiling TearDown" << std::endl; }
};

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                         map<string, string>& socVersion)
{
    string hardwareInfo = R"({
      "hardware_info": {"UB_SIZE": 245760, "CORE_NUM": 64, "socVersion": "Ascend950"}
                        })";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    platFormInfo.Init();
}

// Run tiling for one case. When expectSuccess, also assert the parsed tilingData fields and key.
static void DoTilingCase(const std::vector<int64_t>& inDims, ge::DataType dtype, bool logInput, bool full, float eps,
                         const std::string& reduction, bool expectSuccess, uint32_t expectReduction)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("PoissonNllLoss");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    gert::StorageShape inShape;
    gert::StorageShape outShape;
    for (auto d : inDims) {
        inShape.MutableOriginShape().AppendDim(d);
        inShape.MutableStorageShape().AppendDim(d);
        outShape.MutableOriginShape().AppendDim(d);
        outShape.MutableStorageShape().AppendDim(d);
    }

    optiling::PoissonNllLossCompileInfo compileInfo;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&inShape, &inShape})
                      .OutputShapes({&outShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"log_input", Ops::NN::AnyValue::CreateFrom<bool>(logInput)},
                                  {"full", Ops::NN::AnyValue::CreateFrom<bool>(full)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
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

    int64_t totalNum = 1;
    for (auto d : inDims) {
        totalNum *= d;
    }
    EXPECT_EQ(tilingContext->GetTilingKey(), ExpectedKey(dtype, totalNum));

    auto* rawTiling = tilingContext->GetRawTilingData();
    ASSERT_NE(rawTiling, nullptr);
    ASSERT_GE(rawTiling->GetDataSize(), sizeof(PoissonNllLossTilingData));
    const auto* td = reinterpret_cast<const PoissonNllLossTilingData*>(rawTiling->GetData());

    EXPECT_EQ(td->reduction, expectReduction);
    EXPECT_EQ(td->logInput, logInput ? 1U : 0U);
    EXPECT_EQ(td->full, full ? 1U : 0U);
    EXPECT_EQ(td->totalNum, totalNum);
    if (expectReduction == REDUCTION_MEAN) {
        if (totalNum == 0) {
            // empty tensor: meanCof = 1/0 = inf, so the kernel's total(0)*meanCof = nan (mean of empty).
            EXPECT_TRUE(std::isinf(td->meanCof));
        } else {
            EXPECT_NEAR(td->meanCof, 1.0f / static_cast<float>(totalNum), 1e-9f);
        }
    }
}

// ---- reduction=none (large -> double buffer, key 256/257) ----
TEST_F(PoissonNllLossTiling, none_fp32)
{
    DoTilingCase({128, 128}, ge::DT_FLOAT, true, false, 1e-8f, "none", true, REDUCTION_NONE);
}

TEST_F(PoissonNllLossTiling, none_fp16)
{
    DoTilingCase({64, 64}, ge::DT_FLOAT16, true, false, 1e-8f, "none", true, REDUCTION_NONE);
}

// ---- reduction=sum ----
TEST_F(PoissonNllLossTiling, sum_fp32)
{
    DoTilingCase({256, 256}, ge::DT_FLOAT, true, false, 1e-8f, "sum", true, REDUCTION_SUM);
}

TEST_F(PoissonNllLossTiling, sum_fp16_full_logT)
{
    DoTilingCase({64, 64}, ge::DT_FLOAT16, true, true, 1e-8f, "sum", true, REDUCTION_SUM);
}

// ---- reduction=mean ----
TEST_F(PoissonNllLossTiling, mean_fp32_logF)
{
    DoTilingCase({128, 128}, ge::DT_FLOAT, false, false, 1e-8f, "mean", true, REDUCTION_MEAN);
}

TEST_F(PoissonNllLossTiling, mean_fp16)
{
    DoTilingCase({100, 100}, ge::DT_FLOAT16, true, false, 1e-8f, "mean", true, REDUCTION_MEAN);
}

// ---- large fp16 explicitly exercises the fp16 double-buffer key (257) ----
TEST_F(PoissonNllLossTiling, none_fp16_large)
{
    DoTilingCase({2048, 2048}, ge::DT_FLOAT16, true, false, 1e-8f, "none", true, REDUCTION_NONE);
}

// ---- small shape takes the single-buffer path (key drops the 256 buffer bit) ----
TEST_F(PoissonNllLossTiling, none_fp32_small_single_buffer)
{
    DoTilingCase({8, 8}, ge::DT_FLOAT, true, false, 1e-8f, "none", true, REDUCTION_NONE);
}

// ---- empty tensor is accepted (aligns with A2/torch: none->empty output, sum->0, mean->nan) ----
// The tiling must succeed for all reductions; the kernel produces the reduced scalar (sum=0,
// mean=0*inf=nan) or an empty output (none). block_dim stays 1 for empty.
TEST_F(PoissonNllLossTiling, empty_tensor_accepted_none)
{
    DoTilingCase({0}, ge::DT_FLOAT, true, false, 1e-8f, "none", true, REDUCTION_NONE);
}

TEST_F(PoissonNllLossTiling, empty_tensor_accepted_sum)
{
    DoTilingCase({0}, ge::DT_FLOAT, true, false, 1e-8f, "sum", true, REDUCTION_SUM);
}

TEST_F(PoissonNllLossTiling, empty_tensor_accepted_mean)
{
    DoTilingCase({0}, ge::DT_FLOAT16, true, false, 1e-8f, "mean", true, REDUCTION_MEAN);
}

// ---- invalid reduction string is rejected ----
TEST_F(PoissonNllLossTiling, invalid_reduction_rejected)
{
    DoTilingCase({16, 16}, ge::DT_FLOAT, true, false, 1e-8f, "bogus", false, REDUCTION_NONE);
}

// ---- eps == 0 is rejected (guards log(input+eps); aligns with A2 entry gate) ----
TEST_F(PoissonNllLossTiling, eps_zero_rejected)
{
    DoTilingCase({16, 16}, ge::DT_FLOAT, false, false, 0.0f, "mean", false, REDUCTION_MEAN);
}
