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
 * \file test_inplace_apply_keras_momentum_tiling_arch35.cpp
 * \brief InplaceApplyKerasMomentum Tiling (arch35) host UT.
 *
 * The test calls the real registered Tiling function through TilingContextFaker;
 * it does not copy the implementation into the test file.
 */

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"

#include "../../../../op_kernel/arch35/inplace_apply_keras_momentum_tiling_data.h"

using namespace ge;
using namespace std;

namespace {

constexpr const char* kOpType = "InplaceApplyKerasMomentum";
constexpr uint64_t kUbSize = 192U * 1024U; // matches the documented 192KB fallback

struct InplaceApplyKerasMomentumCompileInfo {};

struct TilingOutcome {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint32_t totalElements = 0;
    uint32_t tileLength = 0;
    uint32_t blockFormer = 0;
    uint32_t blockNum = 0;
    uint32_t blockDim = 0;
    uint8_t useNesterov = 0;
    bool hasData = false;
};

void InitPlatForm(fe::PlatFormInfos& platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                  map<string, string>& intrinsics)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableStorageShape().AppendDim(dim);
        shape.MutableOriginShape().AppendDim(dim);
    }
    return shape;
}

TilingOutcome RunTiling(const std::vector<gert::StorageShape*>& inputs, ge::DataType varDtype, ge::DataType accumDtype,
                        ge::DataType lrDtype, ge::DataType gradDtype, ge::DataType momentumDtype, ge::Format varFormat,
                        bool useNesterov, const std::vector<gert::StorageShape*>& outputs)
{
    TilingOutcome outcome;
    InplaceApplyKerasMomentumCompileInfo compileInfo;
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        ADD_FAILURE() << "tiling function is not registered for " << kOpType;
        return outcome;
    }

    auto tilingData = gert::TilingData::CreateCap(8192);
    if (tilingData == nullptr) {
        ADD_FAILURE() << "failed to allocate TilingData";
        return outcome;
    }
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(32);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto faker = gert::TilingContextFaker();
    faker.NodeIoNum(static_cast<int>(inputs.size()), static_cast<int>(outputs.size()))
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes(inputs)
        .OutputShapes(outputs)
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, varDtype, varFormat, varFormat)
        .NodeInputTd(1, accumDtype, varFormat, varFormat)
        .NodeInputTd(2, lrDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, gradDtype, varFormat, varFormat)
        .NodeInputTd(4, momentumDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, varDtype, varFormat, varFormat)
        .NodeOutputTd(1, accumDtype, varFormat, varFormat)
        .NodeAttrs({{"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"use_nesterov", Ops::NN::AnyValue::CreateFrom<bool>(useNesterov)}})
        .TilingData(tilingData.get())
        .Workspace(workspace);
    auto holder = faker.Build();

    auto context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        ADD_FAILURE() << "failed to build TilingContext";
        return outcome;
    }
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    outcome.status = opImpl->tiling(context);
    outcome.blockDim = context->GetBlockDim();
    if (outcome.status == ge::GRAPH_SUCCESS) {
        auto tiling = context->GetTilingData<InplaceApplyKerasMomentumTilingData>();
        if (tiling == nullptr) {
            ADD_FAILURE() << "tiling data is null after a successful tiling call";
            return outcome;
        }
        outcome.hasData = true;
        outcome.totalElements = tiling->totalElements;
        outcome.tileLength = tiling->tileLength;
        outcome.blockFormer = tiling->blockFormer;
        outcome.blockNum = tiling->blockNum;
        outcome.useNesterov = tiling->useNesterov;
    }
    return outcome;
}

TilingOutcome RunUniform(const std::vector<int64_t>& dims, ge::DataType dtype, bool useNesterov = false,
                         ge::Format format = ge::FORMAT_ND)
{
    auto var = MakeShape(dims);
    auto accum = MakeShape(dims);
    auto lr = MakeShape({});
    auto grad = MakeShape(dims);
    auto momentum = MakeShape({});
    auto varOut = MakeShape(dims);
    auto accumOut = MakeShape(dims);
    return RunTiling({&var, &accum, &lr, &grad, &momentum}, dtype, dtype, ge::DT_FLOAT, dtype, ge::DT_FLOAT, format,
                     useNesterov, {&varOut, &accumOut});
}

class InplaceApplyKerasMomentumTilingTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
        if (opImpl == nullptr || opImpl->tiling == nullptr) {
            GTEST_SKIP() << "tiling function is not registered in this build mode";
        }
    }
};

} // namespace

TEST_F(InplaceApplyKerasMomentumTilingTest, fp32_normal)
{
    auto result = RunUniform({1000, 10, 10, 10}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.totalElements, 1000000U);
    EXPECT_EQ(result.tileLength, 3968U);
    EXPECT_EQ(result.blockFormer, 3584U);
    EXPECT_EQ(result.blockNum, 280U);
    EXPECT_EQ(result.blockDim, 280U);
    EXPECT_EQ(result.useNesterov, 0U);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, fp16_normal)
{
    auto result = RunUniform({1000, 10, 10, 10}, ge::DT_FLOAT16);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.tileLength, 4352U);
    EXPECT_EQ(result.blockFormer, 4096U);
    EXPECT_EQ(result.blockNum, 245U);
    EXPECT_EQ(result.blockDim, 245U);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, bf16_normal)
{
    auto result = RunUniform({1000, 10, 10, 10}, ge::DT_BF16);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.tileLength, 4352U);
    EXPECT_EQ(result.blockFormer, 4096U);
    EXPECT_EQ(result.blockNum, 245U);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, use_nesterov_true)
{
    auto result = RunUniform({64, 3, 6, 6}, ge::DT_FLOAT, true);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.useNesterov, 1U);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, scalar_lr_momentum_is_optional)
{
    // lr/momentum with shape {1} must be accepted as well as rank-0 scalars.
    auto var = MakeShape({1024});
    auto accum = MakeShape({1024});
    auto lr = MakeShape({1});
    auto grad = MakeShape({1024});
    auto momentum = MakeShape({1});
    auto varOut = MakeShape({1024});
    auto accumOut = MakeShape({1024});
    auto result = RunTiling({&var, &accum, &lr, &grad, &momentum}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                            ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND, false, {&varOut, &accumOut});
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, rank_zero_is_treated_as_one_element)
{
    auto result = RunUniform({}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.totalElements, 1U);
    EXPECT_EQ(result.blockDim, 1U);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, empty_tensor_early_return)
{
    auto result = RunUniform({0, 1}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.totalElements, 0U);
    EXPECT_EQ(result.blockDim, 1U);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, rank_nine_rejected)
{
    auto result = RunUniform({1, 1, 1, 1, 1, 1, 1, 1, 1}, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, shape_mismatch_rejected)
{
    auto var = MakeShape({2, 3});
    auto accum = MakeShape({2, 4});
    auto lr = MakeShape({});
    auto grad = MakeShape({2, 3});
    auto momentum = MakeShape({});
    auto varOut = MakeShape({2, 3});
    auto accumOut = MakeShape({2, 3});
    auto result = RunTiling({&var, &accum, &lr, &grad, &momentum}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                            ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND, false, {&varOut, &accumOut});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, dtype_mismatch_rejected)
{
    auto var = MakeShape({1024});
    auto accum = MakeShape({1024});
    auto lr = MakeShape({});
    auto grad = MakeShape({1024});
    auto momentum = MakeShape({});
    auto varOut = MakeShape({1024});
    auto accumOut = MakeShape({1024});
    auto result = RunTiling({&var, &accum, &lr, &grad, &momentum}, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                            ge::DT_FLOAT16, ge::DT_FLOAT, ge::FORMAT_ND, false, {&varOut, &accumOut});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, non_scalar_lr_rejected)
{
    auto var = MakeShape({1024});
    auto accum = MakeShape({1024});
    auto lr = MakeShape({2});
    auto grad = MakeShape({1024});
    auto momentum = MakeShape({});
    auto varOut = MakeShape({1024});
    auto accumOut = MakeShape({1024});
    auto result = RunTiling({&var, &accum, &lr, &grad, &momentum}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                            ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND, false, {&varOut, &accumOut});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, non_nd_format_rejected)
{
    auto result = RunUniform({1024}, ge::DT_FLOAT, false, ge::FORMAT_NCHW);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, unsupported_dtype_rejected)
{
    auto result = RunUniform({1024}, ge::DT_INT32);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, four_g_elements_rejected)
{
    // 65536 * 65536 = 2^32 elements, which does not fit uint32_t totalElements.
    auto result = RunUniform({65536, 65536}, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, uint32_max_elements_accepted)
{
    // 65535 * 65537 = 2^32 - 1 elements, the maximum representable element count.
    auto result = RunUniform({65535, 65537}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.totalElements, 4294967295U);
    EXPECT_LE(result.blockNum, 65535U);
    EXPECT_GE(static_cast<int64_t>(result.blockFormer) * result.blockNum, 4294967295LL);
    EXPECT_LT(static_cast<int64_t>(result.blockFormer) * (result.blockNum - 1), 4294967295LL);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, block_num_is_capped_at_65535)
{
    auto result = RunUniform({65536, 4096}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_LE(result.blockNum, 65535U);
    EXPECT_LE(result.blockDim, 65535U);
    EXPECT_GE(static_cast<int64_t>(result.blockFormer) * result.blockNum, 268435456LL);
    EXPECT_LT(static_cast<int64_t>(result.blockFormer) * (result.blockNum - 1), 268435456LL);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, four_gib_fp32_accepted)
{
    // 32768 * 32768 * 4B = 4 GiB logical storage; no extra 4GiB rejection.
    auto result = RunUniform({32768, 32768}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.totalElements, 1073741824U);
}

TEST_F(InplaceApplyKerasMomentumTilingTest, four_gib_fp16_accepted)
{
    // 65536 * 32768 * 2B = 4 GiB logical storage; no extra 4GiB rejection.
    auto result = RunUniform({65536, 32768}, ge::DT_FLOAT16);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.totalElements, 2147483648U);
}
