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
 * \file test_inplace_apply_momentum_tiling.cpp
 * \brief InplaceApplyMomentum Tiling UT (arch35)
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class TestInplaceApplyMomentumTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TestInplaceApplyMomentumTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TestInplaceApplyMomentumTiling TearDown" << std::endl; }
};

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                         map<string, string>& socVersion)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false,
                           "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true,
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    platFormInfo.Init();
}

struct InplaceApplyMomentumUtCompileInfo {};

static void DoInplaceApplyMomentumTilingCase(std::initializer_list<int64_t>& inputShape, ge::DataType inputDtype,
                                             ge::Format inputFormat, bool use_nesterov)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("InplaceApplyMomentum");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tiling_func, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    gert::StorageShape tensorShape = {inputShape, inputShape};
    gert::StorageShape oneShape = {{1}, {1}};

    InplaceApplyMomentumUtCompileInfo compileInfo;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&tensorShape, &tensorShape, &oneShape, &tensorShape, &oneShape})
                      .OutputShapes({&tensorShape, &tensorShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(1, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(2, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(3, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(4, inputDtype, inputFormat, inputFormat)
                      .NodeOutputTd(0, inputDtype, inputFormat, inputFormat)
                      .NodeOutputTd(1, inputDtype, inputFormat, inputFormat)
                      .NodeAttrs({{"use_nesterov", Ops::NN::AnyValue::CreateFrom<bool>(use_nesterov)},
                                  {"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto blockDim = tiling_context->GetBlockDim();
    EXPECT_GT(blockDim, 0u);
}

TEST_F(TestInplaceApplyMomentumTiling, tiling_float32_standard)
{
    std::initializer_list<int64_t> inputShape = {16, 26, 16, 19};
    DoInplaceApplyMomentumTilingCase(inputShape, ge::DT_FLOAT, ge::FORMAT_ND, false);
}

TEST_F(TestInplaceApplyMomentumTiling, tiling_float32_nesterov)
{
    std::initializer_list<int64_t> inputShape = {16, 26, 16, 19};
    DoInplaceApplyMomentumTilingCase(inputShape, ge::DT_FLOAT, ge::FORMAT_ND, true);
}

TEST_F(TestInplaceApplyMomentumTiling, tiling_float16_standard)
{
    std::initializer_list<int64_t> inputShape = {3761, 4, 44, 4};
    DoInplaceApplyMomentumTilingCase(inputShape, ge::DT_FLOAT16, ge::FORMAT_ND, false);
}

TEST_F(TestInplaceApplyMomentumTiling, tiling_float16_nesterov)
{
    std::initializer_list<int64_t> inputShape = {3761, 4, 44, 4};
    DoInplaceApplyMomentumTilingCase(inputShape, ge::DT_FLOAT16, ge::FORMAT_ND, true);
}

TEST_F(TestInplaceApplyMomentumTiling, tiling_bfloat16_standard)
{
    std::initializer_list<int64_t> inputShape = {7, 2, 7, 8, 10};
    DoInplaceApplyMomentumTilingCase(inputShape, ge::DT_BF16, ge::FORMAT_ND, false);
}

TEST_F(TestInplaceApplyMomentumTiling, tiling_bfloat16_nesterov)
{
    std::initializer_list<int64_t> inputShape = {7, 2, 7, 8, 10};
    DoInplaceApplyMomentumTilingCase(inputShape, ge::DT_BF16, ge::FORMAT_ND, true);
}

TEST_F(TestInplaceApplyMomentumTiling, tiling_1d_scalar_like)
{
    std::initializer_list<int64_t> inputShape = {1};
    DoInplaceApplyMomentumTilingCase(inputShape, ge::DT_FLOAT, ge::FORMAT_ND, false);
}

TEST_F(TestInplaceApplyMomentumTiling, tiling_large_shape)
{
    std::initializer_list<int64_t> inputShape = {1024, 1024};
    DoInplaceApplyMomentumTilingCase(inputShape, ge::DT_FLOAT16, ge::FORMAT_ND, true);
}
