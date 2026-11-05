/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_base_struct.h"
#include "atvoss/broadcast/broadcast_tiling_base.h"
#include "activation/relu_grad/op_kernel/arch35/relu_grad_struct.h"
#include "../../../../op_host/arch35/relu_grad_tiling_arch35.h"

using namespace std;
using ReluGradOp::g_tilingDeclareParams;
using namespace ge;
using namespace ut_util;

class ReluGradDavidTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReluGradDavidTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReluGradDavidTiling TearDown" << std::endl; }
};

static constexpr Ops::Base::BroadcastOneDimTilingDataAdvance EXPECTED_ONE_DIM_TILING_DATA = {256, 512, 1, 2};

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                         map<string, string>& socVersion)
{
    string hardwareInfo = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    platFormInfo.Init();
}

static void DoReluGradTilingCase(std::initializer_list<int64_t>& inputShape1,
                                 std::initializer_list<int64_t>& inputShape2,
                                 std::initializer_list<int64_t>& outputShape, ge::DataType inputDtype,
                                 uint64_t expectedDtypeTilingKey)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion;
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    Ops::Base::BroadcastCompileInfo compileInfo;
    std::string opType("ReluGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    string compileInfoStr = R"({})";

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape x1Shape = {inputShape1, inputShape1};
    gert::StorageShape x2Shape = {inputShape2, inputShape2};
    gert::StorageShape oSahpe = {outputShape, outputShape};
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x1Shape, &x2Shape})
                      .OutputShapes({&oSahpe})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tilingFunc(tiling_context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(tiling_context->GetTilingKey(),
              GET_TPL_TILING_KEY(Ops::Base::SCH_MODE_ONE_DIM_ADVANCE, expectedDtypeTilingKey));
    const auto* rawTilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    ASSERT_EQ(rawTilingData->GetDataSize(), sizeof(EXPECTED_ONE_DIM_TILING_DATA));
    ASSERT_NE(rawTilingData->GetData(), nullptr);
    Ops::Base::BroadcastOneDimTilingDataAdvance actualTilingData{};
    std::memcpy(&actualTilingData, rawTilingData->GetData(), sizeof(actualTilingData));
    EXPECT_EQ(actualTilingData.dimLen, EXPECTED_ONE_DIM_TILING_DATA.dimLen);
    EXPECT_EQ(actualTilingData.tileNum, EXPECTED_ONE_DIM_TILING_DATA.tileNum);
    EXPECT_EQ(actualTilingData.blockNum, EXPECTED_ONE_DIM_TILING_DATA.blockNum);
    EXPECT_EQ(actualTilingData.scalarFlag, EXPECTED_ONE_DIM_TILING_DATA.scalarFlag);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling_failed_0)
{
    std::string op_type("ReluGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape Shape1 = {{2, 128}, {2, 128}};
    gert::StorageShape Shape2 = {{1}, {1}};

    Ops::Base::BroadcastCompileInfo compile_info;

    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape1, &Shape2})
                      .OutputShapes({&Shape1})
                      .CompileInfo(&compile_info)
                      .NodeInputTd(0, ge::DT_UINT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UINT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling_failed_1)
{
    std::string op_type("ReluGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape Shape1 = {{2, 128}, {2, 128}};
    gert::StorageShape Shape2 = {{1}, {1}};

    Ops::Base::BroadcastCompileInfo compile_info;

    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape1, &Shape2})
                      .OutputShapes({&Shape1})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(nullptr)
                      .NodeInputTd(0, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling_failed_2)
{
    std::string op_type("ReluGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape Shape1 = {{2, 128}, {2, 128}};
    gert::StorageShape Shape2 = {{1}, {1}};

    Ops::Base::BroadcastCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape1, &Shape2})
                      .OutputShapes({&Shape1})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(nullptr)
                      .NodeInputTd(0, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling1)
{
    // uint8 case
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {1};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoReluGradTilingCase(inputShape1, inputShape2, outputShape, ge::DT_UINT8 /*inputdtype*/, RELU_GRAD_TPL_UINT8);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling3)
{
    // int8 case
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {1};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoReluGradTilingCase(inputShape1, inputShape2, outputShape, ge::DT_INT8 /*inputdtype*/, RELU_GRAD_TPL_INT8);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling4)
{
    // bf16 case
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {1};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoReluGradTilingCase(inputShape1, inputShape2, outputShape, ge::DT_BF16 /*inputdtype*/, RELU_GRAD_TPL_BF16);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling5)
{
    // fp16 case
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {1};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoReluGradTilingCase(inputShape1, inputShape2, outputShape, ge::DT_FLOAT16 /*inputdtype*/, RELU_GRAD_TPL_FP16);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling6)
{
    // fp32 case
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {1};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoReluGradTilingCase(inputShape1, inputShape2, outputShape, ge::DT_FLOAT /*inputdtype*/, RELU_GRAD_TPL_FP32);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling9)
{
    // int32 case
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {1};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoReluGradTilingCase(inputShape1, inputShape2, outputShape, ge::DT_INT32 /*inputdtype*/, RELU_GRAD_TPL_INT32);
}

TEST_F(ReluGradDavidTiling, relu_grad_david_tiling_int64)
{
    // int64 case
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {1};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoReluGradTilingCase(inputShape1, inputShape2, outputShape, ge::DT_INT64 /*inputdtype*/, RELU_GRAD_TPL_INT64);
}
