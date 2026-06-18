/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;

struct ThresholdGradV2DCompileInfo { uint64_t coreNum; uint64_t ubSize; };

class ThresholdGradV2DTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ThresholdGradV2DTilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ThresholdGradV2DTilingTest TearDown" << std::endl;
    }
};

static void InitPlatform(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics, map<string, string>& socVersion)
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

static void DoThresholdGradV2DTilingCase(std::initializer_list<int64_t>& inputShape1,
    std::initializer_list<int64_t>& inputShape2, std::initializer_list<int64_t>& outputShape,
    ge::DataType inputDtype)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion;
    InitPlatform(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    ThresholdGradV2DCompileInfo compileInfo;
    std::string opType("ThresholdGradV2D");
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
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr) << "tiling_parse not registered for " << opType;
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape x1Shape = {inputShape1, inputShape1};
    gert::StorageShape x2Shape = {inputShape2, inputShape2};
    gert::StorageShape oShape = {outputShape, outputShape};
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr) << "tiling not registered for " << opType;

    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x1Shape, &x2Shape})
                      .OutputShapes({&oShape})
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
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdGradV2DTilingTest, tiling_fp32_2d) {
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {2, 128};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoThresholdGradV2DTilingCase(inputShape1, inputShape2, outputShape, ge::DT_FLOAT);
}

TEST_F(ThresholdGradV2DTilingTest, tiling_fp16_2d) {
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {2, 128};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoThresholdGradV2DTilingCase(inputShape1, inputShape2, outputShape, ge::DT_FLOAT16);
}

TEST_F(ThresholdGradV2DTilingTest, tiling_bf16_2d) {
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {2, 128};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoThresholdGradV2DTilingCase(inputShape1, inputShape2, outputShape, ge::DT_BF16);
}

TEST_F(ThresholdGradV2DTilingTest, tiling_int32_2d) {
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {2, 128};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoThresholdGradV2DTilingCase(inputShape1, inputShape2, outputShape, ge::DT_INT32);
}

TEST_F(ThresholdGradV2DTilingTest, tiling_int8_2d) {
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {2, 128};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoThresholdGradV2DTilingCase(inputShape1, inputShape2, outputShape, ge::DT_INT8);
}

TEST_F(ThresholdGradV2DTilingTest, tiling_uint8_2d) {
    std::initializer_list<int64_t> inputShape1 = {2, 128};
    std::initializer_list<int64_t> inputShape2 = {2, 128};
    std::initializer_list<int64_t> outputShape = {2, 128};
    DoThresholdGradV2DTilingCase(inputShape1, inputShape2, outputShape, ge::DT_UINT8);
}

TEST_F(ThresholdGradV2DTilingTest, tiling_fp32_4d) {
    std::initializer_list<int64_t> inputShape1 = {2, 4, 8, 16};
    std::initializer_list<int64_t> inputShape2 = {2, 4, 8, 16};
    std::initializer_list<int64_t> outputShape = {2, 4, 8, 16};
    DoThresholdGradV2DTilingCase(inputShape1, inputShape2, outputShape, ge::DT_FLOAT);
}

TEST_F(ThresholdGradV2DTilingTest, tiling_fp16_1d) {
    std::initializer_list<int64_t> inputShape1 = {1024};
    std::initializer_list<int64_t> inputShape2 = {1024};
    std::initializer_list<int64_t> outputShape = {1024};
    DoThresholdGradV2DTilingCase(inputShape1, inputShape2, outputShape, ge::DT_FLOAT16);
}

TEST_F(ThresholdGradV2DTilingTest, tiling_failed_unsupported_dtype) {
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion;
    InitPlatform(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string op_type("ThresholdGradV2D");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    ASSERT_NE(tiling_func, nullptr) << "tiling not registered for " << op_type;

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape Shape1 = {{2, 128}, {2, 128}};

    ThresholdGradV2DCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape1, &Shape1})
                      .OutputShapes({&Shape1})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
