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
 * \file test_fused_cross_entropy_loss_with_max_sum_arch3510_tiling.cpp
 * \brief FusedCrossEntropyLossWithMaxSum regbase(ascend950) tiling ut
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

#include "../../../op_kernel/arch35/fused_cross_entropy_loss_with_max_sum_tiling_data.h"

using namespace std;
using namespace ge;

class FusedCrossEntropyLossWithMaxSumArch3510Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedCrossEntropyLossWithMaxSumArch3510Tiling SetUp" << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "FusedCrossEntropyLossWithMaxSumArch3510Tiling TearDown" << std::endl;
    }
};

static void InitRegbasePlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                                map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                                map<string, string>& socVersion)
{
    string hardwareInfo = R"({
        "hardware_info": {"UB_SIZE": 253952, "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    platFormInfo.Init();
}

struct FusedCeMaxSumUtCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t sysWorkspaceSize = 0;
    int64_t ubSizePlatForm = 0;
};

static void DoRegbaseTiling(gert::KernelRunContextHolder& holder, fe::PlatFormInfos& platFormInfo,
                            map<string, string>& socInfos, map<string, string>& aicoreSpec,
                            map<string, string>& intrinsics, map<string, string>& socVersionInfos,
                            uint64_t expectTilingKey, int64_t expectBlockDim,
                            FusedCrossEntropyLossWithMaxSumRegBaseTilingData& tilingDataOut)
{
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);

    std::string op_type("FusedCrossEntropyLossWithMaxSum");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), expectTilingKey);
    EXPECT_EQ(tiling_context->GetBlockDim(), expectBlockDim);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    memcpy(&tilingDataOut, raw_tiling_data->GetData(), sizeof(FusedCrossEntropyLossWithMaxSumRegBaseTilingData));
}

TEST_F(FusedCrossEntropyLossWithMaxSumArch3510Tiling, test_regbase_tiling_full_fp32)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    InitRegbasePlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersionInfos);

    FusedCeMaxSumUtCompileInfo compileInfo;
    gert::StorageShape input0 = {{1024}, {1024}};
    gert::StorageShape input1 = {{1024}, {1024}};
    gert::StorageShape input2 = {{1024}, {1024}};
    gert::StorageShape input3 = {{1024}, {1024}};
    gert::StorageShape input4 = {{1024}, {1024}};
    gert::StorageShape input5 = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape output0 = {{1024}, {1024}};
    gert::StorageShape output1 = {{1024, 4096}, {1024, 4096}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&input0, &input1, &input2, &input3, &input4, &input5})
                      .OutputShapes({&output0, &output1})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    FusedCrossEntropyLossWithMaxSumRegBaseTilingData tilingData;
    // full路径(schId=0)：bt=1024, CORE_NUM=64 -> 每核16行；vPerLoop = FloorAlign((253952-3328)/128, 64) = 1920
    DoRegbaseTiling(holder, platFormInfo, socInfos, aicoreSpec, intrinsics, socVersionInfos, 0, 64, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 0);
    EXPECT_EQ(tilingData.formerRows, 17);
    EXPECT_EQ(tilingData.latterRows, 16);
    EXPECT_EQ(tilingData.vPerLoop, 1920);
    EXPECT_EQ(tilingData.vLen, 4096);
    EXPECT_EQ(tilingData.vCores, 1);
    EXPECT_EQ(tilingData.vChunk, 4096);
}

TEST_F(FusedCrossEntropyLossWithMaxSumArch3510Tiling, test_regbase_tiling_full_bf16_small_bt)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    InitRegbasePlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersionInfos);

    FusedCeMaxSumUtCompileInfo compileInfo;
    gert::StorageShape input0 = {{8}, {8}};
    gert::StorageShape input1 = {{8}, {8}};
    gert::StorageShape input2 = {{8}, {8}};
    gert::StorageShape input3 = {{8}, {8}};
    gert::StorageShape input4 = {{8}, {8}};
    gert::StorageShape input5 = {{8, 128}, {8, 128}};
    gert::StorageShape output0 = {{8}, {8}};
    gert::StorageShape output1 = {{8, 128}, {8, 128}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&input0, &input1, &input2, &input3, &input4, &input5})
                      .OutputShapes({&output0, &output1})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    FusedCrossEntropyLossWithMaxSumRegBaseTilingData tilingData;
    // bt=8 < CORE_NUM=64 -> 启用8核每核1行；v=128整载，vPerLoop=128
    DoRegbaseTiling(holder, platFormInfo, socInfos, aicoreSpec, intrinsics, socVersionInfos, 0, 8, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 0);
    EXPECT_EQ(tilingData.formerRows, 2);
    EXPECT_EQ(tilingData.latterRows, 1);
    EXPECT_EQ(tilingData.vPerLoop, 128);
    EXPECT_EQ(tilingData.vLen, 128);
    EXPECT_EQ(tilingData.vCores, 1);
    EXPECT_EQ(tilingData.vChunk, 128);
}

TEST_F(FusedCrossEntropyLossWithMaxSumArch3510Tiling, test_regbase_tiling_for_memory)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    InitRegbasePlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersionInfos);

    FusedCeMaxSumUtCompileInfo compileInfo;
    gert::StorageShape input0 = {{1024}, {1024}};
    gert::StorageShape input1 = {{1024}, {1024}};
    gert::StorageShape input2 = {{1024}, {1024}};
    gert::StorageShape output0 = {{1024}, {1024}};
    gert::StorageShape output1 = {{1024, 4096}, {1024, 4096}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input0, &input1, &input2})
                      .OutputShapes({&output0, &output1})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    FusedCrossEntropyLossWithMaxSumRegBaseTilingData tilingData;
    // 省显存路径(schId=1)：elementsNumber = FloorAlign((253952-2048)/24, 64) = 10496
    DoRegbaseTiling(holder, platFormInfo, socInfos, aicoreSpec, intrinsics, socVersionInfos, 1, 64, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 0);
    EXPECT_EQ(tilingData.latterRows, 16);
    EXPECT_EQ(tilingData.elementsNumber, 10496);
    EXPECT_EQ(tilingData.vCores, 1);
    EXPECT_EQ(tilingData.vChunk, 0);
}

TEST_F(FusedCrossEntropyLossWithMaxSumArch3510Tiling, test_regbase_tiling_full_fp16_v_split)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    InitRegbasePlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersionInfos);

    FusedCeMaxSumUtCompileInfo compileInfo;
    gert::StorageShape input0 = {{8}, {8}};
    gert::StorageShape input1 = {{8}, {8}};
    gert::StorageShape input2 = {{8}, {8}};
    gert::StorageShape input3 = {{8}, {8}};
    gert::StorageShape input4 = {{8}, {8}};
    gert::StorageShape input5 = {{8, 8192}, {8, 8192}};
    gert::StorageShape output0 = {{8}, {8}};
    gert::StorageShape output1 = {{8, 8192}, {8, 8192}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&input0, &input1, &input2, &input3, &input4, &input5})
                      .OutputShapes({&output0, &output1})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    FusedCrossEntropyLossWithMaxSumRegBaseTilingData tilingData;
    // v切分：bt=8 < CORE_NUM=64，fp16 vPerLoop=2560 -> vCoresNeed=ceil(8192/2560)=4, vCoresMax=64/8=8
    // -> vCores=4，rowCores=8，总核数=32，vChunk=2048
    DoRegbaseTiling(holder, platFormInfo, socInfos, aicoreSpec, intrinsics, socVersionInfos, 0, 32, tilingData);
    EXPECT_EQ(tilingData.formerCoreNum, 0);
    EXPECT_EQ(tilingData.formerRows, 2);
    EXPECT_EQ(tilingData.latterRows, 1);
    EXPECT_EQ(tilingData.vPerLoop, 2560);
    EXPECT_EQ(tilingData.vLen, 8192);
    EXPECT_EQ(tilingData.vCores, 4);
    EXPECT_EQ(tilingData.vChunk, 2048);
}
