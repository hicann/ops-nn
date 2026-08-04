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
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/apply_adagrad_tiling_arch35.h"

using namespace ge;
using namespace ut_util;

class ApplyAdagradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdagradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyAdagradTiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tilingDataV)
{
    auto data = tilingDataV->GetData();
    string result;
    for (size_t i = 0; i < tilingDataV->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tilingDataV->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void InitPlatForm(fe::PlatFormInfos& platformInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    platformInfo.Init();
}

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int64_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

static void DoTest(gert::StorageShape& var, gert::StorageShape& accum, gert::StorageShape& lr, gert::StorageShape& grad,
                   gert::StorageShape& var_out, ge::DataType varDtype, ge::Format format, bool updateSlots,
                   bool useLocking, const string& expectData)
{
    optiling::ApplyAdagradCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    std::string opType("ApplyAdagrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(32);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&var, &accum, &lr, &grad})
                      .OutputShapes({&var_out})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeAttrs({{"update_slots", Ops::NN::AnyValue::CreateFrom<bool>(updateSlots)},
                                  {"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(useLocking)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    EXPECT_EQ(tilingDataResult, expectData);
}

static void DoFailedTest(gert::StorageShape& var, gert::StorageShape& accum, gert::StorageShape& lr,
                         gert::StorageShape& grad, ge::DataType varDtype, ge::DataType accumDtype, ge::DataType lrDtype,
                         ge::DataType gradDtype)
{
    optiling::ApplyAdagradCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    std::string opType("ApplyAdagrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(32);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape var_out = var;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&var, &accum, &lr, &grad})
                      .OutputShapes({&var_out})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, accumDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, lrDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"update_slots", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}

#define RUN_TEST_WITH_SHAPE(...)                                                                          \
    do {                                                                                                  \
        gert::StorageShape var = {{__VA_ARGS__}, {__VA_ARGS__}};                                          \
        gert::StorageShape accum = {{__VA_ARGS__}, {__VA_ARGS__}};                                        \
        gert::StorageShape lr = {{1}, {1}};                                                               \
        gert::StorageShape grad = {{__VA_ARGS__}, {__VA_ARGS__}};                                         \
        gert::StorageShape var_out = {{__VA_ARGS__}, {__VA_ARGS__}};                                      \
        DoTest(var, accum, lr, grad, var_out, varDtype, dataFormat, updateSlots, useLocking, expectData); \
    } while (0)

TEST_F(ApplyAdagradTilingTest, apply_adagrad_tiling_1000)
{
    auto varDtype = ge::DT_FLOAT;
    auto dataFormat = ge::FORMAT_ND;
    bool updateSlots = false;
    bool useLocking = false;
    string expectData = "3840 1024 8768 ";
    RUN_TEST_WITH_SHAPE(768, 5);
}

TEST_F(ApplyAdagradTilingTest, apply_adagrad_tiling_fp16_update_slots_true)
{
    auto varDtype = ge::DT_FLOAT16;
    auto dataFormat = ge::FORMAT_ND;
    bool updateSlots = true;
    bool useLocking = false;
    string expectData = "1024 1024 6400 ";
    RUN_TEST_WITH_SHAPE(1024);
}

TEST_F(ApplyAdagradTilingTest, apply_adagrad_tiling_bf16_multi_core)
{
    auto varDtype = ge::DT_BF16;
    auto dataFormat = ge::FORMAT_ND;
    bool updateSlots = true;
    bool useLocking = true;
    string expectData = "65536 2048 6400 ";
    RUN_TEST_WITH_SHAPE(256, 256);
}

TEST_F(ApplyAdagradTilingTest, apply_adagrad_tiling_empty_tensor)
{
    auto varDtype = ge::DT_FLOAT;
    auto dataFormat = ge::FORMAT_ND;
    bool updateSlots = false;
    bool useLocking = false;
    string expectData = "0 0 1 ";
    RUN_TEST_WITH_SHAPE(0);
}

TEST_F(ApplyAdagradTilingTest, apply_adagrad_tiling_mismatched_shape_failed)
{
    gert::StorageShape var = {{16}, {16}};
    gert::StorageShape accum = {{8}, {8}};
    gert::StorageShape lr = {{1}, {1}};
    gert::StorageShape grad = {{16}, {16}};
    DoFailedTest(var, accum, lr, grad, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
}

TEST_F(ApplyAdagradTilingTest, apply_adagrad_tiling_lr_non_scalar_failed)
{
    gert::StorageShape var = {{16}, {16}};
    gert::StorageShape accum = {{16}, {16}};
    gert::StorageShape lr = {{2}, {2}};
    gert::StorageShape grad = {{16}, {16}};
    DoFailedTest(var, accum, lr, grad, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
}

TEST_F(ApplyAdagradTilingTest, apply_adagrad_tiling_mismatched_dtype_failed)
{
    gert::StorageShape var = {{16}, {16}};
    gert::StorageShape accum = {{16}, {16}};
    gert::StorageShape lr = {{1}, {1}};
    gert::StorageShape grad = {{16}, {16}};
    DoFailedTest(var, accum, lr, grad, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT);
}
