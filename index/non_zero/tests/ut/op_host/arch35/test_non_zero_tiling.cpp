/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "ut_op_util.h"
#include "index/non_zero/op_host/arch35/non_zero_tiling_arch35.h"
#include "test_cube_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace ut_util;

class NonZeroTilingAscendcTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NonZeroTilingAscendcTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NonZeroTilingAscendcTest TearDown" << std::endl;
    }
};

static void InitPlatForm(
    fe::PlatFormInfos& platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
    map<string, string>& intrinsics)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    platformInfo.Init();
}

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_001)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{1, 128, 2}, {1, 128, 2}};
    gert::StorageShape yShape = {{3, 159}, {3, 159}};
    gert::StorageShape out_shape = {{2}, {2}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &out_shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    //    EXPECT_EQ(tilingDataResult, "3 64 4 4 60864 0 4 0 4 1 3 0 24 7608 0 10112 4 0 4 1264 192 32 256 2 1 1 1 1 1 8
    //    1 0 0 0 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_002)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{128}, {128}};
    gert::StorageShape yShape = {{3}, {3}};
    gert::StorageShape out_shape = {{1}, {1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &out_shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult,
    //   "1 64 2 2 60864 0 2 0 2 1 1 8 0 7608 0 30400 2 0 2 3800 64 32 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_003)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{128, 2}, {128, 2}};
    gert::StorageShape yShape = {{159, 2}, {159, 2}};
    gert::StorageShape out_shape = {{2}, {2}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &out_shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    //  EXPECT_EQ(tilingDataResult, "2 64 4 4 60864 0 4 0 4 0 9 0 16 7608 0 15168 4 0 4 1896 128 32 2 1 1 1 1 1 1 1 0 0
    //  0 0 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_005)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{1024, 1024, 1024, 1024}, {1024, 1024, 1024, 1024}};
    gert::StorageShape yShape = {{4, 38849}, {4, 38849}};
    gert::StorageShape out_shape = {{2}, {2}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &out_shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    // EXPECT_EQ(tilingDataResult,
    //   "4 64 17179869184 0 3136 5478274 0 1920 0 0 20000 0 0 0 0 0 0 0 0 0 0 8589934592 1073741824 1048576 1024 1 1 1
    //   1 30 20 10 0 0 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_006)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{1, 128, 2}, {1, 128, 2}};
    gert::StorageShape yShape = {{3, 159}, {3, 159}};
    gert::StorageShape out_shape = {{2}, {2}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &out_shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    //   EXPECT_EQ(tilingDataResult, "3 64 4 4 60864 0 4 0 4 0 10 0 24 7608 0 10112 4 0 4 1264 192 32 256 2 1 1 1 1 1 8
    //   1 0 0 0 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_007)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{2, 97, 19, 0, 1}, {2, 97, 19, 0, 1}};
    gert::StorageShape yShape = {{1716, 5}, {1716, 5}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    EXPECT_EQ(
        tilingDataResult, "5 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_full_load_001)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{97}, {97}};
    gert::StorageShape yShape = {{97, 1}, {97, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingKey = tilingContext->GetTilingKey();
    auto blockDim = tilingContext->GetBlockDim();
    auto rawTilingData = tilingContext->GetRawTilingData();
    // auto tilingDataResult = to_string<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize());
    EXPECT_EQ(tilingKey, 40001);
    EXPECT_EQ(blockDim, 1);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_full_load_002)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{97, 2}, {97, 2}};
    gert::StorageShape yShape = {{2, 194}, {2, 194}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingKey = tilingContext->GetTilingKey();
    auto blockDim = tilingContext->GetBlockDim();
    auto rawTilingData = tilingContext->GetRawTilingData();
    // auto tilingDataResult = to_string<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize());
    EXPECT_EQ(tilingKey, 40002);
    EXPECT_EQ(blockDim, 1);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_full_load_003)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{97, 2, 2}, {97, 2, 2}};
    gert::StorageShape yShape = {{3, 388}, {3, 388}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingKey = tilingContext->GetTilingKey();
    auto blockDim = tilingContext->GetBlockDim();
    auto rawTilingData = tilingContext->GetRawTilingData();
    // auto tilingDataResult = to_string<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize());
    EXPECT_EQ(tilingKey, 40003);
    EXPECT_EQ(blockDim, 1);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_full_load_004)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{2, 2, 2, 4}, {2, 2, 2, 4}};
    gert::StorageShape yShape = {{32, 4}, {32, 4}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)));

    // compile info
    optiling::NonZeroCompileInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tilingKey = tilingContext->GetTilingKey();
    auto blockDim = tilingContext->GetBlockDim();
    auto rawTilingData = tilingContext->GetRawTilingData();
    // auto tilingDataResult = to_string<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize());
    EXPECT_EQ(tilingKey, 40004);
    EXPECT_EQ(blockDim, 1);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_small_mask_dim1_no_trans)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{100}, {100}};
    gert::StorageShape yShape = {{100, 1}, {100, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 40001);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_small_mask_dim1_trans)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{100}, {100}};
    gert::StorageShape yShape = {{1, 100}, {1, 100}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 40001);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_small_mask_dim2_trans)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{10, 10}, {10, 10}};
    gert::StorageShape yShape = {{2, 100}, {2, 100}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 40002);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_small_mask_dim3_no_trans)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{5, 5, 5}, {5, 5, 5}};
    gert::StorageShape yShape = {{125, 3}, {125, 3}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 40003);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_small_mask_dim4_trans)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{4, 4, 4, 4}, {4, 4, 4, 4}};
    gert::StorageShape yShape = {{4, 256}, {4, 256}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 40004);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_small_mask_dim5)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{3, 3, 3, 3, 3}, {3, 3, 3, 3, 3}};
    gert::StorageShape yShape = {{243, 5}, {243, 5}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 5);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_small_mask_dim6_trans)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{3, 3, 3, 3, 3, 3}, {3, 3, 3, 3, 3, 3}};
    gert::StorageShape yShape = {{6, 729}, {6, 729}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 13);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_small_mask_dim7)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2}};
    gert::StorageShape yShape = {{128, 7}, {128, 7}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 7);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_small_mask_dim8_trans)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};
    gert::StorageShape yShape = {{8, 256}, {8, 256}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 15);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_big_mask_dim1)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{10000}, {10000}};
    gert::StorageShape yShape = {{10000, 1}, {10000, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 40001);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_big_mask_dim2)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{200, 200}, {200, 200}};
    gert::StorageShape yShape = {{40000, 2}, {40000, 2}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 2);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_big_mask_dim3)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{50, 50, 50}, {50, 50, 50}};
    gert::StorageShape yShape = {{125000, 3}, {125000, 3}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 3);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_big_mask_dim5)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{20, 20, 20, 20, 20}, {20, 20, 20, 20, 20}};
    gert::StorageShape yShape = {{3200000, 5}, {3200000, 5}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 5);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_big_mask_dim6)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{10, 10, 10, 10, 10, 10}, {10, 10, 10, 10, 10, 10}};
    gert::StorageShape yShape = {{1000000, 6}, {1000000, 6}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 6);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_big_mask_dim7)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{8, 8, 8, 8, 8, 8, 8}, {8, 8, 8, 8, 8, 8, 8}};
    gert::StorageShape yShape = {{2097152, 7}, {2097152, 7}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 7);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_big_mask_dim8)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{6, 6, 6, 6, 6, 6, 6, 6}, {6, 6, 6, 6, 6, 6, 6, 6}};
    gert::StorageShape yShape = {{1679616, 8}, {1679616, 8}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 8);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_null_tensor)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{0}, {0}};
    gert::StorageShape yShape = {{2, 0}, {2, 0}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, 30001);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_output_int32)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{128, 2}, {128, 2}};
    gert::StorageShape yShape = {{159, 2}, {159, 2}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

TEST_F(NonZeroTilingAscendcTest, NonZeroTiling_ascendc_fp16_dtype)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    gert::StorageShape xShape = {{128, 2}, {128, 2}};
    gert::StorageShape yShape = {{159, 2}, {159, 2}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(false)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)));

    optiling::NonZeroCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;
    compileInfo.block_dim = 64;
    std::string opType("NonZero");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}