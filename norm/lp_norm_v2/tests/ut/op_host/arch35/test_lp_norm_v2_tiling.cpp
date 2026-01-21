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
#include <fstream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "norm/lp_norm_v2/op_host/arch35/lp_norm_v2_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class LpNormV2DavidTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "LpNormV2DavidTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LpNormV2DavidTiling TearDown" << std::endl;
    }
};

TEST_F(LpNormV2DavidTiling, lp_norm_v2_david_tiling1_difftype)
{
    std::string opType("LpNormV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({
                                            "hardware_info": {
                                                "UB_SIZE": 253952,
                                                "CORE_NUM": 64,
                                                "socVersion": "Ascend910_95"
                                            }
                                        })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // compile info
    Ops::Base::ReduceOpCompileInfo compileInfo;
    compileInfo.vectorCoreNum = 64;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape x = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape y = {{2, 2, 1}, {2, 2, 1}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&x})
                      .OutputShapes({&y})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"axes", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({2})},
                           {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0)}})
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
    std::cout << "test>> holder.GetContext end" << std::endl;

    if (tilingFunc == nullptr) {
        std::cout << "test>> tilingFunc is invalid" << std::endl;
    } else {
        std::cout << "test>> tilingFunc is valid" << std::endl;
        EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    }
}

TEST_F(LpNormV2DavidTiling, lp_norm_v2_david_tiling2_difftype)
{
    std::string opType("LpNormV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({
                                            "hardware_info": {
                                                "UB_SIZE": 253952,
                                                "CORE_NUM": 64,
                                                "socVersion": "Ascend910_95"
                                            }
                                        })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // compile info
    Ops::Base::ReduceOpCompileInfo compileInfo;
    compileInfo.vectorCoreNum = 64;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape x = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape y = {{2, 2, 1}, {2, 2, 1}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&x})
                      .OutputShapes({&y})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<float>(2.0)},
                           {"axes", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({2})},
                           {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0)}})
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
    std::cout << "test>> holder.GetContext end" << std::endl;

    if (tilingFunc == nullptr) {
        std::cout << "test>> tilingFunc is invalid" << std::endl;
    } else {
        std::cout << "test>> tilingFunc is valid" << std::endl;
        EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    }
}

TEST_F(LpNormV2DavidTiling, lp_norm_v2_david_tiling3_difftype)
{
    std::string opType("LpNormV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = R"({
                                            "hardware_info": {
                                                "UB_SIZE": 253952,
                                                "CORE_NUM": 64,
                                                "socVersion": "Ascend910_95"
                                            }
                                        })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // compile info
    Ops::Base::ReduceOpCompileInfo compileInfo;
    compileInfo.vectorCoreNum = 64;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    gert::StorageShape x = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape y = {{2, 2, 1}, {2, 2, 1}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&x})
                      .OutputShapes({&y})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                           {"axes", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({2})},
                           {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0)}})
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
    std::cout << "test>> holder.GetContext end" << std::endl;

    if (tilingFunc == nullptr) {
        std::cout << "test>> tilingFunc is invalid" << std::endl;
    } else {
        std::cout << "test>> tilingFunc is valid" << std::endl;
        EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    }
}
