/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/ge_glu_grad_v2_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

class GeGluGradV2Tiling : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        cout << "GeGluGradV2Tiling SetUpTestSuite" << endl;
        compileInfoStr = R"({"hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 196608,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 48
                                        }})";

        GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
        platformInfo.Init();

        string opType("GeGluGradV2");
        ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
        tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
        tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    }

    static void TearDownTestSuite()
    {
        cout << "GeGluGradV2Tiling TearDownTestSuite" << endl;
    }

protected:
    static map<string, string> socInfos;
    static map<string, string> aicoreSpec;
    static map<string, string> intrinsics;
    static string compileInfoStr;
    // platform info
    static fe::PlatFormInfos platformInfo;
    // compile info
    static optiling::GeGluGradV2CompileInfo compileInfo;
    static gert::OpImplKernelRegistry::KernelFunc tilingParseFunc;
    static gert::OpImplKernelRegistry::TilingKernelFunc tilingFunc;
};

map<string, string> GeGluGradV2Tiling::socInfos;
map<string, string> GeGluGradV2Tiling::aicoreSpec;
map<string, string> GeGluGradV2Tiling::intrinsics;
string GeGluGradV2Tiling::compileInfoStr;
// platform info
fe::PlatFormInfos GeGluGradV2Tiling::platformInfo;
// compile info
optiling::GeGluGradV2CompileInfo GeGluGradV2Tiling::compileInfo;
gert::OpImplKernelRegistry::KernelFunc GeGluGradV2Tiling::tilingParseFunc = nullptr;
gert::OpImplKernelRegistry::TilingKernelFunc GeGluGradV2Tiling::tilingFunc = nullptr;

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_parse_func)
{
    // tilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_001)
{
    gert::StorageShape dyShape = {{23, 12, 1024}, {23, 12, 1024}};
    gert::StorageShape xShape = {{23, 12, 1024 * 2}, {23, 12, 1024 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 101);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_002)
{
    gert::StorageShape dyShape = {{23, 12, 5457}, {23, 12, 5457}};
    gert::StorageShape xShape = {{23, 12, 5457 * 2}, {23, 12, 5457 * 2}};

    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 102);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_003)
{
    gert::StorageShape dyShape = {{1132, 71, 7}, {1132, 12, 7}};
    gert::StorageShape xShape = {{1132, 71, 7 * 2}, {1132, 12, 7 * 2}};

    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 103);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_004)
{
    gert::StorageShape dyShape = {{23, 12, 1024}, {23, 12, 1024}};
    gert::StorageShape xShape = {{23, 12, 1024 * 2}, {23, 12, 1024 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 701);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp16_001)
{
    gert::StorageShape dyShape = {{24, 576, 5120}, {24, 576, 5120}};
    gert::StorageShape xShape = {{24, 576, 5120 * 2}, {24, 576, 5120 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 201);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp16_002)
{
    gert::StorageShape dyShape = {{24, 576, 6145}, {24, 576, 6145}};
    gert::StorageShape xShape = {{24, 576, 6145 * 2}, {24, 576, 6145 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 202);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp16_003)
{
    gert::StorageShape dyShape = {{24, 576, 13}, {24, 576, 13}};
    gert::StorageShape xShape = {{24, 576, 13 * 2}, {24, 576, 13 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 203);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp16_004)
{
    gert::StorageShape dyShape = {{24, 576, 1033}, {24, 576, 1033}};
    gert::StorageShape xShape = {{24, 576, 1033 * 2}, {24, 576, 1033 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 801);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_001)
{
    gert::StorageShape dyShape = {{24, 576, 3120}, {24, 576, 3120}};
    gert::StorageShape xShape = {{24, 576, 3120 * 2}, {24, 576, 3120 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 301);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_002)
{
    gert::StorageShape dyShape = {{24, 576, 4465}, {24, 576, 4465}};
    gert::StorageShape xShape = {{24, 576, 4465 * 2}, {24, 576, 4465 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 302);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_003)
{
    gert::StorageShape dyShape = {{24, 576, 13}, {24, 576, 13}};
    gert::StorageShape xShape = {{24, 576, 13 * 2}, {24, 576, 13 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 303);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_004)
{
    gert::StorageShape dyShape = {{24, 576, 1033}, {24, 576, 1033}};
    gert::StorageShape xShape = {{24, 576, 1033 * 2}, {24, 576, 1033 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 901);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_005)
{
    gert::StorageShape dyShape = {{24, 576, 1033}, {24, 576, 1033}};
    gert::StorageShape xShape = {{24, 576, 1033 * 2}, {24, 576, 1033 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {64, 262144, platform_ascendc::SocVersion::ASCEND910_95, true};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 901);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_310p_erf)
{
    gert::StorageShape dyShape = {{24, 576, 1033}, {24, 576, 1033}};
    gert::StorageShape xShape = {{24, 576, 1033 * 2}, {24, 576, 1033 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND310P, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_310p)
{
    gert::StorageShape dyShape = {{23, 12, 1024}, {23, 12, 1024}};
    gert::StorageShape xShape = {{23, 12, 1024 * 2}, {23, 12, 1024 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND310P, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs(
                                {{"dim", ge::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                 {"approximate", ge::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                 {"activate_left", ge::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}
