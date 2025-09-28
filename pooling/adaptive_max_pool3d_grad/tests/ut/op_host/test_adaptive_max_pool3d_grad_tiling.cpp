/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_adaptive_max_pool3d_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry_base.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "adaptive_max_pool3d_grad_compile_info.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AdaptiveMaxPool3DGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveMaxPool3DGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveMaxPool3DGradTiling TearDown" << std::endl;
    }
};

void TestAdaptiveMaxPool3DGradTiling(
    gert::StorageShape& xShape, gert::StorageShape& gradShape, gert::StorageShape& argmaxShape,
    gert::StorageShape& dxShape, std::vector<std::pair<std::string, ge::AnyValue>>& AttrList, ge::DataType dataType,
    uint64_t expectTilingKey)
{
    // dlog_setlevel(0, 0, 0);
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string COMPILE_INFO_STRING_910B = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 40}})";
    GetPlatFormInfos(COMPILE_INFO_STRING_910B.c_str(), socInfos, aicoreSpec, intrinsics);

    // Platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // Compile info
    optiling::Tiling4AdaptiveMaxPool3DGradCompileInfo compileInfo;

    std::string op_type("AdaptiveMaxPool3DGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // TilingParseFunc simulate
    auto kernelHolder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(COMPILE_INFO_STRING_910B.c_str()), reinterpret_cast<void*>(&platformInfo)})
            .Outputs({&compileInfo})
            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaptiveMaxPool3DGrad")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &gradShape, &argmaxShape})
                      .OutputShapes({&dxShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dataType, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, dataType, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, dataType, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeAttrs(AttrList)
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

    auto realTilingKey = tilingContext->GetTilingKey();
    ASSERT_EQ(realTilingKey, expectTilingKey);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(AdaptiveMaxPool3DGradTiling, adaptive_max_pool3d_grad_tilingkey_2_networkcase_0)
{
    std::cout << "run case: " << "adaptive_max_pool3d_grad_tilingkey_2_networkcase_0" << std::endl;
    // network case
    gert::StorageShape xShape = {{5, 640, 1, 64, 64}, {5, 640, 1, 64, 64}};
    gert::StorageShape gradShape = {{5, 640, 1, 1, 1}, {5, 640, 1, 1, 1}};
    gert::StorageShape argmaxShape = {{5, 640, 1, 1, 1}, {5, 640, 1, 1, 1}};
    gert::StorageShape dxShape = {{5, 640, 1, 64, 64}, {5, 640, 1, 64, 64}};
    std::vector<std::pair<std::string, ge::AnyValue>> attrList = {};
    TestAdaptiveMaxPool3DGradTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 2);
}

TEST_F(AdaptiveMaxPool3DGradTiling, adaptive_max_pool3d_grad_tilingkey_0_networkcase_0)
{
    std::cout << "run case: " << "adaptive_max_pool3d_grad_tilingkey_0_networkcase_0" << std::endl;
    // network case
    gert::StorageShape xShape = {{39, 39, 14, 16, 30}, {39, 39, 14, 16, 30}};
    gert::StorageShape gradShape = {{39, 39, 14, 8, 5}, {39, 39, 14, 8, 5}};
    gert::StorageShape argmaxShape = {{39, 39, 14, 8, 5}, {39, 39, 14, 8, 5}};
    gert::StorageShape dxShape = {{39, 39, 14, 16, 30}, {39, 39, 14, 16, 30}};
    std::vector<std::pair<std::string, ge::AnyValue>> attrList = {};
    TestAdaptiveMaxPool3DGradTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT16, 0);
}
