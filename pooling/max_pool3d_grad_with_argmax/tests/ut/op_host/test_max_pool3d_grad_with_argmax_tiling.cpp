/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_max_pool3d_grad_with_argmax_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace optiling {
struct Tiling4MaxPool3DGradWithArgmaxCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};
} // namespace optiling

class MaxPool3dGradWithArgmaxTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPool3dGradWithArgmaxTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPool3dGradWithArgmaxTiling TearDown" << std::endl;
    }
};

void TestMaxPool3dGradWithArgmaxTiling(
    gert::StorageShape& xShape, gert::StorageShape& gradShape, gert::StorageShape& argmaxShape,
    gert::StorageShape& dxShape, std::vector<std::pair<std::string, Ops::NN::AnyValue>>& AttrList, ge::DataType dataType,
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
    optiling::Tiling4MaxPool3DGradWithArgmaxCompileInfo compileInfo;

    std::string op_type("MaxPool3DGradWithArgmax");
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
                      .SetOpType("MaxPool3DGradWithArgmax")
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

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_0)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_0" << std::endl;
    gert::StorageShape xShape = {{1, 40, 16, 90, 160}, {1, 40, 16, 90, 160}};
    gert::StorageShape gradShape = {{1, 40, 16, 45, 80}, {1, 40, 16, 45, 80}};
    gert::StorageShape argmaxShape = {{1, 40, 16, 45, 80}, {1, 40, 16, 45, 80}};
    gert::StorageShape dxShape = {{1, 40, 16, 90, 160}, {1, 40, 16, 90, 160}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_0_case_1)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_0" << std::endl;
    gert::StorageShape xShape = {{1, 40, 16, 9, 9}, {1, 40, 16, 9, 9}};
    gert::StorageShape gradShape = {{1, 40, 16, 4, 4}, {1, 40, 16, 4, 4}};
    gert::StorageShape argmaxShape = {{1, 40, 16, 4, 4}, {1, 40, 16, 4, 4}};
    gert::StorageShape dxShape = {{1, 40, 16, 9, 9}, {1, 40, 16, 9, 9}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3, 3})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 100);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_0_case_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_0" << std::endl;
    gert::StorageShape xShape = {{1, 40, 16, 16, 16}, {1, 40, 16, 16, 16}};
    gert::StorageShape gradShape = {{1, 40, 16, 8, 8}, {1, 40, 16, 8, 8}};
    gert::StorageShape argmaxShape = {{1, 40, 16, 8, 8}, {1, 40, 16, 8, 8}};
    gert::StorageShape dxShape = {{1, 40, 16, 16, 16}, {1, 40, 16, 16, 16}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_0_case_3)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_0" << std::endl;
    gert::StorageShape xShape = {{1, 40, 16, 32, 32}, {1, 40, 16, 32, 32}};
    gert::StorageShape gradShape = {{1, 40, 16, 16, 16}, {1, 40, 16, 16, 16}};
    gert::StorageShape argmaxShape = {{1, 40, 16, 16, 16}, {1, 40, 16, 16, 16}};
    gert::StorageShape dxShape = {{1, 40, 16, 32, 32}, {1, 40, 16, 32, 32}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_0_case_4)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_0" << std::endl;
    gert::StorageShape xShape = {{64, 40, 1, 4, 4}, {64, 40, 1, 4, 4}};
    gert::StorageShape gradShape = {{64, 40, 1, 2, 2}, {64, 40, 1, 2, 2}};
    gert::StorageShape argmaxShape = {{64, 40, 1, 2, 2}, {64, 40, 1, 2, 2}};
    gert::StorageShape dxShape = {{64, 40, 1, 4, 4}, {64, 40, 1, 4, 4}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_0_case_5)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_0_case_5" << std::endl;
    gert::StorageShape xShape = {{64, 40, 1, 15, 15}, {64, 40, 1, 15, 15}};
    gert::StorageShape gradShape = {{64, 40, 1, 15, 15}, {64, 40, 1, 15, 15}};
    gert::StorageShape argmaxShape = {{64, 40, 1, 15, 15}, {64, 40, 1, 15, 15}};
    gert::StorageShape dxShape = {{64, 40, 1, 15, 15}, {64, 40, 1, 15, 15}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 9, 9})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 4, 4})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_0_case_cut_w)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_0" << std::endl;
    gert::StorageShape xShape = {{64, 1, 1, 4, 400}, {64, 1, 1, 4, 400}};
    gert::StorageShape gradShape = {{64, 1, 1, 2, 200}, {64, 1, 1, 2, 200}};
    gert::StorageShape argmaxShape = {{64, 1, 1, 2, 200}, {64, 1, 1, 2, 200}};
    gert::StorageShape dxShape = {{64, 1, 1, 4, 400}, {64, 1, 1, 4, 400}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_31_networkcase_0)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_31_networkcase_0" << std::endl;
    // network case
    gert::StorageShape xShape = {{64, 48, 10, 78, 146}, {64, 48, 10, 78, 146}};
    gert::StorageShape gradShape = {{64, 48, 1, 6, 8}, {64, 48, 1, 6, 8}};
    gert::StorageShape argmaxShape = {{64, 48, 1, 6, 8}, {64, 48, 1, 6, 8}};
    gert::StorageShape dxShape = {{64, 48, 10, 78, 146}, {64, 48, 10, 78, 146}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({5, 1, 6})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({8, 13, 18})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 0, 1})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 31);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_31_networkcase_1)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_31_networkcase_0" << std::endl;
    // network case
    gert::StorageShape xShape = {{64, 1, 10, 78, 146}, {64, 1, 10, 78, 146}};
    gert::StorageShape gradShape = {{64, 1, 1, 6, 8}, {64, 1, 1, 6, 8}};
    gert::StorageShape argmaxShape = {{64, 1, 1, 6, 8}, {64, 1, 1, 6, 8}};
    gert::StorageShape dxShape = {{64, 1, 10, 78, 146}, {64, 1, 10, 78, 146}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({5, 1, 6})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({8, 13, 18})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 0, 1})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 31);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_100000_to_0)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_100000_to_0" << std::endl;
    gert::StorageShape xShape = {{1, 8, 6, 6, 6}, {1, 8, 6, 6, 6}};
    gert::StorageShape gradShape = {{1, 8, 3, 3, 3}, {1, 8, 3, 3, 3}};
    gert::StorageShape argmaxShape = {{1, 8, 3, 3, 3}, {1, 8, 3, 3, 3}};
    gert::StorageShape dxShape = {{1, 8, 6, 6, 6}, {1, 8, 6, 6, 6}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_100001_to_0)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_100001_to_0" << std::endl;
    gert::StorageShape xShape = {{1, 16, 6, 6, 6}, {1, 16, 6, 6, 6}};
    gert::StorageShape gradShape = {{1, 16, 3, 3, 3}, {1, 16, 3, 3, 3}};
    gert::StorageShape argmaxShape = {{1, 16, 3, 3, 3}, {1, 16, 3, 3, 3}};
    gert::StorageShape dxShape = {{1, 16, 6, 6, 6}, {1, 16, 6, 6, 6}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT16, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_100002_to_0)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_100002_to_0" << std::endl;
    gert::StorageShape xShape = {{1, 8, 6, 6, 6}, {1, 8, 6, 6, 6}};
    gert::StorageShape gradShape = {{1, 8, 3, 3, 3}, {1, 8, 3, 3, 3}};
    gert::StorageShape argmaxShape = {{1, 8, 3, 3, 3}, {1, 8, 3, 3, 3}};
    gert::StorageShape dxShape = {{1, 8, 6, 6, 6}, {1, 8, 6, 6, 6}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_BF16, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_110000_to_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_110000_to_2" << std::endl;
    gert::StorageShape xShape = {{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}};
    gert::StorageShape gradShape = {{1, 8, 5, 5, 4}, {1, 8, 5, 5, 4}};
    gert::StorageShape argmaxShape = {{1, 8, 5, 5, 4}, {1, 8, 5, 5, 4}};
    gert::StorageShape dxShape = {{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3, 3})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3, 3})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_210001_to_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_210001_to_2" << std::endl;
    gert::StorageShape xShape = {{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}};
    gert::StorageShape gradShape = {{1, 8, 5, 5, 4}, {1, 8, 5, 5, 4}};
    gert::StorageShape argmaxShape = {{1, 8, 5, 5, 4}, {1, 8, 5, 5, 4}};
    gert::StorageShape dxShape = {{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3, 3})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3, 3})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT16, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_110002_to_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_110002_to_2" << std::endl;
    gert::StorageShape xShape = {{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}};
    gert::StorageShape gradShape = {{1, 8, 5, 5, 4}, {1, 8, 5, 5, 4}};
    gert::StorageShape argmaxShape = {{1, 8, 5, 5, 4}, {1, 8, 5, 5, 4}};
    gert::StorageShape dxShape = {{1, 8, 16, 16, 16}, {1, 8, 16, 16, 16}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3, 3})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3, 3})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_BF16, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_111000_to_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_111000_to_2" << std::endl;
    gert::StorageShape xShape = {{1, 8, 64, 64, 64}, {1, 8, 64, 64, 64}};
    gert::StorageShape gradShape = {{1, 8, 32, 32, 31}, {1, 8, 32, 32, 31}};
    gert::StorageShape argmaxShape = {{1, 8, 32, 32, 31}, {1, 8, 32, 32, 31}};
    gert::StorageShape dxShape = {{1, 8, 64, 64, 64}, {1, 8, 64, 64, 64}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_211001_to_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_211001_to_2" << std::endl;
    gert::StorageShape xShape = {{1, 16, 64, 64, 64}, {1, 16, 64, 64, 64}};
    gert::StorageShape gradShape = {{1, 16, 32, 32, 31}, {1, 16, 32, 32, 31}};
    gert::StorageShape argmaxShape = {{1, 16, 32, 32, 31}, {1, 16, 32, 32, 31}};
    gert::StorageShape dxShape = {{1, 16, 64, 64, 64}, {1, 16, 64, 64, 64}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT16, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_111002_to_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_111002_to_2" << std::endl;
    gert::StorageShape xShape = {{1, 8, 64, 64, 64}, {1, 8, 64, 64, 64}};
    gert::StorageShape gradShape = {{1, 8, 32, 32, 31}, {1, 8, 32, 32, 31}};
    gert::StorageShape argmaxShape = {{1, 8, 32, 32, 31}, {1, 8, 32, 32, 31}};
    gert::StorageShape dxShape = {{1, 8, 64, 64, 64}, {1, 8, 64, 64, 64}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_BF16, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_111100_to_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_111100_to_2" << std::endl;
    gert::StorageShape xShape = {{1, 8, 16, 16, 1024}, {1, 8, 16, 16, 1024}};
    gert::StorageShape gradShape = {{1, 8, 8, 8, 511}, {1, 8, 8, 8, 511}};
    gert::StorageShape argmaxShape = {{1, 8, 8, 8, 511}, {1, 8, 8, 8, 511}};
    gert::StorageShape dxShape = {{1, 8, 16, 16, 1024}, {1, 8, 16, 16, 1024}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_211101_to_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_211101_to_2" << std::endl;
    gert::StorageShape xShape = {{1, 16, 16, 16, 1024}, {1, 16, 16, 16, 1024}};
    gert::StorageShape gradShape = {{1, 16, 8, 8, 511}, {1, 16, 8, 8, 511}};
    gert::StorageShape argmaxShape = {{1, 16, 8, 8, 511}, {1, 16, 8, 8, 511}};
    gert::StorageShape dxShape = {{1, 16, 16, 16, 1024}, {1, 16, 16, 16, 1024}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT16, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_111102_to_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_111102_to_2" << std::endl;
    gert::StorageShape xShape = {{1, 8, 16, 16, 1024}, {1, 8, 16, 16, 1024}};
    gert::StorageShape gradShape = {{1, 8, 8, 8, 511}, {1, 8, 8, 8, 511}};
    gert::StorageShape argmaxShape = {{1, 8, 8, 8, 511}, {1, 8, 8, 8, 511}};
    gert::StorageShape dxShape = {{1, 8, 16, 16, 1024}, {1, 8, 16, 16, 1024}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_BF16, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_2_case_nc_small)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_2" << std::endl;
    gert::StorageShape xShape = {{4, 1, 190, 126, 96}, {4, 1, 190, 126, 96}};
    gert::StorageShape gradShape = {{4, 1, 8, 6, 24}, {4, 1, 8, 6, 24}};
    gert::StorageShape argmaxShape = {{4, 1, 8, 6, 24}, {4, 1, 8, 6, 24}};
    gert::StorageShape dxShape = {{4, 1, 190, 126, 96}, {4, 1, 190, 126, 96}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({22, 19, 4})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({22, 19, 4})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_2_case_cut_do)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_2" << std::endl;
    gert::StorageShape xShape = {{256, 4, 190, 126, 96}, {256, 4, 190, 126, 96}};
    gert::StorageShape gradShape = {{256, 4, 8, 6, 24}, {256, 4, 8, 6, 24}};
    gert::StorageShape argmaxShape = {{256, 4, 8, 6, 24}, {256, 4, 8, 6, 24}};
    gert::StorageShape dxShape = {{256, 4, 190, 126, 96}, {256, 4, 190, 126, 96}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({22, 19, 4})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({22, 19, 4})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_2_case_cut_ho)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_2" << std::endl;
    gert::StorageShape xShape = {{1, 1, 3200, 126, 96}, {1, 1, 3200, 126, 96}};
    gert::StorageShape gradShape = {{1, 1, 160, 6, 24}, {1, 1, 160, 6, 24}};
    gert::StorageShape argmaxShape = {{1, 1, 160, 6, 24}, {1, 1, 160, 6, 24}};
    gert::StorageShape dxShape = {{1, 1, 3200, 126, 96}, {1, 1, 3200, 126, 96}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({20, 19, 4})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({20, 19, 4})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_2_case_cut_wo_1)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_2" << std::endl;
    gert::StorageShape xShape = {{1, 1, 1, 26000, 960}, {1, 1, 1, 26000, 960}};
    gert::StorageShape gradShape = {{1, 1, 1, 1300, 20}, {1, 1, 1, 1300, 20}};
    gert::StorageShape argmaxShape = {{1, 1, 1, 1300, 20}, {1, 1, 1, 1300, 20}};
    gert::StorageShape dxShape = {{1, 1, 1, 26000, 960}, {1, 1, 1, 26000, 960}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 20, 48})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 20, 48})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 2);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, max_pool3d_grad_with_argmax_tilingkey_2_case_cut_wo_2)
{
    std::cout << "run case: " << "max_pool3d_grad_with_argmax_tilingkey_2" << std::endl;
    gert::StorageShape xShape = {{1, 1, 1, 40, 2400000}, {1, 1, 1, 40, 2400000}};
    gert::StorageShape gradShape = {{1, 1, 1, 1, 60000}, {1, 1, 1, 1, 60000}};
    gert::StorageShape argmaxShape = {{1, 1, 1, 1, 60000}, {1, 1, 1, 1, 60000}};
    gert::StorageShape dxShape = {{1, 1, 1, 40, 2400000}, {1, 1, 1, 40, 2400000}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {
        {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 40, 40})},
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 40, 40})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
        {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
    TestMaxPool3dGradWithArgmaxTiling(xShape, gradShape, argmaxShape, dxShape, attrList, ge::DT_FLOAT, 2);
}

static void ExecuteTestCase(gert::StorageShape xShape, gert::StorageShape gradShape,gert::StorageShape argmaxShape,
                            gert::StorageShape yShape, std::vector<int64_t> ksize, std::vector<int64_t> strides,
                            std::vector<int64_t> pads, std::vector<int64_t> dilation, ge::DataType dtype,
                            int64_t index_dtype, bool ceil_mode, std::string data_format, uint64_t except_tilingkey,
                            std::string expect)
{
    dlog_setlevel(0, 0, 0);

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::Tiling4MaxPool3DGradWithArgmaxCompileInfo compile_info;

    std::string op_type("MaxPool3DGradWithArgmax");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ge::DataType indicesDtype = (index_dtype == 3) ? ge::DT_INT32 : ge::DT_INT64;
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape, &gradShape, &argmaxShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceil_mode)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(index_dtype)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, except_tilingkey);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    dlog_setlevel(0, 3, 0);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, MaxPool3DGradWithArgmax_tiling_simt_test_0001_NCDHW)
{
    gert::StorageShape xShape = {{2,4,5,6,3}, {2,4,5,6,3}};
    gert::StorageShape gradShape = {{2,2,2,3,3}, {2,2,2,3,3}};
    gert::StorageShape argmaxShape = {{2,2,2,3,3}, {2,2,2,3,3}};
    gert::StorageShape yShape = {{2,4,5,6,3}, {2,4,5,6,3}};
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {2,2,2};
    std::vector<int64_t> pads = {0,0,0};
    std::vector<int64_t> dilation = {1,1,1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NDHWC";
    uint64_t except_tilingkey = 769;
    std::string expect = " ";
    ExecuteTestCase(xShape, gradShape, argmaxShape,  yShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                    data_format, except_tilingkey, expect);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, MaxPool3DGradWithArgmax_tiling_simd_test_0001_NCDHW)
{
    gert::StorageShape xShape = {{1,2,8,10,514}, {1,2,8,10,514}};
    gert::StorageShape gradShape = {{1,2,4,5,257}, {1,2,4,5,257}};
    gert::StorageShape argmaxShape = {{1,2,4,5,257}, {1,2,4,5,257}};
    gert::StorageShape yShape = {{1,2,8,10,514}, {1,2,8,10,514}};
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {2,2,2};
    std::vector<int64_t> pads = {0,0,0};
    std::vector<int64_t> dilation = {1,1,1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCDHW";
    uint64_t except_tilingkey = 1;
    std::string expect = " ";
    ExecuteTestCase(xShape, gradShape, argmaxShape,  yShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                    data_format, except_tilingkey, expect);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, MaxPool3DGradWithArgmax_tiling_simd_test_0002_NCDHW)
{
    gert::StorageShape xShape = {{2,2,10,11,1024}, {2,2,10,11,1024}};
    gert::StorageShape gradShape = {{2,2,11,6,342}, {2,2,11,6,342}};
    gert::StorageShape argmaxShape = {{2,2,11,6,342}, {2,2,11,6,342}};
    gert::StorageShape yShape = {{2,2,10,11,1024}, {2,2,10,11,1024}};
    std::vector<int64_t> ksize = {2,4,5};
    std::vector<int64_t> strides = {1,2,3};
    std::vector<int64_t> pads = {1,2,2};
    std::vector<int64_t> dilation = {1,1,1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCDHW";
    uint64_t except_tilingkey = 1025;
    std::string expect = " ";
    ExecuteTestCase(xShape, gradShape, argmaxShape,  yShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                    data_format, except_tilingkey, expect);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, MaxPool3DGradWithArgmax_tiling_simd_test_0003_NCDHW)
{
    gert::StorageShape xShape = {{38,235,4,4,4}, {38,235,4,4,4}};
    gert::StorageShape gradShape = {{38,235,3,3,3}, {38,235,3,3,3}};
    gert::StorageShape argmaxShape = {{38,235,3,3,3}, {38,235,3,3,3}};
    gert::StorageShape yShape = {{38,235,4,4,4}, {38,235,4,4,4}};
    std::vector<int64_t> ksize = {2,2,2};
    std::vector<int64_t> strides = {2,2,2};
    std::vector<int64_t> pads = {1,1,1};
    std::vector<int64_t> dilation = {1,1,1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCDHW";
    uint64_t except_tilingkey = 1;
    std::string expect = " ";
    ExecuteTestCase(xShape, gradShape, argmaxShape,  yShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                    data_format, except_tilingkey, expect);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, MaxPool3DGradWithArgmax_tiling_simd_test_0004_NCDHW)
{
    gert::StorageShape xShape = {{2,2,9,10,960}, {2,2,9,10,960}};
    gert::StorageShape gradShape = {{2,2,9,5,240}, {2,2,9,5,240}};
    gert::StorageShape argmaxShape = {{2,2,9,5,240}, {2,2,9,5,240}};
    gert::StorageShape yShape = {{2,2,9,10,960}, {2,2,9,10,960}};
    std::vector<int64_t> ksize = {3,2,7};
    std::vector<int64_t> strides = {1,2,4};
    std::vector<int64_t> pads = {1,0,2};
    std::vector<int64_t> dilation = {1,1,1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCDHW";
    uint64_t except_tilingkey = 1025;
    std::string expect = " ";
    ExecuteTestCase(xShape, gradShape, argmaxShape,  yShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                    data_format, except_tilingkey, expect);
}

TEST_F(MaxPool3dGradWithArgmaxTiling, MaxPool3DGradWithArgmax_tiling_simd_test_0005_NCDHW)
{
    gert::StorageShape xShape = {{1,8,12,14,1152}, {1,8,12,14,1152}};
    gert::StorageShape gradShape = {{1,8,11,7,287}, {1,8,11,7,287}};
    gert::StorageShape argmaxShape = {{1,8,11,7,287}, {1,8,11,7,287}};
    gert::StorageShape yShape = {{1,8,12,14,1152}, {1,8,12,14,1152}};
    std::vector<int64_t> ksize = {4,3,7};
    std::vector<int64_t> strides = {1,2,4};
    std::vector<int64_t> pads = {1,1,1};
    std::vector<int64_t> dilation = {1,1,1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCDHW";
    uint64_t except_tilingkey = 1025;
    std::string expect = " ";
    ExecuteTestCase(xShape, gradShape, argmaxShape,  yShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                    data_format, except_tilingkey, expect);
}