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
 * \file test_max_pool_v3_grad_tiling.cpp
 * \brief Tiling UT for max_pool_v3_grad operator (nn TilingContextFaker pattern).
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

using namespace ut_util;
using namespace std;
using namespace ge;

class MaxPoolV3GradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolV3GradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaxPoolV3GradTilingTest TearDown" << std::endl; }
};

// Tiling 测试参数：输入输出 shape、属性、期望返回值和期望 tiling key
struct TilingTestCase {
    std::vector<int64_t> origInput{1, 1, 4, 4};
    std::vector<int64_t> origOutput{1, 1, 2, 2};
    std::vector<int64_t> grad{1, 1, 2, 2};
    std::vector<int64_t> outGrad{1, 1, 4, 4};
    std::vector<int64_t> ksize{1, 1, 2, 2};
    std::vector<int64_t> strides{1, 1, 2, 2};
    std::string paddingMode = "CALCULATED";
    std::vector<int64_t> pads{0, 0, 0, 0};
    std::string dataFormat = "NCHW";
    bool globalPooling = false;
    bool ceilMode = false;
    ge::graphStatus expectResult = ge::GRAPH_SUCCESS;
    uint64_t expectTilingKey = 258;
};

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableShape().SetDim(i, dims[i]);
    }
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

static void DoTilingTest(ge::DataType dt, const TilingTestCase& tc)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 262144,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 64
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct MaxPoolV3GradCompileInfo {
    } compile_info;

    std::string op_type("MaxPoolV3Grad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(4, 2)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
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

    gert::StorageShape origInputShape = MakeStorageShape(tc.origInput);
    gert::StorageShape origOutputShape = MakeStorageShape(tc.origOutput);
    gert::StorageShape gradShape = MakeStorageShape(tc.grad);
    gert::StorageShape outGradShape = MakeStorageShape(tc.outGrad);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&origInputShape, &origOutputShape, &gradShape})
                      .OutputShapes({&outGradShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(tc.ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(tc.strides)},
                                  {"padding_mode", Ops::NN::AnyValue::CreateFrom<std::string>(tc.paddingMode)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(tc.pads)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(tc.dataFormat)},
                                  {"global_pooling", Ops::NN::AnyValue::CreateFrom<bool>(tc.globalPooling)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(tc.ceilMode)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    EXPECT_EQ(tiling_func(tiling_context), tc.expectResult);
    if (tc.expectResult != ge::GRAPH_SUCCESS) {
        return;
    }

    auto tiling_key = tiling_context->GetTilingKey();
    std::cout << "tiling_key: " << tiling_key << std::endl;
    ASSERT_EQ(tiling_key, tc.expectTilingKey);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_float32)
{
    TilingTestCase tc;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_float16)
{
    TilingTestCase tc;
    DoTilingTest(ge::DT_FLOAT16, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_float32)
{
    TilingTestCase tc;
    tc.origInput = {1, 4, 4, 1};
    tc.origOutput = {1, 2, 2, 1};
    tc.grad = {1, 2, 2, 1};
    tc.outGrad = {1, 4, 4, 1};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 274;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_global_pooling_true)
{
    TilingTestCase tc;
    tc.origOutput = {1, 1, 1, 1};
    tc.grad = {1, 1, 1, 1};
    tc.strides = {1, 1, 1, 1};
    tc.globalPooling = true;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_ceil_mode_true)
{
    TilingTestCase tc;
    tc.ksize = {1, 1, 3, 3};
    tc.strides = {1, 1, 2, 2};
    tc.ceilMode = true;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_pads_length2)
{
    TilingTestCase tc;
    tc.pads = {0, 0};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_ksize_length2)
{
    TilingTestCase tc;
    tc.ksize = {2, 2};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_strides_length2)
{
    TilingTestCase tc;
    tc.strides = {2, 2};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
