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
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../op_host/arch35/max_pool_grad_with_argmax_v3_tiling_base.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MaxPoolGradWithArgmaxV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPoolGradWithArgmaxV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPoolGradWithArgmaxV3Tiling TearDown" << std::endl;
    }
};

static void ExecuteTestCase(
    gert::StorageShape xShape, gert::StorageShape yShape, gert::StorageShape gradShape, gert::StorageShape argmaxShape,
    std::vector<int64_t> ksize, std::vector<int64_t> strides, std::vector<int64_t> pads, std::vector<int64_t> dilation,
    ge::DataType dtype, int64_t index_dtype, ge::DataType index_dtype_enum, bool ceil_mode, std::string data_format,
    uint64_t except_tilingkey, std::string expect)
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
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910_95"}};
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolGradWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolGradWithArgmaxV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &gradShape, &argmaxShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, index_dtype_enum, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(index_dtype)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceil_mode)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
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
    //    auto tilingData = tiling_context->GetRawTilingData();
    //    ASSERT_NE(tilingData, nullptr);
    //    dlog_setlevel(0, 3, 0);
    //    auto tiling_data_result = to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize());
    //    std::cout<<tiling_data_result<<std::endl;
    //    EXPECT_EQ(tiling_data_result, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test1)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    ;
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape gradShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    std::vector<int64_t> ksize = {64, 64};
    std::vector<int64_t> strides = {64, 64};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    std::string expect =
        "1 1 64 64 64 64 64 64 0 0 1 1 1 1 6 64 64 1 64 64 1 1 1 6 61440 30720 30720 1 1 1 1 1 1 1 1 1 1 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test2)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {64, 64};
    std::vector<int64_t> strides = {64, 64};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    std::string expect =
        "1 1 64 64 64 64 64 64 0 0 1 1 1 1 6 64 64 1 64 64 1 1 1 6 61440 20480 40960 1 1 1 1 1 1 1 1 1 1 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test3)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {64, 64};
    std::vector<int64_t> strides = {64, 64};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_BF16;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    std::string expect =
        "1 1 64 64 64 64 64 64 0 0 1 1 1 1 6 64 64 1 64 64 1 1 1 6 61440 20480 40960 1 1 1 1 1 1 1 1 1 1 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test4)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {64, 64};
    std::vector<int64_t> strides = {64, 64};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT64;
    int64_t index_dtype = 9;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    std::string expect =
        "1 1 64 64 64 64 64 64 0 0 1 1 1 1 6 64 64 1 64 64 1 1 1 6 61440 20480 40960 1 1 1 1 1 1 1 1 1 1 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test5)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {64, 64};
    std::vector<int64_t> strides = {64, 64};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    ge::DataType dtype_index = ge::DT_INT64;
    int64_t index_dtype = 9;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    std::string expect =
        "1 1 64 64 64 64 64 64 0 0 1 1 1 1 6 64 64 1 64 64 1 1 1 6 61440 12288 49152 1 1 1 1 1 1 1 1 1 1 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test6)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {64, 64};
    std::vector<int64_t> strides = {64, 64};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_BF16;
    ge::DataType dtype_index = ge::DT_INT64;
    int64_t index_dtype = 9;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    std::string expect =
        "1 1 64 64 64 64 64 64 0 0 1 1 1 1 6 64 64 1 64 64 1 1 1 6 61440 12288 49152 1 1 1 1 1 1 1 1 1 1 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test7)
{
    gert::StorageShape xShape = {{33, 12, 2, 22}, {33, 12, 2, 22}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{33, 12, 1, 4}, {33, 12, 1, 4}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 6};
    std::vector<int64_t> strides = {2, 6};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    std::string expect = "1 4 2 22 2 6 2 6 0 0 1 1 6 6 66 2 2 1 22 22 1 2 2 33 1152 384 384 1 1 100 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test8)
{
    gert::StorageShape xShape = {{33, 12, 2, 22}, {33, 12, 2, 22}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{33, 12, 1, 4}, {33, 12, 1, 4}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 6};
    std::vector<int64_t> strides = {2, 6};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    std::string expect = "1 4 2 22 2 6 2 6 0 0 1 1 6 6 66 2 2 1 22 22 1 2 2 33 1536 384 768 1 1 100 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test9)
{
    gert::StorageShape xShape = {{33, 12, 2, 22}, {33, 12, 2, 22}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{33, 12, 1, 4}, {33, 12, 1, 4}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 6};
    std::vector<int64_t> strides = {2, 6};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_BF16;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    std::string expect = "1 4 2 22 2 6 2 6 0 0 1 1 6 6 66 2 2 1 22 22 1 2 2 33 1536 384 768 1 1 100 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test10)
{
    gert::StorageShape xShape = {{33, 12, 2, 22}, {33, 12, 2, 22}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{33, 12, 1, 4}, {33, 12, 1, 4}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 6};
    std::vector<int64_t> strides = {2, 6};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT64;
    int64_t index_dtype = 9;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    std::string expect = "1 4 2 22 2 6 2 6 0 0 1 1 6 6 66 2 2 1 22 22 1 2 2 33 1152 384 768 1 1 100 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test11)
{
    gert::StorageShape xShape = {{33, 12, 2, 22}, {33, 12, 2, 22}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{33, 12, 1, 4}, {33, 12, 1, 4}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 6};
    std::vector<int64_t> strides = {2, 6};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    ge::DataType dtype_index = ge::DT_INT64;
    int64_t index_dtype = 9;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    std::string expect = "1 4 2 22 2 6 2 6 0 0 1 1 6 6 66 2 2 1 22 22 1 2 2 33 1536 384 1536 1 1 100 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NCHW_Test12)
{
    gert::StorageShape xShape = {{33, 12, 2, 22}, {33, 12, 2, 22}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{33, 12, 1, 4}, {33, 12, 1, 4}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 6};
    std::vector<int64_t> strides = {2, 6};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_BF16;
    ge::DataType dtype_index = ge::DT_INT64;
    int64_t index_dtype = 9;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    std::string expect = "1 4 2 22 2 6 2 6 0 0 1 1 6 6 66 2 2 1 22 22 1 2 2 33 1536 384 1536 1 1 100 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NHWC_Test1)
{
    gert::StorageShape xShape = {{16, 18, 30, 2}, {16, 18, 30, 2}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{16, 10, 16, 2}, {16, 10, 16, 2}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 201;
    std::string expect = "10 16 2 18 30 2 2 2 2 1 1 1 1 1 1 16 5 3 4 30 30 1 2 2 1 1 1 64 4800 1792 1792 1 1 201 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NHWC_Test2)
{
    gert::StorageShape xShape = {{16, 18, 30, 2}, {16, 18, 30, 2}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{16, 10, 16, 2}, {16, 10, 16, 2}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 201;
    std::string expect = "10 16 2 18 30 2 2 2 2 1 1 1 1 1 1 16 5 3 4 30 30 1 2 2 1 1 1 64 9600 1792 3328 1 1 201 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NHWC_Test3)
{
    gert::StorageShape xShape = {{16, 18, 30, 2}, {16, 18, 30, 2}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{16, 10, 16, 2}, {16, 10, 16, 2}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_BF16;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 201;
    std::string expect = "10 16 2 18 30 2 2 2 2 1 1 1 1 1 1 16 5 3 4 30 30 1 2 2 1 1 1 64 9600 1792 3328 1 1 201 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}

TEST_F(MaxPoolGradWithArgmaxV3Tiling, MaxPoolGradWithArgmaxV3Tiling_NHWC_Test4)
{
    gert::StorageShape xShape = {{2, 64, 64, 3}, {2, 64, 64, 3}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 1, 1, 3}, {2, 1, 1, 3}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {64, 64};
    std::vector<int64_t> strides = {64, 64};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 201;
    std::string expect = "1 1 3 64 64 64 64 64 64 0 0 1 1 1 1 2 2 2 32 64 64 1 3 3 1 1 1 64 4096 384 384 1 1 201 ";
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, dtype_index,
        ceil_mode, data_format, except_tilingkey, expect);
}
