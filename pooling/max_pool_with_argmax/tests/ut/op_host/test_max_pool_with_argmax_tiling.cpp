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
#include "../../../op_host/max_pool_with_argmax_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

template <typename T>
static string to_string(void *buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

class MaxPoolWithArgmaxTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPoolWithArgmaxTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPoolWithArgmaxTiling TearDown" << std::endl;
    }
};

static void ExecuteTestCase(
    gert::StorageShape& xStorageShape, gert::StorageShape& yStorageShape, gert::StorageShape& argmaxStorageShape,
    std::vector<int64_t> ksize, std::vector<int64_t> strides, std::string pads, ge::DataType dtype, int64_t index_dtype,
    bool includeBatchInIndex, std::string data_format, bool nan_prop, uint64_t except_tilingkey, std::string expect)
{
    dlog_setlevel(0, 0, 0);

    gert::Shape xShape = xStorageShape.GetStorageShape();
    gert::Shape yShape = yStorageShape.GetStorageShape();
    gert::Shape argmaxShape = argmaxStorageShape.GetStorageShape();

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
    optiling::MaxPoolWithArgmaxCompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmax");
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

    std::vector<gert::Shape*> input_shape_ptrs = {&xShape};
    std::vector<gert::Shape*> output_shape_ptrs = {&yShape, &argmaxShape};
    ge::DataType argmax_dtype = (index_dtype == 3) ? ge::DT_INT32 : ge::DT_INT64;
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes(input_shape_ptrs)
                      .OutputShapes(output_shape_ptrs)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(pads)},
                           {"Targmax", Ops::NN::AnyValue::CreateFrom<int64_t>(index_dtype)},
                           {"includeBatchInIndex", Ops::NN::AnyValue::CreateFrom<bool>(includeBatchInIndex)},
                           {"dataFormat", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                           {"nanProp", Ops::NN::AnyValue::CreateFrom<bool>(nan_prop)}})
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
    auto tiling_data_result = to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize());
    std::cout << tiling_data_result << std::endl;
    EXPECT_EQ(tiling_data_result, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Big_C_800001)
{
    gert::StorageShape xStorageShape = {{1, 3, 3, 2}, {1, 3, 3, 2}};
    gert::StorageShape yStorageShape = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    gert::StorageShape argmaxStorageShape = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    std::vector<int64_t> ksize = {1, 2, 2, 1};
    std::vector<int64_t> strides = {1, 2, 2, 1};
    std::string pads = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool includeBatchInIndex = false;
    std::string data_format = "NHWC";
    bool nan_prop = false;
    uint64_t except_tilingkey = 800001;
    std::string expect = "2 3 3 1 1 2 2 2 2 0 0 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 128 32 32 0 0 0 0 0 0 0 0 800001 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Big_C_800002)
{
    gert::StorageShape xStorageShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yStorageShape = {{2, 1, 1, 64}, {2, 1, 1, 64}};
    gert::StorageShape argmaxStorageShape = {{2, 1, 1, 64}, {2, 1, 1, 64}};
    std::vector<int64_t> ksize = {1, 64, 64, 1};
    std::vector<int64_t> strides = {1, 64, 64, 1};
    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool includeBatchInIndex = false;
    std::string data_format = "NHWC";
    bool nan_prop = false;
    uint64_t except_tilingkey = 800002;
    std::string expect = "64 3 64 1 1 64 64 64 64 0 30 1 1 2 1 1 1 1 1 1 64 64 1 1 1 2 114688 256 256 1 1 7 1 10 64 64 1 800002 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Big_C_800011)
{
    gert::StorageShape xStorageShape = {{1, 3, 3, 2}, {1, 3, 3, 2}};
    gert::StorageShape yStorageShape = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    gert::StorageShape argmaxStorageShape = {{1, 1, 1, 2}, {1, 1, 1, 2}};
    std::vector<int64_t> ksize = {1, 2, 2, 1};
    std::vector<int64_t> strides = {1, 2, 2, 1};
    std::string pads = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool includeBatchInIndex = false;
    std::string data_format = "NHWC";
    bool nan_prop = true;
    uint64_t except_tilingkey = 800011;
    std::string expect = "2 3 3 1 1 2 2 2 2 0 0 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 128 32 32 0 0 0 0 0 0 0 0 800011 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Big_C_800012)
{
    gert::StorageShape xStorageShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yStorageShape = {{2, 1, 1, 64}, {2, 1, 1, 64}};
    gert::StorageShape argmaxStorageShape = {{2, 1, 1, 64}, {2, 1, 1, 64}};
    std::vector<int64_t> ksize = {1, 64, 64, 1};
    std::vector<int64_t> strides = {1, 64, 64, 1};
    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool includeBatchInIndex = false;
    std::string data_format = "NHWC";
    bool nan_prop = true;
    uint64_t except_tilingkey = 800012;
    std::string expect = "64 3 64 1 1 64 64 64 64 0 30 1 1 2 1 1 1 1 1 1 64 64 1 1 1 2 114688 256 256 1 1 7 1 10 64 64 1 800012 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Simt_Nchw_500001)
{
    gert::StorageShape xStorageShape = {{7, 8, 8, 12}, {7, 8, 8, 12}};
    gert::StorageShape yStorageShape = {{7, 8, 2, 2}, {7, 8, 2, 2}};
    gert::StorageShape argmaxStorageShape = {{7, 8, 2, 2}, {7, 8, 2, 2}};
    std::vector<int64_t> ksize = {1, 1, 4, 5};
    std::vector<int64_t> strides = {1, 1, 6, 6};
    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 9;
    bool includeBatchInIndex = false;
    std::string data_format = "NCHW";
    bool nan_prop = false;
    uint64_t except_tilingkey = 500001;
    std::string expect = "224 1 7 8 8 12 2 2 4 5 6 6 1 0 0 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Simt_Nhwc_500102)
{
    gert::StorageShape xStorageShape = {{11, 19, 7, 15}, {11, 19, 7, 15}};
    gert::StorageShape yStorageShape = {{11, 4, 2, 15}, {11, 4, 2, 15}};
    gert::StorageShape argmaxStorageShape = {{11, 4, 2, 15}, {11, 4, 2, 15}};
    std::vector<int64_t> ksize = {1, 4, 3, 1};
    std::vector<int64_t> strides = {1, 6, 4, 1};
    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool includeBatchInIndex = false;
    std::string data_format = "NHWC";
    bool nan_prop = true;
    uint64_t except_tilingkey = 500102;
    std::string expect = "256 6 11 15 19 7 4 2 4 3 6 4 1 0 0 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Simt_Nchw_500101_test1)
{
    gert::StorageShape xStorageShape = {{19, 23, 20, 23}, {19, 23, 20, 23}};
    gert::StorageShape yStorageShape = {{19, 23, 3, 3}, {19, 23, 3, 3}};
    gert::StorageShape argmaxStorageShape = {{19, 23, 3, 3}, {19, 23, 3, 3}};
    std::vector<int64_t> ksize = {1, 1, 6, 6};
    std::vector<int64_t> strides = {1, 1, 6, 8};
    std::string pads = "VALID";
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool includeBatchInIndex = false;
    std::string data_format = "NCHW";
    bool nan_prop = true;
    uint64_t except_tilingkey = 500101;
    std::string expect = "256 16 19 23 20 23 3 3 6 6 6 8 0 0 0 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Simt_Nchw_500101_test2)
{
    gert::StorageShape xStorageShape = {{19, 23, 20, 23}, {19, 23, 20, 23}};
    gert::StorageShape yStorageShape = {{19, 23, 4, 3}, {19, 23, 4, 3}};
    gert::StorageShape argmaxStorageShape = {{19, 23, 4, 3}, {19, 23, 4, 3}};
    std::vector<int64_t> ksize = {1, 1, 6, 6};
    std::vector<int64_t> strides = {1, 1, 6, 8};
    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool includeBatchInIndex = false;
    std::string data_format = "NCHW";
    bool nan_prop = true;
    uint64_t except_tilingkey = 500101;
    std::string expect = "256 21 19 23 20 23 4 3 6 6 6 8 2 0 0 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Small_C_700001)
{
    gert::StorageShape xStorageShape = {{15, 12, 17, 13}, {15, 12, 17, 13}};
    gert::StorageShape yStorageShape = {{15, 4, 5, 13}, {15, 4, 5, 13}};
    gert::StorageShape argmaxStorageShape = {{15, 4, 5, 13}, {15, 4, 5, 13}};
    std::vector<int64_t> ksize = {1, 3, 4, 1};
    std::vector<int64_t> strides = {1, 3, 3, 1};
    std::string pads = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool includeBatchInIndex = false;
    std::string data_format = "NHWC";
    bool nan_prop = false;
    uint64_t except_tilingkey = 700001;
    std::string expect = "13 12 17 4 5 3 4 3 3 0 0 1 1 15 1 1 4 4 1 2 13 13 1 2 2 60 2496 256 256 0 0 0 0 0 0 0 0 700001 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxTiling, MaxPoolWithArgmaxTiling_Small_C_700011)
{
    gert::StorageShape xStorageShape = {{15, 12, 17, 13}, {15, 12, 17, 13}};
    gert::StorageShape yStorageShape = {{15, 4, 5, 13}, {15, 4, 5, 13}};
    gert::StorageShape argmaxStorageShape = {{15, 4, 5, 13}, {15, 4, 5, 13}};
    std::vector<int64_t> ksize = {1, 3, 4, 1};
    std::vector<int64_t> strides = {1, 3, 3, 1};
    std::string pads = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool includeBatchInIndex = false;
    std::string data_format = "NHWC";
    bool nan_prop = true;
    uint64_t except_tilingkey = 700011;
    std::string expect = "13 12 17 4 5 3 4 3 3 0 0 1 1 15 1 1 4 4 1 2 13 13 1 2 2 60 2496 256 256 0 0 0 0 0 0 0 0 700011 ";
    ExecuteTestCase(
        xStorageShape, yStorageShape, argmaxStorageShape, ksize, strides, pads, dtype, index_dtype, includeBatchInIndex, data_format,
        nan_prop, except_tilingkey, expect);
}