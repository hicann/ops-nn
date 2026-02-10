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
#include "../../../op_host/max_pool_grad_with_argmax_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MaxPoolGradWithArgmaxTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPoolGradWithArgmaxTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPoolGradWithArgmaxTiling TearDown" << std::endl;
    }
};

static void ExecuteTestCase(
    gert::StorageShape xShape, gert::StorageShape yShape, gert::StorageShape gradShape, gert::StorageShape argmaxShape,
    std::vector<int64_t> ksize, std::vector<int64_t> strides, std::string padding, ge::DataType dtype,  ge::DataType dtypeIdx,
    bool include_batch_in_index, std::string data_format, uint64_t except_tilingkey)
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
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolGradWithArgmaxCompileInfo compile_info;

    std::string op_type("MaxPoolGradWithArgmax");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs(std::vector<void*>{&compile_info})
            .Build();

    ASSERT_TRUE((kernel_holder.GetContext<gert::TilingParseContext>()) ->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func((kernel_holder.GetContext<gert::KernelContext>())), ge::GRAPH_SUCCESS);

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
                      .NodeInputTd(2, dtypeIdx, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)},
                           {"include_batch_in_index", Ops::NN::AnyValue::CreateFrom<bool>(include_batch_in_index)},
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
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    optiling::MaxPoolGradWithArgmaxBaseTiling* base = new optiling::MaxPoolGradWithArgmaxBaseTiling(tiling_context);
    base->GetShapeAttrsInfo();
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, except_tilingkey);
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NHWC_Test701)
{
    gert::StorageShape xShape = {{23, 29, 31, 36}, {23, 29, 31, 36}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{23, 8, 6, 36}, {23, 8, 6, 36}};
    gert::StorageShape gradShape = {{23, 8, 6, 36}, {23, 8, 6, 36}};
    std::vector<int64_t> ksize = {1, 4, 5, 1};
    std::vector<int64_t> strides = {1, 4, 6, 1};
    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT16;
    ge::DataType dtypeIdx = ge::DT_INT64;
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 701;

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dtype, dtypeIdx,
        include_batch_in_index, data_format, except_tilingkey);
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NHWC_Test700)
{
    gert::StorageShape xShape = {{81, 29, 11, 17}, {81, 29, 11, 17}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{81, 3, 3, 17}, {81, 3, 3, 17}};
    gert::StorageShape gradShape = argmaxShape;

    std::vector<int64_t> ksize = {1, 6, 4, 1};
    std::vector<int64_t> strides = {1, 11, 4, 1};
    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_BF16;
    ge::DataType dtypeIdx = ge::DT_INT64;
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 700;

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dtype, dtypeIdx,
        include_batch_in_index, data_format, except_tilingkey);
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NCHW_Test801)
{
    gert::StorageShape xShape = {{2, 2, 3911, 2}, {2, 2, 3911, 2}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 2, 1955, 1}, {2, 2, 1955, 1}};
    gert::StorageShape gradShape = {{2, 2, 1955, 1}, {2, 2, 1955, 1}};
    std::vector<int64_t> ksize = {1, 1, 2, 2};
    std::vector<int64_t> strides = {1, 1, 2, 366};
    std::string pads = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 801;

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dtype, dtypeIdx,
        include_batch_in_index, data_format, except_tilingkey);
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NCHW_Test803)
{
    gert::StorageShape xShape = {{2, 2, 268435457, 2}, {2, 2, 268435457, 2}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 2, 134217728, 1}, {2, 2, 134217728, 1}};
    gert::StorageShape gradShape = argmaxShape;

    std::vector<int64_t> ksize = {1, 1, 2, 2};
    std::vector<int64_t> strides = {1, 1, 2, 366};
    std::string pads = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 803;

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dtype, dtypeIdx,
        include_batch_in_index, data_format, except_tilingkey);
}


TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NHWC_Test802)
{
    gert::StorageShape xShape = {{3, 3, 1870, 6}, {3, 3, 1870, 6}};
    gert::StorageShape yShape = xShape;

    gert::StorageShape gradShape = {{3, 2, 935, 6}, {3, 2, 935, 6}};
    gert::StorageShape argmaxShape = {{3, 2, 935, 6}, {3, 2, 935, 6}};

    std::vector<int64_t> ksize = {1, 4377, 2, 1};
    std::vector<int64_t> strides = {1, 2, 2, 1};

    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 802;

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape,
        ksize, strides, pads,
        dtype, dtypeIdx,
        include_batch_in_index,
        data_format,
        except_tilingkey
    );
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NHWC_Test804)
{
    gert::StorageShape xShape = {{2, 10000, 10000, 11}, {2, 10000, 10000, 11}};
    gert::StorageShape yShape = xShape;

    gert::StorageShape gradShape = {{2, 5000, 5000, 11}, {2, 5000, 5000, 11}};
    gert::StorageShape argmaxShape = {{2, 5000, 5000, 11}, {2, 5000, 5000, 11}};

    std::vector<int64_t> ksize = {1, 4377, 2, 1};
    std::vector<int64_t> strides = {1, 2, 2, 1};

    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 804;

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape,
        ksize, strides, pads,
        dtype, dtypeIdx,
        include_batch_in_index,
        data_format,
        except_tilingkey
    );
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NHWC_Test500)
{
    gert::StorageShape xShape = {{4, 6679, 2, 2}, {4, 6679, 2, 2}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{4, 3340, 1, 2}, {4, 3340, 1, 2}};
    gert::StorageShape gradShape = {{4, 3340, 1, 2}, {4, 3340, 1, 2}};
    std::vector<int64_t> ksize = {{1, 2, 2, 1}};
    std::vector<int64_t> strides = {{1, 2, 2, 1}};
    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT16;
    ge::DataType dtypeIdx = ge::DT_INT64;
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 500;
    
    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dtype, dtypeIdx,
        include_batch_in_index, data_format, except_tilingkey);
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NHWC_Test501)
{
    gert::StorageShape xShape = {{5, 8, 4, 4}, {5, 8, 4, 4}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{5, 3, 2, 4}, {5, 3, 2, 4}};
    gert::StorageShape gradShape = argmaxShape;

    std::vector<int64_t> ksize = {{1, 4, 2, 1}};
    std::vector<int64_t> strides = {{1, 3, 3, 1}};
    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT64;  
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 501;  

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dtype, dtypeIdx,
        include_batch_in_index, data_format, except_tilingkey);
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NHWC_Test510)
{
    gert::StorageShape xShape = {{1, 67108865, 32, 1}, {1, 67108865, 32, 1}};
    gert::StorageShape yShape = xShape;

    gert::StorageShape gradShape = {{1, 33554433, 16, 1}, {1, 33554433, 16, 1}};
    gert::StorageShape argmaxShape = {{1, 33554433, 16, 1}, {1, 33554433, 16, 1}};

    std::vector<int64_t> ksize = {{1, 2, 2, 1}};
    std::vector<int64_t> strides = {{1, 2, 2, 1}};

    std::string pads = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT64;   
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 510;      

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape,
        ksize, strides, pads,
        dtype, dtypeIdx,  
        include_batch_in_index,
        data_format,
        except_tilingkey
    );
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NHWC_Test601)
{
    gert::StorageShape xShape = {{2, 11, 15184, 3}, {2, 11, 15184, 3}};
    gert::StorageShape yShape = xShape;

    gert::StorageShape gradShape = {{2, 5, 7111, 3}, {2, 5, 7111, 3}};
    gert::StorageShape argmaxShape = {{2, 5, 7111, 3}, {2, 5, 7111, 3}};

    std::vector<int64_t> ksize = {1, 2, 963, 1};
    std::vector<int64_t> strides = {1, 2, 2, 1};

    std::string pads = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;      
    ge::DataType dtypeIdx = ge::DT_INT32;   
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 601;       

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape,
        ksize, strides, pads,
        dtype, dtypeIdx,  
        include_batch_in_index,
        data_format,
        except_tilingkey
    );
}

TEST_F(MaxPoolGradWithArgmaxTiling, MaxPoolGradWithArgmaxTiling_NHWC_Test600)
{
    gert::StorageShape xShape = {{16, 18, 23, 16}, {16, 18, 23, 16}};
    gert::StorageShape yShape = xShape;

    gert::StorageShape gradShape = {{16, 3, 4, 16}, {16, 3, 4, 16}};
    gert::StorageShape argmaxShape = {{16, 3, 4, 16}, {16, 3, 4, 16}};

    std::vector<int64_t> ksize = {1, 6, 5, 1};
    std::vector<int64_t> strides = {1, 6, 5, 1};

    std::string pads = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;      
    ge::DataType dtypeIdx = ge::DT_INT64;   
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 600;       

    ExecuteTestCase(
        xShape, yShape, gradShape, argmaxShape,
        ksize, strides, pads,
        dtype, dtypeIdx,  
        include_batch_in_index,
        data_format,
        except_tilingkey
    );
}
