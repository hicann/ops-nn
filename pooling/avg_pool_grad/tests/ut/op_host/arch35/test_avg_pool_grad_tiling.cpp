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
#include "../../../../op_host/arch35/avg_pool_grad_tiling_base.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AvgPoolGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AvgPoolGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AvgPoolGradTiling TearDown" << std::endl;
    }
};

template <typename T>
void SetConstInput(
    size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder =
        std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor(
        {{data_size}, {data_size}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, dtype, nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

static void ExecuteTestCase(
    gert::StorageShape xShape, gert::StorageShape yShape, gert::StorageShape gradShape,
    std::vector<int64_t> ksize, std::vector<int64_t> strides, std::string padding,
    std::string data_format,
    ge::DataType dtype,  ge::DataType dtypeIdx, uint64_t except_tilingkey,
    int32_t* shape_data)
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
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "ascend950"}};
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::AvgPoolGradCompileInfo compile_info;

    std::string op_type("AvgPoolGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(0, DT_INT32, shape_data, 4, const_tensors);

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
    ASSERT_EQ(tiling_parse_func((kernel_holder.GetContext<gert::KernelContext>())), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &gradShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtypeIdx, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
                      .TilingData(param.get())
                      .ConstInput(const_tensors)
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
}

TEST_F(AvgPoolGradTiling, AvgPoolGradTiling_Test_1)
{
    gert::StorageShape xShape = {{4}, {4}};
    gert::StorageShape gradShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape yShape = {{1, 3, 3, 1}, {1, 3, 3, 1}};
    std::vector<int64_t> ksize = {3, 3};
    std::vector<int64_t> strides = {1, 1};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 274;
    int shape_data[4] = {1, 3, 3, 1};

    ExecuteTestCase(
        xShape, yShape, gradShape, ksize, strides, padding,
        data_format, dtype, dtypeIdx, except_tilingkey, shape_data);
}

TEST_F(AvgPoolGradTiling, AvgPoolGradTiling_Test_2)
{
    gert::StorageShape xShape = {{4}, {4}};
    gert::StorageShape gradShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape yShape = {{1, 1, 3, 3}, {1, 1, 3, 3}};
    std::vector<int64_t> ksize = {3, 3};
    std::vector<int64_t> strides = {1, 1};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 258;
    int shape_data[4] = {1, 1, 3, 3};

    ExecuteTestCase(
        xShape, yShape, gradShape, ksize, strides, padding,
        data_format, dtype, dtypeIdx, except_tilingkey, shape_data);
}

TEST_F(AvgPoolGradTiling, AvgPoolGradTiling_Test_3)
{
    gert::StorageShape xShape = {{4}, {4}};
    gert::StorageShape gradShape = {{1, 3, 3, 1}, {1, 3, 3, 1}};
    gert::StorageShape yShape = {{1, 5, 5, 1}, {1, 5, 5, 1}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::string padding = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    std::string data_format = "NHWC";
    uint64_t except_tilingkey = 1297;
    int shape_data[4] = {1, 5, 5, 1};

    ExecuteTestCase(
        xShape, yShape, gradShape, ksize, strides, padding,
        data_format, dtype, dtypeIdx, except_tilingkey, shape_data);
}