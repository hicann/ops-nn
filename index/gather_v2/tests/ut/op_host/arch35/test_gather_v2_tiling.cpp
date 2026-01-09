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
#include <vector>
#include <fstream>

#include <gtest/gtest.h>

#include "array_ops.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/gather_v2_tiling_arch35.h"
#include "test_cube_util.h"
#include "graph/graph.h"
#include "../../../../op_host/arch35/gather_v2_tiling.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;

class GatherV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GatherV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GatherV2Tiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int64_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
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

template <typename T>
void SetConstInput(
    size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder =
        std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    static int64_t offset = 0;
    gert::Tensor tensor(
        {{data_size}, {data_size}},         // shape
        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
        gert::kFollowing,                   // placement
        dtype,                              // dt
        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    offset += sizeof(T) * data_size;
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

TEST_F(GatherV2TilingTest, gather_v2_simt_tiling_1)
{
    gert::StorageShape params_shape = {{2, 5}, {2, 5}};
    gert::StorageShape indices_shape = {{2, 3}, {2, 3}};
    gert::StorageShape axis_shape = {{1}, {1}};
    gert::StorageShape y_shape = {{2, 3, 5}, {2, 3, 5}};
    ge::DataType params_dtype = ge::DT_INT32;
    ge::DataType indices_dtype = ge::DT_INT32;

    int64_t axis = 0;
    int64_t batch_dims = 0;

    int64_t deq_data[1] = {0};
    deq_data[0] = axis;
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> axis_tensors;
    SetConstInput(2, DT_INT32, deq_data, 1, axis_tensors);

    string compile_info_string = R"({
 	         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
 	                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
 	                           "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
 	                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
 	                           "CORE_NUM": 64}
 	                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GatherV2CompileInfo compile_info;

    std::string op_type("GatherV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&params_shape, &indices_shape, &axis_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, params_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indices_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, params_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(axis_tensors)
                      .NodeAttrs(
                          {{"batch_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(batch_dims)},
                           {"negative_index_support", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 2000000004);

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_data_result << std::endl;
}

TEST_F(GatherV2TilingTest, gather_v2_simt_tiling_2)
{
    gert::StorageShape params_shape = {{11, 69}, {11, 69}};
    gert::StorageShape indices_shape = {{5, 3, 16, 7, 3, 1, 7}, {5, 3, 16, 7, 3, 1, 7}};
    gert::StorageShape axis_shape = {{1}, {1}};
    gert::StorageShape y_shape = {{5, 3, 16, 7, 3, 1, 7, 69}, {5, 3, 16, 7, 3, 1, 7, 69}};
    ge::DataType params_dtype = ge::DT_FLOAT16;
    ge::DataType indices_dtype = ge::DT_INT32;

    int64_t axis = 0;
    int64_t batch_dims = 0;

    int64_t deq_data[1] = {0};
    deq_data[0] = axis;
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> axis_tensors;
    SetConstInput(2, DT_INT32, deq_data, 1, axis_tensors);

    string compile_info_string = R"({
 	         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
 	                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
 	                           "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
 	                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
 	                           "CORE_NUM": 64}
 	                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GatherV2CompileInfo compile_info;

    std::string op_type("GatherV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&params_shape, &indices_shape, &axis_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, params_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indices_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, params_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(axis_tensors)
                      .NodeAttrs(
                          {{"batch_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(batch_dims)},
                           {"negative_index_support", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 3000); // 新增SIMD模板收编

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_data_result << std::endl;
}