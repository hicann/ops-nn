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
 * \file test_scatter_elements_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <thread>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "log/log.h"
#include "graph/graph.h"
#include "kernel_run_context_facker.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/sparse_to_dense_tiling.h"

using namespace std;
using namespace ge;

class SparseToDenseTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseToDenseTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseToDenseTiling TearDown" << std::endl;
    }
};

template <typename T>
void SetConstInput(size_t const_index, ge::DataType dtype, T *const_data, int64_t data_size,
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> &const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder =
        std::make_unique<uint8_t[]>(sizeof(gert::Tensor) + sizeof(T) * data_size);
    auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
    gert::Tensor tensor({ { data_size }, { data_size } }, // shape
        { ge::FORMAT_ND, ge::FORMAT_ND, {} },             // format
        gert::kFollowing,                                 // placement
        dtype,                                            // dtype
        nullptr);
    memcpy_s(input_tensor, sizeof(gert::Tensor), &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T *>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{ tensor_data });
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy_s(&tmp, sizeof(tmp), data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

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

template <typename T>
static void ExecuteTestCase(ge::DataType indices_dtype, ge::DataType output_shape_dtype, ge::DataType values_dtype,
                            ge::DataType default_value_dtype, ge::DataType output_dtype, gert::StorageShape indices_shape, gert::StorageShape output_shape_shape,
                            gert::StorageShape values_shape, gert::StorageShape default_value_shape, gert::StorageShape output_shape, bool validate_indices,
                            uint64_t tilingKeyValue, string expectTilingData, T* output_shape_value, int64_t output_dim, ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::SparseToDenseCompileInfo compile_info;

    std::string op_type("SparseToDense");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    if (output_shape_dtype == ge::DT_INT32) {
        SetConstInput(1, ge::DT_INT32, output_shape_value, output_dim, const_tensors);
    } else {
        SetConstInput(1, ge::DT_INT64, output_shape_value, output_dim, const_tensors);
    }

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&indices_shape, &output_shape_shape, &values_shape, &default_value_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, indices_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, output_shape_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, values_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, default_value_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"validate_indices", Ops::NN::AnyValue::CreateFrom<bool>(validate_indices)}})
                      .ConstInput(const_tensors)
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
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    ASSERT_EQ(tiling_key, tilingKeyValue);
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_normal01)
{
    gert::StorageShape shape1 = {{3}, {3}};
    gert::StorageShape shape2 = {{1}, {1}};
    gert::StorageShape shape3 = {{3}, {3}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{12}, {12}};
    int64_t output_dim = 1;
    int32_t output_shape_value[output_dim] = {12};
    string expectTilingData = "1 12 16 12 12 1 1 3 3 1 1 0 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int32_t>(ge::DT_INT32, ge::DT_INT32, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, shape1, shape2, shape3, shape4, 
                            outputshape, false, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_SUCCESS);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_normal02)
{
    gert::StorageShape shape1 = {{10000, 4}, {10000, 4}};
    gert::StorageShape shape2 = {{4}, {4}};
    gert::StorageShape shape3 = {{10000}, {10000}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{12, 200, 11, 12}, {12, 200, 11, 12}};
    int64_t output_dim = 4;
    int32_t output_shape_value[output_dim] = {12, 200, 11, 12};
    string expectTilingData = "4 4950 4960 4950 4950 1 1 157 109 64 64 257 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int32_t>(ge::DT_INT32, ge::DT_INT32, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_SUCCESS);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_normal03)
{
    gert::StorageShape shape1 = {{10000, 4}, {10000, 4}};
    gert::StorageShape shape2 = {{4}, {4}};
    gert::StorageShape shape3 = {{10000}, {10000}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{99998721, 1, 1, 1}, {99998721, 1, 1, 1}};
    int64_t output_dim = 4;
    int32_t output_shape_value[output_dim] = {99998721, 1, 1, 1};
    string expectTilingData = "4 1562481 106496 71537 71474 15 15 157 109 64 64 257 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int32_t>(ge::DT_INT32, ge::DT_INT32, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_SUCCESS);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_error01)
{
    gert::StorageShape shape1 = {{3, 1, 1}, {3, 1, 1}};
    gert::StorageShape shape2 = {{1}, {1}};
    gert::StorageShape shape3 = {{3}, {3}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{12}, {12}};
    int64_t output_dim = 1;
    int32_t output_shape_value[output_dim] = {12};
    string expectTilingData = "3 12 1 12 12 3 3 1 1 1 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int32_t>(ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_FAILED);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_error02)
{
    gert::StorageShape shape1 = {{3}, {3}};
    gert::StorageShape shape2 = {{1, 1}, {1, 1}};
    gert::StorageShape shape3 = {{3}, {3}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{12}, {12}};
    int64_t output_dim = 1;
    int32_t output_shape_value[output_dim] = {12};
    string expectTilingData = "3 12 1 12 12 3 3 1 1 1 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int32_t>(ge::DT_INT32, ge::DT_INT32, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_FAILED);
}


TEST_F(SparseToDenseTiling, test_tiling_ascendc_error03)
{
    gert::StorageShape shape1 = {{3}, {3}};
    gert::StorageShape shape2 = {{1}, {1}};
    gert::StorageShape shape3 = {{3}, {3}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{12}, {12}};
    int64_t output_dim = 1;
    int32_t output_shape_value[output_dim] = {12};
    string expectTilingData = "3 12 1 12 12 3 3 1 1 1 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int32_t>(ge::DT_INT16, ge::DT_INT16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_FAILED);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_error04)
{
    gert::StorageShape shape1 = {{3}, {3}};
    gert::StorageShape shape2 = {{1}, {1}};
    gert::StorageShape shape3 = {{3}, {3}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{12}, {12}};
    int64_t output_dim = 1;
    int64_t output_shape_value[output_dim] = {12};
    string expectTilingData = "3 12 1 12 12 3 3 1 1 1 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int64_t>(ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_FAILED);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_error05)
{
    gert::StorageShape shape1 = {{3}, {3}};
    gert::StorageShape shape2 = {{2}, {2}};
    gert::StorageShape shape3 = {{3}, {3}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{12}, {12}};
    int64_t output_dim = 2;
    int64_t output_shape_value[output_dim] = {12, 1};
    string expectTilingData = "3 12 1 12 12 3 3 1 1 1 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int64_t>(ge::DT_INT64, ge::DT_INT64, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_FAILED);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_error06)
{
    gert::StorageShape shape1 = {{3}, {3}};
    gert::StorageShape shape2 = {{1}, {1}};
    gert::StorageShape shape3 = {{3, 1}, {3, 1}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{12}, {12}};
    int64_t output_dim = 1;
    int64_t output_shape_value[output_dim] = {12};
    string expectTilingData = "3 12 1 12 12 3 3 1 1 1 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int64_t>(ge::DT_INT64, ge::DT_INT64, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_FAILED);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_error07)
{
    gert::StorageShape shape1 = {{3}, {3}};
    gert::StorageShape shape2 = {{1}, {1}};
    gert::StorageShape shape3 = {{4}, {4}};
    gert::StorageShape shape4 = {{1}, {1}};
    gert::StorageShape outputshape = {{12}, {12}};
    int64_t output_dim = 1;
    int64_t output_shape_value[output_dim] = {12};
    string expectTilingData = "3 12 1 12 12 3 3 1 1 1 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int64_t>(ge::DT_INT64, ge::DT_INT64, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_FAILED);
}

TEST_F(SparseToDenseTiling, test_tiling_ascendc_error08)
{
    gert::StorageShape shape1 = {{3}, {3}};
    gert::StorageShape shape2 = {{1}, {1}};
    gert::StorageShape shape3 = {{3}, {3}};
    gert::StorageShape shape4 = {{2}, {2}};
    gert::StorageShape outputshape = {{12}, {12}};
    int64_t output_dim = 1;
    int64_t output_shape_value[output_dim] = {12};
    string expectTilingData = "3 12 1 12 12 3 3 1 1 1 ";
    uint64_t tilingKeyValue = 1000000;

    ExecuteTestCase<int64_t>(ge::DT_INT64, ge::DT_INT64, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, shape1, shape2, shape3, shape4, 
                            outputshape, true, tilingKeyValue, expectTilingData, output_shape_value, output_dim, ge::GRAPH_FAILED);
}