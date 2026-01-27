/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "matmul/sparse_tensor_dense_mat_mul/op_host/op_tiling/sparse_tensor_dense_mat_mul_tiling.h"
#include "test_cube_util.h"
#include "tests/ut/common/ut_op_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SparseTensorDenseMatMulTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SparseTensorDenseMatMulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SparseTensorDenseMatMulTiling TearDown" << std::endl;
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
        {{data_size}, {data_size}},         // shape
        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
        gert::kFollowing,                   // placement
        dtype,                              // dt
        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

void TestSparseTensorDenseMatMulTiling(
    gert::StorageShape& x1_indices_shape, gert::StorageShape& x1_values_shape, gert::StorageShape& x1_shape_shape,
    gert::StorageShape& x2_shape, gert::StorageShape& y_shape, int64_t x1_shape_value[2], ge::DataType x1_indices_dtype,
    ge::DataType x1_values_dtype, ge::DataType x1_shape_dtype, ge::DataType x2_dtype, ge::DataType y_dtype,
    bool adjoint_a_value, bool adjoint_b_value, ge::graphStatus expectTilingResult, uint64_t expectTilingKey)
{
    constexpr size_t INPUT_IDX_X1_SHAPE = 2;
    constexpr int INPUT_SIZE_X1_SHAPE = 2;
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(INPUT_IDX_X1_SHAPE, ge::DT_INT64, x1_shape_value, INPUT_SIZE_X1_SHAPE, const_tensors);

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 262144, "L2_SIZE": 102760448, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 32,
                          "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    version["Short_SoC_version"] = "Ascend910_95";
    version["SoC_version"] = "Ascend910_9589";

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    Ops::NN::Optiling::SparseTensorDenseMatMulCompileInfo compile_info;

    std::string op_type("SparseTensorDenseMatMul");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 1)
                      .SetOpType(op_type)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_indices_shape, &x1_values_shape, &x1_shape_shape, &x2_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, x1_indices_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, x1_values_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, x1_shape_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, x2_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, y_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"adjoint_a", Ops::NN::AnyValue::CreateFrom<bool>(adjoint_a_value)},
                           {"adjoint_b", Ops::NN::AnyValue::CreateFrom<bool>(adjoint_b_value)}})
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
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), expectTilingResult);

    if (expectTilingResult == ge::GRAPH_SUCCESS) {
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, expectTilingKey);
    }
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_001) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{13, 25}, {13, 25}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {25, 33};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, true, true, ge::GRAPH_SUCCESS, 20001);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_002) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {25, 33};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, true, false, ge::GRAPH_SUCCESS, 20002);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_003) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{13, 25}, {13, 25}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, true, ge::GRAPH_SUCCESS, 20003);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_004) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_SUCCESS, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_005) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT16, ge::DT_INT64, ge::DT_FLOAT16, ge::DT_FLOAT16, false, false, ge::GRAPH_SUCCESS, 10004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_006) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 0}, {25, 0}};
    gert::StorageShape y_shape = {{33, 0}, {33, 0}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_SUCCESS, 30000);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_007) {
    gert::StorageShape x1_indices_shape = {{0, 2}, {0, 2}};
    gert::StorageShape x1_values_shape = {{0,}, {0,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{0, 13}, {0, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 0};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_SUCCESS, 40000);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_008) {
    gert::StorageShape x1_indices_shape = {{88, 2, 2}, {88, 2, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_009) {
    gert::StorageShape x1_indices_shape = {{88, 3}, {88, 3}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_010) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{100,}, {100,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_011) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2, 3}, {2, 3}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_012) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{5,}, {5,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_013) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13, 2}, {25, 13, 2}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_014) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_FLOAT,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_015) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_016) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_017) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_018) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, true, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_019) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, true, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_020) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, -13}, {25, -13}};
    gert::StorageShape y_shape = {{33, 13}, {33, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_021) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 13, 2}, {33, 13, 2}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_022) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{2, 13}, {2, 13}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}

TEST_F(SparseTensorDenseMatMulTiling, sparse_tensor_dense_mat_mul_tiling_023) {
    gert::StorageShape x1_indices_shape = {{88, 2}, {88, 2}};
    gert::StorageShape x1_values_shape = {{88,}, {88,}};
    gert::StorageShape x1_shape_shape = {{2,}, {2,}};
    gert::StorageShape x2_shape = {{25, 13}, {25, 13}};
    gert::StorageShape y_shape = {{33, 2}, {33, 2}};
    int64_t x1_shape_value[2] = {33, 25};

    TestSparseTensorDenseMatMulTiling(
        x1_indices_shape, x1_values_shape, x1_shape_shape, x2_shape, y_shape, x1_shape_value, ge::DT_INT64,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, false, false, ge::GRAPH_FAILED, 20004);
}
