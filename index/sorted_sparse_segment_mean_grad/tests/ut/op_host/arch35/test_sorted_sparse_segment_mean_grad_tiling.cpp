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
 * \file test_sorted_sparse_segment_mean_grad_tiling.cpp
 * \brief dynamic tiling test of sorted_sparse_segment_mean_grad
 */
#include <iostream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/sorted_sparse_segment_mean_grad_tiling_base.h"
#include "../../../../op_host/arch35/sorted_sparse_segment_mean_grad_tiling_simt_tiling.h"

using namespace ut_util;
using namespace ge;

class SortedSparseSegmentMeanGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SortedSparseSegmentMeanGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SortedSparseSegmentMeanGradTiling TearDown" << std::endl;
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
        {{data_size}, {data_size}},
        {ge::FORMAT_ND, ge::FORMAT_ND, {}},
        gert::kFollowing,
        dtype,
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

static void RunTilingTest(gert::StorageShape xShape, gert::StorageShape sortedIndicesShape,
                          gert::StorageShape preLocationIndicesShape, gert::StorageShape segmentIdsShape,
                          gert::StorageShape outputDim0Shape, gert::StorageShape yShape,
                          ge::DataType xDtype, ge::DataType sortedIndicesDtype,
                          ge::DataType preLocationIndicesDtype, ge::DataType segmentIdsDtype,
                          int32_t outputDim0Value, bool expectSuccess,
                          std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& constTensors)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910_95"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::SortedSparseSegmentMeanGradCompileInfo compile_info;
    compile_info.ubSize = 253952;
    compile_info.coreNum = 64;

    std::string op_type("SortedSparseSegmentMeanGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    int32_t outputDim0Data[1] = {outputDim0Value};
    SetConstInput(4, DT_INT32, outputDim0Data, 1, constTensors);

    auto holder = gert::TilingContextFaker()
        .SetOpType(op_type)
        .NodeIoNum(5, 1)
        .IrInstanceNum({1,1,1,1,1})
        .InputShapes({&xShape, &sortedIndicesShape, &preLocationIndicesShape, &segmentIdsShape, &outputDim0Shape})
        .OutputShapes({&yShape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, sortedIndicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, preLocationIndicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, segmentIdsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .ConstInput(constTensors)
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    if (expectSuccess) {
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    } else {
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    }
}

TEST_F(SortedSparseSegmentMeanGradTiling, sorted_sparse_segment_mean_grad_simt_tiling_float_int32_int32_int32) {
    gert::StorageShape x = {{8, 64}, {8, 64}};
    gert::StorageShape sorted_indices = {{8}, {8}};
    gert::StorageShape pre_location_indices = {{8}, {8}};
    gert::StorageShape segment_ids = {{8}, {8}};
    gert::StorageShape output_dim0 = {{1}, {1}};
    gert::StorageShape y = {{4, 64}, {4, 64}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> constTensors;
    RunTilingTest(x, sorted_indices, pre_location_indices, segment_ids, output_dim0, y,
                  ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, 4, true, constTensors);
}

TEST_F(SortedSparseSegmentMeanGradTiling, sorted_sparse_segment_mean_grad_simt_tiling_float16_int32_int32_int32) {
    gert::StorageShape x = {{8, 64}, {8, 64}};
    gert::StorageShape sorted_indices = {{8}, {8}};
    gert::StorageShape pre_location_indices = {{8}, {8}};
    gert::StorageShape segment_ids = {{8}, {8}};
    gert::StorageShape output_dim0 = {{1}, {1}};
    gert::StorageShape y = {{4, 64}, {4, 64}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> constTensors;
    RunTilingTest(x, sorted_indices, pre_location_indices, segment_ids, output_dim0, y,
                  ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, 4, true, constTensors);
}

TEST_F(SortedSparseSegmentMeanGradTiling, sorted_sparse_segment_mean_grad_simt_tiling_bf16_int32_int32_int32) {
    gert::StorageShape x = {{8, 64}, {8, 64}};
    gert::StorageShape sorted_indices = {{8}, {8}};
    gert::StorageShape pre_location_indices = {{8}, {8}};
    gert::StorageShape segment_ids = {{8}, {8}};
    gert::StorageShape output_dim0 = {{1}, {1}};
    gert::StorageShape y = {{4, 64}, {4, 64}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> constTensors;
    RunTilingTest(x, sorted_indices, pre_location_indices, segment_ids, output_dim0, y,
                  ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, 4, true, constTensors);
}

TEST_F(SortedSparseSegmentMeanGradTiling, sorted_sparse_segment_mean_grad_simt_tiling_float_int64_int32_int64) {
    gert::StorageShape x = {{8, 64}, {8, 64}};
    gert::StorageShape sorted_indices = {{8}, {8}};
    gert::StorageShape pre_location_indices = {{8}, {8}};
    gert::StorageShape segment_ids = {{8}, {8}};
    gert::StorageShape output_dim0 = {{1}, {1}};
    gert::StorageShape y = {{4, 64}, {4, 64}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> constTensors;
    RunTilingTest(x, sorted_indices, pre_location_indices, segment_ids, output_dim0, y,
                  ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, 4, true, constTensors);
}

TEST_F(SortedSparseSegmentMeanGradTiling, sorted_sparse_segment_mean_grad_simt_tiling_float_int32_int64_int32) {
    gert::StorageShape x = {{8, 64}, {8, 64}};
    gert::StorageShape sorted_indices = {{8}, {8}};
    gert::StorageShape pre_location_indices = {{8}, {8}};
    gert::StorageShape segment_ids = {{8}, {8}};
    gert::StorageShape output_dim0 = {{1}, {1}};
    gert::StorageShape y = {{4, 64}, {4, 64}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> constTensors;
    RunTilingTest(x, sorted_indices, pre_location_indices, segment_ids, output_dim0, y,
                  ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, 4, true, constTensors);
}

TEST_F(SortedSparseSegmentMeanGradTiling, sorted_sparse_segment_mean_grad_simt_tiling_failed_unsupported_dtype) {
    gert::StorageShape x = {{8, 64}, {8, 64}};
    gert::StorageShape sorted_indices = {{8}, {8}};
    gert::StorageShape pre_location_indices = {{8}, {8}};
    gert::StorageShape segment_ids = {{8}, {8}};
    gert::StorageShape output_dim0 = {{1}, {1}};
    gert::StorageShape y = {{4, 64}, {4, 64}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> constTensors;
    RunTilingTest(x, sorted_indices, pre_location_indices, segment_ids, output_dim0, y,
                  ge::DT_DOUBLE, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, 4, false, constTensors);
}