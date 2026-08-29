/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/avg_pool3_d_grad_tiling_base.h"
#include "../../../../op_host/arch35/avg_pool3_d_grad_simt_tiling.h"
#include "../../../../op_kernel/arch35/avg_pool3_d_grad_tiling_data.h"

using namespace std;
using namespace ge;
using namespace AvgPool3DGrad;

class AvgPool3DGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AvgPool3DGradTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AvgPool3DGradTiling TearDown" << std::endl; }
};

template <typename T>
static void SetConstInput(size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
                          std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    // orig_input_shape is a 1D tensor whose data holds the shape values ([N,C,D,H,W] or [C,D,H,W]).
    gert::Tensor tensor({{data_size}, {data_size}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, dtype,
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

static fe::PlatFormInfos InitPlatformInfo(map<string, string>& soc_infos, map<string, string>& aicore_spec,
                                          map<string, string>& intrinsics)
{
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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    return platform_info;
}

// Runs the arch35 tiling and returns the tiling context. Asserts success and expected key.
static gert::TilingContext* RunTiling(const vector<int64_t>& origInput, const gert::StorageShape& gradsShape,
                                      const gert::StorageShape& outputShape, const vector<int64_t>& ksize,
                                      const vector<int64_t>& strides, const vector<int64_t>& pads, bool ceilMode,
                                      bool countIncludePad, int64_t divisorOverride, const string& dataFormat,
                                      ge::DataType dtype, uint64_t expectedKey)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    auto platform_info = InitPlatformInfo(soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    optiling::AvgPool3DGradCompileInfo compile_info;
    std::string op_type("AvgPool3DGrad");
    EXPECT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    EXPECT_NE(param, nullptr);

    int32_t shape_data[5] = {0};
    for (size_t i = 0; i < origInput.size(); i++) {
        shape_data[i] = static_cast<int32_t>(origInput[i]);
    }
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(0, DT_INT32, shape_data, static_cast<int64_t>(origInput.size()), const_tensors);

    gert::StorageShape input_0 = {{static_cast<int64_t>(origInput.size())}, {static_cast<int64_t>(origInput.size())}};
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_0, const_cast<gert::StorageShape*>(&gradsShape)})
                      .OutputShapes({const_cast<gert::StorageShape*>(&outputShape)})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceilMode)},
                                  {"count_include_pad", Ops::NN::AnyValue::CreateFrom<bool>(countIncludePad)},
                                  {"divisor_override", Ops::NN::AnyValue::CreateFrom<int64_t>(divisorOverride)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)}})
                      .TilingData(param.get())
                      .ConstInput(const_tensors)
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), expectedKey);
    return tiling_context;
}

// ============================ TilingBase validation ============================

TEST_F(AvgPool3DGradTiling, base_invalid_format)
{
    // For NCDHW input, data_format must be NCDHW or NDHWC. Use an invalid one.
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    auto platform_info = InitPlatformInfo(soc_infos, aicore_spec, intrinsics);
    auto compile_info = optiling::AvgPool3DGradCompileInfo();
    std::string op_type("AvgPool3DGrad");

    int32_t shape_data[5] = {1, 16, 4, 4, 4};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(0, DT_INT32, shape_data, 5, const_tensors);
    gert::StorageShape input_0 = {{5}, {5}};
    gert::StorageShape gradsShape = {{1, 16, 1, 1, 1}, {1, 16, 1, 1, 1}};
    gert::StorageShape outputShape = {{1, 16, 4, 4, 4}, {1, 16, 4, 4, 4}};
    auto param = gert::TilingData::CreateCap(4096);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_0, &gradsShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("INVALID")}})
                      .TilingData(param.get())
                      .ConstInput(const_tensors)
                      .Build();
    optiling::AvgPool3DGradTilingBase base(holder.GetContext<gert::TilingContext>());
    EXPECT_EQ(base.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

// The aclnn/eager layers feed a mixed-rank pair: a 4D channel-first orig_input_shape
// [C,D,H,W] together with 5D grads in the runtime layout. The base parsing must resolve
// D/H/W independently per tensor instead of sharing a single commInfo.
static void RunMixedRankBaseCheck(const vector<int64_t>& origInput, const gert::StorageShape& gradsShape,
                                  const gert::StorageShape& outputShape, const vector<int64_t>& ksize,
                                  const vector<int64_t>& strides, const vector<int64_t>& pads, bool ceilMode,
                                  const string& dataFormat, ge::graphStatus expectedStatus)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    auto platform_info = InitPlatformInfo(soc_infos, aicore_spec, intrinsics);
    auto compile_info = optiling::AvgPool3DGradCompileInfo();
    std::string op_type("AvgPool3DGrad");

    int32_t shape_data[5] = {0};
    for (size_t i = 0; i < origInput.size(); i++) {
        shape_data[i] = static_cast<int32_t>(origInput[i]);
    }
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(0, DT_INT32, shape_data, static_cast<int64_t>(origInput.size()), const_tensors);
    gert::StorageShape input_0 = {{static_cast<int64_t>(origInput.size())}, {static_cast<int64_t>(origInput.size())}};
    auto param = gert::TilingData::CreateCap(4096);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_0, const_cast<gert::StorageShape*>(&gradsShape)})
                      .OutputShapes({const_cast<gert::StorageShape*>(&outputShape)})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceilMode)},
                                  {"count_include_pad", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"divisor_override", Ops::NN::AnyValue::CreateFrom<int64_t>(-4)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)}})
                      .TilingData(param.get())
                      .ConstInput(const_tensors)
                      .Build();
    optiling::AvgPool3DGradTilingBase base(holder.GetContext<gert::TilingContext>());
    EXPECT_EQ(base.GetShapeAttrsInfo(), expectedStatus);
}

TEST_F(AvgPool3DGradTiling, base_mixed_rank_cdhw_grads_ndhwc)
{
    // Channel-first declared 4D output under NDHWC is no longer accepted (candidate A was
    // removed): only runtime-layout or N*C-merged output declarations pass.
    gert::StorageShape gradsShape = {{1, 3, 9, 9, 1}, {1, 3, 9, 9, 1}};
    gert::StorageShape outputShape = {{1, 5, 15, 9}, {1, 5, 15, 9}};
    RunMixedRankBaseCheck({1, 5, 15, 9}, gradsShape, outputShape, {5, 4, 1}, {2, 2, 1}, {2, 2, 0}, true, "NDHWC",
                          ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradTiling, base_mixed_rank_cdhw_grads_ncdhw)
{
    // EnableBlock feeding: 4D orig [C,D,H,W]=[15,9,16,1], 5D NCDHW grads [1,15,2,17,1].
    gert::StorageShape gradsShape = {{1, 15, 2, 17, 1}, {1, 15, 2, 17, 1}};
    gert::StorageShape outputShape = {{15, 9, 16, 1}, {15, 9, 16, 1}};
    RunMixedRankBaseCheck({15, 9, 16, 1}, gradsShape, outputShape, {5, 4, 1}, {5, 1, 1}, {1, 2, 0}, true, "NCDHW",
                          ge::GRAPH_SUCCESS);
}

TEST_F(AvgPool3DGradTiling, base_mixed_rank_invalid_grad)
{
    // Wrong grads D/H/W must still be rejected by CheckGradValid.
    gert::StorageShape gradsShape = {{1, 7, 9, 9, 1}, {1, 7, 9, 9, 1}};
    gert::StorageShape outputShape = {{1, 5, 15, 9}, {1, 5, 15, 9}};
    RunMixedRankBaseCheck({1, 5, 15, 9}, gradsShape, outputShape, {5, 4, 1}, {2, 2, 1}, {2, 2, 0}, true, "NDHWC",
                          ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradTiling, base_mixed_rank_5d_out_ndhwc)
{
    // Runtime aclnn NDHWC branch: 4D orig [C,D,H,W], 5D NDHWC grads [1,D,H,W,C] and
    // the op output is 5D NDHWC [1,D,H,W,C] as well (transposed back by the caller).
    gert::StorageShape gradsShape = {{1, 3, 9, 9, 1}, {1, 3, 9, 9, 1}};
    gert::StorageShape outputShape = {{1, 5, 15, 9, 1}, {1, 5, 15, 9, 1}};
    RunMixedRankBaseCheck({1, 5, 15, 9}, gradsShape, outputShape, {5, 4, 1}, {2, 2, 1}, {2, 2, 0}, true, "NDHWC",
                          ge::GRAPH_SUCCESS);
}

TEST_F(AvgPool3DGradTiling, base_mixed_rank_5d_out_ncdhw_fmt_ndhwc)
{
    // Channel-first declared 5D output under NDHWC is no longer accepted (candidate A was
    // removed); only runtime-layout or N*C-merged output declarations pass.
    gert::StorageShape gradsShape = {{19, 1, 5, 1, 19}, {19, 1, 5, 1, 19}};
    gert::StorageShape outputShape = {{19, 19, 1, 15, 1}, {19, 19, 1, 15, 1}};
    RunMixedRankBaseCheck({19, 19, 1, 15, 1}, gradsShape, outputShape, {1, 3, 1}, {1, 3, 1}, {0, 0, 1, 1, 0, 0}, false,
                          "NDHWC", ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradTiling, base_mixed_rank_out_inconsistent)
{
    // Output that matches neither the channel-first nor the data_format mapping is rejected.
    gert::StorageShape gradsShape = {{1, 3, 9, 9, 1}, {1, 3, 9, 9, 1}};
    gert::StorageShape outputShape = {{1, 7, 15, 9, 1}, {1, 7, 15, 9, 1}};
    RunMixedRankBaseCheck({1, 5, 15, 9}, gradsShape, outputShape, {5, 4, 1}, {2, 2, 1}, {2, 2, 0}, true, "NDHWC",
                          ge::GRAPH_FAILED);
}

TEST_F(AvgPool3DGradTiling, base_5d_merged_ndhwc_out)
{
    // aclnn merges N*C into the trailing channel for 5D inputs, so the op output may be
    // declared as [1, D, H, W, N*C] while orig_input_shape keeps [N,C,D,H,W].
    gert::StorageShape gradsShape = {{1, 1, 5, 1, 3321}, {1, 1, 5, 1, 3321}};
    gert::StorageShape outputShape = {{1, 1, 17, 1, 3321}, {1, 1, 17, 1, 3321}};
    RunMixedRankBaseCheck({3321, 1, 1, 17, 1}, gradsShape, outputShape, {1, 4, 1}, {1, 4, 1}, {0, 0, 2, 2, 0, 0}, true,
                          "NDHWC", ge::GRAPH_SUCCESS);
}

TEST_F(AvgPool3DGradTiling, base_5d_merged_out_h_mismatch)
{
    // Merged representation with a wrong H dimension is still rejected.
    gert::StorageShape gradsShape = {{1, 1, 5, 1, 3321}, {1, 1, 5, 1, 3321}};
    gert::StorageShape outputShape = {{1, 1, 9, 1, 3321}, {1, 1, 9, 1, 3321}};
    RunMixedRankBaseCheck({3321, 1, 1, 17, 1}, gradsShape, outputShape, {1, 4, 1}, {1, 4, 1}, {0, 0, 2, 2, 0, 0}, true,
                          "NDHWC", ge::GRAPH_FAILED);
}

// ============================ SIMT ============================

TEST_F(AvgPool3DGradTiling, simt_ndhwc_fp32)
{
    vector<int64_t> origInput = {1, 4, 4, 4, 4};
    gert::StorageShape gradsShape = {{1, 1, 1, 1, 4}, {1, 1, 1, 1, 4}};
    gert::StorageShape outputShape = {{1, 4, 4, 4, 4}, {1, 4, 4, 4, 4}};
    RunTiling(origInput, gradsShape, outputShape, {4, 4, 4}, {4, 4, 4}, {0, 0, 0}, false, true, 0, "NDHWC",
              ge::DT_FLOAT, 2322);
}

TEST_F(AvgPool3DGradTiling, simt_ncdhw_fp16)
{
    vector<int64_t> origInput = {2, 8, 8, 8, 8};
    gert::StorageShape gradsShape = {{2, 8, 2, 2, 2}, {2, 8, 2, 2, 2}};
    gert::StorageShape outputShape = {{2, 8, 8, 8, 8}, {2, 8, 8, 8, 8}};
    RunTiling(origInput, gradsShape, outputShape, {4, 4, 4}, {4, 4, 4}, {0, 0, 0}, false, true, 0, "NCDHW",
              ge::DT_FLOAT16, 2306);
}

TEST_F(AvgPool3DGradTiling, simt_cdhw_fp16)
{
    // 4D (CDHW) input on arch35: orig_input_shape = [C,D,H,W].
    vector<int64_t> origInput = {8, 8, 8, 8};
    gert::StorageShape gradsShape = {{8, 4, 4, 4}, {8, 4, 4, 4}};
    gert::StorageShape outputShape = {{8, 8, 8, 8}, {8, 8, 8, 8}};
    RunTiling(origInput, gradsShape, outputShape, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, false, true, 0, "NCDHW",
              ge::DT_FLOAT16, 2304);
}

TEST_F(AvgPool3DGradTiling, simt_overlap_divisor)
{
    // small grads (all-overlapped) to force SIMT fallback; also exercise divisor/pad.
    vector<int64_t> origInput = {1, 4, 4, 4, 4};
    gert::StorageShape gradsShape = {{1, 1, 1, 1, 4}, {1, 1, 1, 1, 4}};
    gert::StorageShape outputShape = {{1, 4, 4, 4, 4}, {1, 4, 4, 4, 4}};
    RunTiling(origInput, gradsShape, outputShape, {4, 4, 4}, {4, 4, 4}, {0, 0, 0}, false, false, 5, "NDHWC",
              ge::DT_FLOAT, 4370);
}
