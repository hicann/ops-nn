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
 * \file test_deep_norm_tiling.cpp
 * \brief DeepNorm arch35 (Ascend950) tiling UT.
 *
 * Focused on the guards added for arch35:
 *   - non-64-aligned reduce axis D (numColAlign rounding) still produces a valid tiling;
 *   - UB capacity guard rejects D that would overflow UB at kernel InitBuffer
 *     (fp32 numColAlign over ~10176, fp16/bf16 over ~17472 at a 240KB UB);
 *   - numRow/numColAlign uint32 range guard.
 * The UB threshold is driven by the UB_SIZE we feed below (245760 == 240KB), which makes
 * the documented fp32 D=10176 / fp16 D=17472 the accept boundary.
 */

#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../op_host/arch35/deep_norm_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class DeepNormTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DeepNormTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DeepNormTilingArch35 TearDown" << std::endl;
    }
};

namespace {
// 240KB UB makes the documented accept boundary (fp32 numColAlign 10176, fp16/bf16 17472) exact.
constexpr uint64_t UB_SIZE_240K = 245760;

// Runs the DeepNorm arch35 tiling once and returns its graphStatus.
// leadingDims are the rows (N) dims; D is the reduce axis (== gamma length).
// On GRAPH_SUCCESS, tilingKey is filled with the produced tiling key.
static ge::graphStatus RunDeepNormArch35Tiling(
    const std::vector<int64_t>& leadingDims, int64_t D, ge::DataType dt, uint64_t ubSize, int64_t& tilingKey)
{
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": )" + std::to_string(ubSize) + R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 24}
                          })";
    map<string, string> soc_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::DeepNormCompileInfo compile_info;

    std::string op_type("DeepNorm");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = opImpl->tiling;
    auto tiling_parse_func = opImpl->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Build shapes: x/gx/y = leadingDims + [D]; gamma/beta = [D]; mean/rstd = leadingDims + [1].
    std::vector<int64_t> xDims = leadingDims;
    xDims.push_back(D);
    std::vector<int64_t> scalarDims = leadingDims;
    scalarDims.push_back(1);

    gert::StorageShape x_shape;
    gert::StorageShape gx_shape;
    gert::StorageShape y_shape;
    gert::StorageShape mean_shape;
    gert::StorageShape rstd_shape;
    for (auto d : xDims) {
        x_shape.MutableStorageShape().AppendDim(d);
        x_shape.MutableOriginShape().AppendDim(d);
        gx_shape.MutableStorageShape().AppendDim(d);
        gx_shape.MutableOriginShape().AppendDim(d);
        y_shape.MutableStorageShape().AppendDim(d);
        y_shape.MutableOriginShape().AppendDim(d);
    }
    for (auto d : scalarDims) {
        mean_shape.MutableStorageShape().AppendDim(d);
        mean_shape.MutableOriginShape().AppendDim(d);
        rstd_shape.MutableStorageShape().AppendDim(d);
        rstd_shape.MutableOriginShape().AppendDim(d);
    }
    gert::StorageShape gamma_shape = {{D}, {D}};
    gert::StorageShape beta_shape = {{D}, {D}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x_shape, &gx_shape, &beta_shape, &gamma_shape})
                      .OutputShapes({&mean_shape, &rstd_shape, &y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(0.3)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.000001)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto status = tiling_func(tiling_context);
    if (status == ge::GRAPH_SUCCESS) {
        tilingKey = tiling_context->GetTilingKey();
    }
    return status;
}
} // namespace

// (1) non-64-aligned D produces a valid tiling (key 0), covers numColAlign rounding.
TEST_F(DeepNormTilingArch35, deep_norm_arch35_fp16_unaligned_d)
{
    int64_t key = -1;
    auto status = RunDeepNormArch35Tiling({4}, 1000, ge::DT_FLOAT16, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// (4) dtype coverage: fp32 typical D.
TEST_F(DeepNormTilingArch35, deep_norm_arch35_fp32_basic)
{
    int64_t key = -1;
    auto status = RunDeepNormArch35Tiling({8}, 2560, ge::DT_FLOAT, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// (4) dtype coverage: bf16 typical D.
TEST_F(DeepNormTilingArch35, deep_norm_arch35_bf16_basic)
{
    int64_t key = -1;
    auto status = RunDeepNormArch35Tiling({4}, 4096, ge::DT_BF16, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// (2) UB boundary: fp32 D=10176 is the largest accepted at 240KB UB.
TEST_F(DeepNormTilingArch35, deep_norm_arch35_fp32_ub_boundary_ok)
{
    int64_t key = -1;
    auto status = RunDeepNormArch35Tiling({2}, 10176, ge::DT_FLOAT, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// (2) UB over-limit: fp32 large D must be rejected (regression for the UB capacity guard).
TEST_F(DeepNormTilingArch35, deep_norm_arch35_fp32_ub_over_limit)
{
    int64_t key = -1;
    auto status = RunDeepNormArch35Tiling({2}, 11000, ge::DT_FLOAT, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// (2) UB boundary: fp16 D=17472 is the largest accepted at 240KB UB.
TEST_F(DeepNormTilingArch35, deep_norm_arch35_fp16_ub_boundary_ok)
{
    int64_t key = -1;
    auto status = RunDeepNormArch35Tiling({2}, 17472, ge::DT_FLOAT16, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// (2) UB over-limit: fp16/bf16 large D must be rejected.
TEST_F(DeepNormTilingArch35, deep_norm_arch35_fp16_ub_over_limit)
{
    int64_t key = -1;
    auto status = RunDeepNormArch35Tiling({2}, 18000, ge::DT_FLOAT16, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// (3) numRow boundary: product of leading dims overflowing uint32 must be rejected (range guard).
TEST_F(DeepNormTilingArch35, deep_norm_arch35_numrow_over_uint32)
{
    int64_t key = -1;
    // 65536 * 65536 == 2^32 == UINT32_MAX + 1; D small so the UB guard passes first.
    auto status = RunDeepNormArch35Tiling({65536, 65536}, 128, ge::DT_FLOAT16, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
