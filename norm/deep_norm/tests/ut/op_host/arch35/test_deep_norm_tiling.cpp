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
    static void SetUpTestCase() { std::cout << "DeepNormTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DeepNormTilingArch35 TearDown" << std::endl; }
};

namespace {
// 240KB UB makes the documented accept boundary (fp32 numColAlign 10176, fp16/bf16 17472) exact.
constexpr uint64_t UB_SIZE_240K = 245760;

// Builds a StorageShape (storage == origin) from a dim vector.
static gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableStorageShape().AppendDim(d);
        s.MutableOriginShape().AppendDim(d);
    }
    return s;
}

// Runs the DeepNorm arch35 tiling once with fully explicit input/output shapes and returns its
// graphStatus. Used by the negative shape-legality tests that need mismatched shapes.
// On GRAPH_SUCCESS, tilingKey is filled with the produced tiling key.
static ge::graphStatus RunDeepNormArch35TilingShapes(
    const std::vector<int64_t>& xDims, const std::vector<int64_t>& gxDims, const std::vector<int64_t>& betaDims,
    const std::vector<int64_t>& gammaDims, const std::vector<int64_t>& meanDims, const std::vector<int64_t>& rstdDims,
    const std::vector<int64_t>& yDims, ge::DataType dt, uint64_t ubSize, int64_t& tilingKey)
{
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": )" +
                                      std::to_string(ubSize) + R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
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

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::StorageShape x_shape = MakeShape(xDims);
    gert::StorageShape gx_shape = MakeShape(gxDims);
    gert::StorageShape y_shape = MakeShape(yDims);
    gert::StorageShape mean_shape = MakeShape(meanDims);
    gert::StorageShape rstd_shape = MakeShape(rstdDims);
    gert::StorageShape gamma_shape = MakeShape(gammaDims);
    gert::StorageShape beta_shape = MakeShape(betaDims);

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
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(0.3)},
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

// Convenience wrapper building well-formed shapes: x/gx/y = leadingDims + [D];
// gamma/beta = [D]; mean/rstd = leadingDims + [1].
static ge::graphStatus RunDeepNormArch35Tiling(const std::vector<int64_t>& leadingDims, int64_t D, ge::DataType dt,
                                               uint64_t ubSize, int64_t& tilingKey)
{
    std::vector<int64_t> xDims = leadingDims;
    xDims.push_back(D);
    std::vector<int64_t> scalarDims = leadingDims;
    scalarDims.push_back(1);
    std::vector<int64_t> gammaDims = {D};
    return RunDeepNormArch35TilingShapes(xDims, xDims, gammaDims, gammaDims, scalarDims, scalarDims, xDims, dt, ubSize,
                                         tilingKey);
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

// Shape-legality regression (previously missing on arch35): these must all be rejected.

// x dim num out of range (9 > MAX_DIM_X == 8) must fail.
TEST_F(DeepNormTilingArch35, deep_norm_arch35_x_dim_over_range)
{
    int64_t key = -1;
    // 8 leading dims + D == 9-dim x.
    auto status = RunDeepNormArch35Tiling({2, 2, 2, 2, 2, 2, 2, 2}, 8, ge::DT_FLOAT16, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// gx shape not equal x shape must fail (same dim num, differing value).
TEST_F(DeepNormTilingArch35, deep_norm_arch35_gx_shape_mismatch)
{
    int64_t key = -1;
    // x = {4, 128}, gx = {8, 128} (leading dim differs).
    auto status = RunDeepNormArch35TilingShapes({4, 128}, {8, 128}, {128}, {128}, {4, 1}, {4, 1}, {4, 128},
                                                ge::DT_FLOAT16, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// beta dim num not equal gamma dim num must fail.
TEST_F(DeepNormTilingArch35, deep_norm_arch35_beta_dim_mismatch)
{
    int64_t key = -1;
    // gamma = {128} (1-dim), beta = {1, 128} (2-dim).
    auto status = RunDeepNormArch35TilingShapes({4, 128}, {4, 128}, {1, 128}, {128}, {4, 1}, {4, 1}, {4, 128},
                                                ge::DT_FLOAT16, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// zero dim in x (empty tensor) must fail.
TEST_F(DeepNormTilingArch35, deep_norm_arch35_x_zero_dim)
{
    int64_t key = -1;
    // x = {4, 0, 128}; a leading dim is 0.
    auto status = RunDeepNormArch35Tiling({4, 0}, 128, ge::DT_FLOAT16, UB_SIZE_240K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
