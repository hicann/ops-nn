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
 * \file test_turbo_quant_compress_latent_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "experimental/quant/turbo_quant_compress_latent/op_host/turbo_quant_compress_latent_tiling.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;

namespace {
constexpr size_t TILING_FIELD_NUM = 6;
constexpr size_t IDX_NUM_TOKENS = 0;
constexpr size_t IDX_TOKENS_PER_CORE = 1;
constexpr size_t IDX_HEAD_DIM = 2;
constexpr size_t IDX_SLOT_SIZE = 3;
constexpr size_t IDX_TOKENS_PER_BATCH = 4;
constexpr size_t IDX_OUTPUT_MODE = 5;
} // namespace

class TurboQuantCompressLatentTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TurboQuantCompressLatentTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TurboQuantCompressLatentTiling TearDown" << std::endl; }
};

struct TilingResult {
    ge::graphStatus status;
    uint32_t blockDim;
    uint32_t fields[TILING_FIELD_NUM];
};

static TilingResult RunTiling(int64_t numTokens, int64_t headDim, int64_t centCount, int64_t outputMode = 0)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_versions = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::TurboQuantCompressLatentCompileInfo compile_info;

    std::string op_type("TurboQuantCompressLatent");
    EXPECT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    EXPECT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_versions);
    EXPECT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape latentShape = {{numTokens, headDim}, {numTokens, headDim}};
    gert::StorageShape centShape = {{centCount}, {centCount}};
    gert::StorageShape slotShape = {{numTokens, optiling::TqCompressOutputSlotSize(headDim, outputMode)},
                                    {numTokens, optiling::TqCompressOutputSlotSize(headDim, outputMode)}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    EXPECT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&latentShape, &centShape})
                      .OutputShapes({&slotShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(outputMode)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    TilingResult result{};
    result.status = tiling_func(tiling_context);
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }
    result.blockDim = tiling_context->GetBlockDim();
    auto raw = tiling_context->GetRawTilingData();
    EXPECT_GE(raw->GetDataSize(), TILING_FIELD_NUM * sizeof(uint32_t));
    const uint32_t* data = reinterpret_cast<const uint32_t*>(raw->GetData());
    for (size_t i = 0; i < TILING_FIELD_NUM; ++i) {
        result.fields[i] = data[i];
    }
    return result;
}

// Every token owns one core slot; the split must cover all tokens without over-allocating cores.
static void ExpectValidSplit(const TilingResult& r, int64_t numTokens)
{
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.fields[IDX_NUM_TOKENS], static_cast<uint32_t>(numTokens));
    EXPECT_EQ(r.fields[IDX_HEAD_DIM], 512U);
    EXPECT_EQ(r.fields[IDX_SLOT_SIZE], 320U);
    EXPECT_EQ(r.fields[IDX_OUTPUT_MODE], 0U);
    EXPECT_GE(r.fields[IDX_TOKENS_PER_CORE], 1U);
    EXPECT_GE(r.blockDim, 1U);
    // the last core must not be fully empty, and together the cores must cover every token
    EXPECT_GE(static_cast<uint64_t>(r.blockDim) * r.fields[IDX_TOKENS_PER_CORE], static_cast<uint64_t>(numTokens));
    EXPECT_LT(static_cast<uint64_t>(r.blockDim - 1) * r.fields[IDX_TOKENS_PER_CORE],
              static_cast<uint64_t>(numTokens < 1 ? 1 : numTokens));
    // batching never exceeds what a core owns (that would idle cores) nor the UB-derived cap
    EXPECT_GE(r.fields[IDX_TOKENS_PER_BATCH], 1U);
    EXPECT_LE(r.fields[IDX_TOKENS_PER_BATCH], static_cast<uint32_t>(optiling::TQ_COMPRESS_MAX_TOKENS_PER_BATCH));
    EXPECT_LE(r.fields[IDX_TOKENS_PER_BATCH], r.fields[IDX_TOKENS_PER_CORE]);
}

TEST_F(TurboQuantCompressLatentTiling, single_token)
{
    TilingResult r = RunTiling(1, 512, 16);
    ExpectValidSplit(r, 1);
    // decode-shaped work must not batch, otherwise most cores would sit idle
    EXPECT_EQ(r.fields[IDX_TOKENS_PER_BATCH], 1U);
}

TEST_F(TurboQuantCompressLatentTiling, fewer_tokens_than_cores)
{
    TilingResult r = RunTiling(7, 512, 16);
    ExpectValidSplit(r, 7);
    EXPECT_EQ(r.fields[IDX_TOKENS_PER_BATCH], 1U);
}

// prefill-shaped work: every core owns far more than the cap, so the batch saturates it
TEST_F(TurboQuantCompressLatentTiling, prefill_saturates_batch)
{
    TilingResult r = RunTiling(2048, 512, 16);
    ExpectValidSplit(r, 2048);
    EXPECT_EQ(r.fields[IDX_TOKENS_PER_BATCH], static_cast<uint32_t>(optiling::TQ_COMPRESS_MAX_TOKENS_PER_BATCH));
}

// in between, the batch tracks what a core actually owns
TEST_F(TurboQuantCompressLatentTiling, batch_tracks_tokens_per_core)
{
    TilingResult r = RunTiling(128, 512, 16);
    ExpectValidSplit(r, 128);
    EXPECT_EQ(r.fields[IDX_TOKENS_PER_BATCH], r.fields[IDX_TOKENS_PER_CORE]);
}

TEST_F(TurboQuantCompressLatentTiling, ragged_token_count) { ExpectValidSplit(RunTiling(4097, 512, 16), 4097); }

TEST_F(TurboQuantCompressLatentTiling, large_batch) { ExpectValidSplit(RunTiling(65536, 512, 16), 65536); }

TEST_F(TurboQuantCompressLatentTiling, zero_tokens)
{
    TilingResult r = RunTiling(0, 512, 16);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.fields[IDX_NUM_TOKENS], 0U);
    EXPECT_GE(r.blockDim, 1U);
}

// headDim other than 512 is deliberately rejected until it has been validated on hardware.
TEST_F(TurboQuantCompressLatentTiling, unsupported_head_dim)
{
    EXPECT_EQ(RunTiling(16, 256, 16).status, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantCompressLatentTiling, wrong_centroid_count)
{
    EXPECT_EQ(RunTiling(16, 512, 8).status, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantCompressLatentTiling, compact_corrected)
{
    TilingResult r = RunTiling(128, 512, 16, 1);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.fields[IDX_SLOT_SIZE], 258U);
    EXPECT_EQ(r.fields[IDX_OUTPUT_MODE], 1U);
}

TEST_F(TurboQuantCompressLatentTiling, invalid_output_mode)
{
    EXPECT_EQ(RunTiling(16, 512, 16, 2).status, ge::GRAPH_FAILED);
}
