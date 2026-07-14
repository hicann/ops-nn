/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"

#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "../../../../op_host/arch35/hard_swish_grad_v2_tiling_arch35.h"
#include "../../../../op_kernel/arch35/hard_swish_grad_v2_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024; // tiling switches to double buffer when totalNum > this
}

class HardSwishGradV2TilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HardSwishGradV2TilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "HardSwishGradV2TilingArch35 TearDown" << std::endl; }
};

// Drives one arch35 tiling invocation. dtype is applied to both inputs and the output.
// On success, the raw tiling data is reinterpreted as HardSwishGradV2Arch35TilingData and returned via outTiling.
static ge::graphStatus RunArch35Tiling(gert::StorageShape& gradShape, gert::StorageShape& selfShape,
                                       gert::StorageShape& outShape, ge::DataType dtype,
                                       HardSwishGradV2Arch35TilingData& outTiling)
{
    std::string op_type("HardSwishGradV2");
    EXPECT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 1048576,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 24
                                        }
                                    })";
    map<string, string> soc_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::HardSwishGradV2CompileInfo compile_info;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    EXPECT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    EXPECT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    EXPECT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradShape, &selfShape})
                      .OutputShapes({&outShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto status = tiling_func(tiling_context);
    if (status == ge::GRAPH_SUCCESS) {
        auto raw = tiling_context->GetRawTilingData();
        if (raw != nullptr && raw->GetDataSize() >= sizeof(HardSwishGradV2Arch35TilingData)) {
            outTiling = *reinterpret_cast<const HardSwishGradV2Arch35TilingData*>(raw->GetData());
        }
    }
    return status;
}

// fp32, totalNum <= 1024 -> single buffer branch
TEST_F(HardSwishGradV2TilingArch35, test_tiling_fp32_single_buffer)
{
    gert::StorageShape shape = {{8, 8}, {8, 8}}; // 64 elements
    HardSwishGradV2Arch35TilingData tiling{};
    ASSERT_EQ(RunArch35Tiling(shape, shape, shape, ge::DT_FLOAT, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalNum, 64);
    EXPECT_LE(tiling.totalNum, MIN_SPLIT_THRESHOLD);
    EXPECT_GT(tiling.blockFactor, 0);
    EXPECT_GT(tiling.ubFactor, 0);
}

// fp16, totalNum <= 1024 -> single buffer branch (cast path dtype)
TEST_F(HardSwishGradV2TilingArch35, test_tiling_fp16_single_buffer)
{
    gert::StorageShape shape = {{256}, {256}};
    HardSwishGradV2Arch35TilingData tiling{};
    ASSERT_EQ(RunArch35Tiling(shape, shape, shape, ge::DT_FLOAT16, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalNum, 256);
    EXPECT_GT(tiling.ubFactor, 0);
}

// fp32, totalNum > 1024 -> double buffer branch
TEST_F(HardSwishGradV2TilingArch35, test_tiling_fp32_double_buffer)
{
    gert::StorageShape shape = {{64, 32}, {64, 32}}; // 2048 elements
    HardSwishGradV2Arch35TilingData tiling{};
    ASSERT_EQ(RunArch35Tiling(shape, shape, shape, ge::DT_FLOAT, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalNum, 2048);
    EXPECT_GT(tiling.totalNum, MIN_SPLIT_THRESHOLD);
    EXPECT_GT(tiling.ubFactor, 0);
}

// fp16, totalNum > 1024 -> double buffer branch
TEST_F(HardSwishGradV2TilingArch35, test_tiling_fp16_double_buffer)
{
    gert::StorageShape shape = {{64, 32}, {64, 32}}; // 2048 elements
    HardSwishGradV2Arch35TilingData tiling{};
    ASSERT_EQ(RunArch35Tiling(shape, shape, shape, ge::DT_FLOAT16, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalNum, 2048);
    EXPECT_GT(tiling.ubFactor, 0);
}

// bf16, totalNum > 1024 -> double buffer branch
TEST_F(HardSwishGradV2TilingArch35, test_tiling_bf16_double_buffer)
{
    gert::StorageShape shape = {{64, 32}, {64, 32}}; // 2048 elements
    HardSwishGradV2Arch35TilingData tiling{};
    ASSERT_EQ(RunArch35Tiling(shape, shape, shape, ge::DT_BF16, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalNum, 2048);
    EXPECT_GT(tiling.ubFactor, 0);
}

// Single buffer (fp32) packs more elements per UB iteration than double buffer for the same dtype.
// This deterministically distinguishes the two branches without hardcoding the UB-derived value.
TEST_F(HardSwishGradV2TilingArch35, test_tiling_fp32_single_gt_double_ubfactor)
{
    gert::StorageShape smallShape = {{64}, {64}};         // single buffer
    gert::StorageShape largeShape = {{64, 32}, {64, 32}}; // double buffer
    HardSwishGradV2Arch35TilingData single{};
    HardSwishGradV2Arch35TilingData dbl{};
    ASSERT_EQ(RunArch35Tiling(smallShape, smallShape, smallShape, ge::DT_FLOAT, single), ge::GRAPH_SUCCESS);
    ASSERT_EQ(RunArch35Tiling(largeShape, largeShape, largeShape, ge::DT_FLOAT, dbl), ge::GRAPH_SUCCESS);
    EXPECT_GT(single.ubFactor, dbl.ubFactor);
}

// Large multi-core case: blockFactor splits totalNum across cores.
TEST_F(HardSwishGradV2TilingArch35, test_tiling_fp32_multi_core)
{
    gert::StorageShape shape = {{1024, 128}, {1024, 128}}; // 131072 elements
    HardSwishGradV2Arch35TilingData tiling{};
    ASSERT_EQ(RunArch35Tiling(shape, shape, shape, ge::DT_FLOAT, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalNum, 131072);
    EXPECT_GT(tiling.blockFactor, 0);
    EXPECT_LE(tiling.blockFactor, tiling.totalNum);
    EXPECT_GT(tiling.ubFactor, 0);
}

// Empty tensor: arch35 tiling sets all factors to 0 and returns success (ubFactor=0 path).
TEST_F(HardSwishGradV2TilingArch35, test_tiling_empty_tensor)
{
    gert::StorageShape shape = {{2, 0, 4}, {2, 0, 4}}; // 0 elements
    HardSwishGradV2Arch35TilingData tiling{};
    ASSERT_EQ(RunArch35Tiling(shape, shape, shape, ge::DT_FLOAT, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalNum, 0);
    EXPECT_EQ(tiling.blockFactor, 0);
    EXPECT_EQ(tiling.ubFactor, 0);
}

// Unsupported dtype is rejected by tiling dtype validation.
TEST_F(HardSwishGradV2TilingArch35, test_tiling_unsupported_dtype)
{
    gert::StorageShape shape = {{8, 8}, {8, 8}};
    HardSwishGradV2Arch35TilingData tiling{};
    EXPECT_EQ(RunArch35Tiling(shape, shape, shape, ge::DT_INT32, tiling), ge::GRAPH_FAILED);
}

// grad_output and self shape mismatch is rejected.
TEST_F(HardSwishGradV2TilingArch35, test_tiling_shape_mismatch)
{
    gert::StorageShape gradShape = {{8, 8}, {8, 8}}; // 64
    gert::StorageShape selfShape = {{8, 4}, {8, 4}}; // 32
    HardSwishGradV2Arch35TilingData tiling{};
    EXPECT_EQ(RunArch35Tiling(gradShape, selfShape, gradShape, ge::DT_FLOAT, tiling), ge::GRAPH_FAILED);
}

// dim num greater than 8 is rejected by tiling shape validation.
TEST_F(HardSwishGradV2TilingArch35, test_tiling_dim_num_over_8)
{
    gert::StorageShape shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1, 1}}; // 9 dims
    HardSwishGradV2Arch35TilingData tiling{};
    EXPECT_EQ(RunArch35Tiling(shape, shape, shape, ge::DT_FLOAT, tiling), ge::GRAPH_FAILED);
}
