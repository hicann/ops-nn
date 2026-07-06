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
 * \file test_fake_quant_with_min_max_vars_per_channel_tiling.cpp
 * \brief op_host tiling UT for FakeQuantWithMinMaxVarsPerChannel (arch35 / DAV_3510).
 *
 * 模板参考: quant/ascend_quant_v2/tests/ut/op_host/test_ascend_quant_v2_tiling.cpp
 *           quant/dequant_bias/tests/ut/op_host/test_dequant_bias_tiling.cpp
 *
 * Tiling 实现位于:
 *   op_host/arch35/fake_quant_with_min_max_vars_per_channel_tiling.cpp
 *
 * 覆盖：
 *   - tiling_block_axis_1_small_n : blockAxis=1（切 N 维）多核
 *   - tiling_block_axis_0_large_m : blockAxis=0（切 M 维）多核
 *   - tiling_num_bits_2           : 边界 num_bits=2 (quantMax=3)
 *   - tiling_num_bits_16          : 边界 num_bits=16 (quantMax=65535)
 *   - tiling_narrow_range_true    : narrow_range=true (quantMin=1.0)
 *   - tiling_single_core_tiny     : 单核退化
 *   - tiling_3d_merge_shape       : 3D 输入合轴 (M=6, N=16)
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_kernel/arch35/fake_quant_with_min_max_vars_per_channel_tiling_data.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class FakeQuantWithMinMaxVarsPerChannelTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FakeQuantWithMinMaxVarsPerChannelTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FakeQuantWithMinMaxVarsPerChannelTiling TearDown" << std::endl; }
};

// CompileInfo 占位（本算子 TilingParse 为空，但 facker 需要一个输出 holder）
struct FakeQuantWithMinMaxVarsPerChannelCompileInfo {};

static const char* kCompileInfoStr = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                      "Intrinsic_fix_pipe_l0c2out": false,
                      "Intrinsic_data_move_l12ub": true,
                      "Intrinsic_data_move_l0c2ub": true,
                      "Intrinsic_data_move_out2l1_nd2nz": false,
                      "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                      "CORE_NUM": 48}
                      })";

// =============================================================================
// 测试宏：每个 case 直接展开，保证 holder 作用域贯穿整个 case
// =============================================================================

#define SETUP_TILING_CTX(xShapeVar, mnShapeVar, mxShapeVar, yShapeVar, numBitsVal, narrowRangeVal)                     \
    fe::PlatFormInfos platform_info;                                                                                   \
    map<string, string> soc_infos;                                                                                     \
    map<string, string> aicore_spec;                                                                                   \
    map<string, string> intrinsics;                                                                                    \
    GetPlatFormInfos(kCompileInfoStr, soc_infos, aicore_spec, intrinsics);                                             \
    platform_info.Init();                                                                                              \
                                                                                                                       \
    std::string op_type("FakeQuantWithMinMaxVarsPerChannel");                                                          \
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);                                \
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;                         \
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;             \
                                                                                                                       \
    FakeQuantWithMinMaxVarsPerChannelCompileInfo compile_info;                                                         \
    auto kernel_holder = gert::KernelRunContextFaker()                                                                 \
                             .KernelIONum(2, 1)                                                                        \
                             .Inputs({const_cast<char*>(kCompileInfoStr), reinterpret_cast<void*>(&platform_info)})    \
                             .Outputs({&compile_info})                                                                 \
                             .Build();                                                                                 \
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());                      \
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);     \
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",              \
                                                                                            aicore_spec);              \
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");           \
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", \
                                                                                            intrinsics);               \
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);                  \
                                                                                                                       \
    auto param = gert::TilingData::CreateCap(4096);                                                                    \
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);                                          \
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());                              \
    ASSERT_NE(param, nullptr);                                                                                         \
    auto holder = gert::TilingContextFaker()                                                                           \
                      .NodeIoNum(3, 1)                                                                                 \
                      .IrInstanceNum({1, 1, 1})                                                                        \
                      .InputShapes({&(xShapeVar), &(mnShapeVar), &(mxShapeVar)})                                       \
                      .OutputShapes({&(yShapeVar)})                                                                    \
                      .CompileInfo(&compile_info)                                                                      \
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))                                           \
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                      \
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                      \
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                      \
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                     \
                      .NodeAttrs({                                                                                     \
                          {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(numBitsVal)},                            \
                          {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(narrowRangeVal)},                       \
                      })                                                                                               \
                      .TilingData(param.get())                                                                         \
                      .Workspace(ws_size)                                                                              \
                      .Build();                                                                                        \
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();                                    \
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);                                                             \
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);                                           \
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);                                      \
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");                                                 \
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);                          \
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

// 从 RawTilingData 读出我们关心的字段
static const FakeQuantWithMinMaxVarsPerChannelTilingData* ExtractTilingData(gert::TilingContext* ctx)
{
    auto* raw = ctx->GetRawTilingData();
    EXPECT_NE(raw, nullptr);
    EXPECT_GE(raw->GetDataSize(), sizeof(FakeQuantWithMinMaxVarsPerChannelTilingData));
    return reinterpret_cast<const FakeQuantWithMinMaxVarsPerChannelTilingData*>(raw->GetData());
}

// =============================================================================
// case 1: blockAxis=1 (切 N 维) —— M=4 vs CacheLineNumN=16，N 大占优
// =============================================================================
TEST_F(FakeQuantWithMinMaxVarsPerChannelTiling, tiling_block_axis_1_small_n)
{
    int64_t M = 4, N = 128;
    gert::StorageShape x_shape = {{M, N}, {M, N}};
    gert::StorageShape mn_shape = {{N}, {N}};
    gert::StorageShape mx_shape = {{N}, {N}};
    gert::StorageShape y_shape = {{M, N}, {M, N}};

    SETUP_TILING_CTX(x_shape, mn_shape, mx_shape, y_shape, 8, false);

    const auto* td = ExtractTilingData(tiling_context);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->dim0, M);
    EXPECT_EQ(td->dim1, N);
    EXPECT_EQ(td->headNum, M);
    EXPECT_EQ(td->tailDim, N);
    EXPECT_EQ(td->hasZeroPoint, 0);
    EXPECT_EQ(td->blockUnion, 1);
    EXPECT_EQ(td->numBits, 8);
    EXPECT_EQ(td->narrowRange, 0);
    EXPECT_FLOAT_EQ(td->quantMin, 0.0f);
    EXPECT_FLOAT_EQ(td->quantMax, 255.0f);

    EXPECT_EQ(td->blockAxis, 1); // 切 N 维
    EXPECT_GT(td->numCore, 1);
    EXPECT_GT(td->blockFactor, 0);
    EXPECT_GT(td->blockTailFactor, 0);
    EXPECT_GT(td->baseN, 0);
    EXPECT_GT(td->baseLen, 0);
    EXPECT_EQ(td->baseLen % 64, 0); // 64 对齐
    EXPECT_EQ(static_cast<uint32_t>(td->numCore), tiling_context->GetBlockDim());
}

// case 2: blockAxis=0 (切 M 维) —— M=64 vs CacheLineNumN=16
TEST_F(FakeQuantWithMinMaxVarsPerChannelTiling, tiling_block_axis_0_large_m)
{
    int64_t M = 64, N = 128;
    gert::StorageShape x_shape = {{M, N}, {M, N}};
    gert::StorageShape mn_shape = {{N}, {N}};
    gert::StorageShape mx_shape = {{N}, {N}};
    gert::StorageShape y_shape = {{M, N}, {M, N}};

    SETUP_TILING_CTX(x_shape, mn_shape, mx_shape, y_shape, 8, false);

    const auto* td = ExtractTilingData(tiling_context);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->dim0, M);
    EXPECT_EQ(td->dim1, N);
    EXPECT_EQ(td->blockAxis, 0);
    EXPECT_GT(td->numCore, 1);
    EXPECT_LE(td->numCore, 48);
    EXPECT_GE(td->blockFactor, 1);
    EXPECT_GE(td->blockTailFactor, 1);
    EXPECT_EQ(td->baseLen % 64, 0);
    EXPECT_EQ(td->numBits, 8);
    EXPECT_FLOAT_EQ(td->quantMax, 255.0f);
    EXPECT_EQ(static_cast<uint32_t>(td->numCore), tiling_context->GetBlockDim());

    // blockFactor * (numCore-1) + blockTailFactor == M
    int64_t consumed = td->blockFactor * (td->numCore - 1) + td->blockTailFactor;
    EXPECT_EQ(consumed, M);
}

// case 3: num_bits=2 → quantMax=3
TEST_F(FakeQuantWithMinMaxVarsPerChannelTiling, tiling_num_bits_2)
{
    int64_t M = 8, N = 32;
    gert::StorageShape x_shape = {{M, N}, {M, N}};
    gert::StorageShape mn_shape = {{N}, {N}};
    gert::StorageShape mx_shape = {{N}, {N}};
    gert::StorageShape y_shape = {{M, N}, {M, N}};

    SETUP_TILING_CTX(x_shape, mn_shape, mx_shape, y_shape, 2, false);

    const auto* td = ExtractTilingData(tiling_context);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->numBits, 2);
    EXPECT_EQ(td->narrowRange, 0);
    EXPECT_FLOAT_EQ(td->quantMin, 0.0f);
    EXPECT_FLOAT_EQ(td->quantMax, 3.0f);
    EXPECT_GT(td->numCore, 0);
}

// case 4: num_bits=16 → quantMax=65535
TEST_F(FakeQuantWithMinMaxVarsPerChannelTiling, tiling_num_bits_16)
{
    int64_t M = 8, N = 32;
    gert::StorageShape x_shape = {{M, N}, {M, N}};
    gert::StorageShape mn_shape = {{N}, {N}};
    gert::StorageShape mx_shape = {{N}, {N}};
    gert::StorageShape y_shape = {{M, N}, {M, N}};

    SETUP_TILING_CTX(x_shape, mn_shape, mx_shape, y_shape, 16, false);

    const auto* td = ExtractTilingData(tiling_context);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->numBits, 16);
    EXPECT_FLOAT_EQ(td->quantMax, 65535.0f);
    EXPECT_GT(td->numCore, 0);
}

// case 5: narrow_range=true → quantMin=1.0
TEST_F(FakeQuantWithMinMaxVarsPerChannelTiling, tiling_narrow_range_true)
{
    int64_t M = 16, N = 64;
    gert::StorageShape x_shape = {{M, N}, {M, N}};
    gert::StorageShape mn_shape = {{N}, {N}};
    gert::StorageShape mx_shape = {{N}, {N}};
    gert::StorageShape y_shape = {{M, N}, {M, N}};

    SETUP_TILING_CTX(x_shape, mn_shape, mx_shape, y_shape, 8, true);

    const auto* td = ExtractTilingData(tiling_context);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->numBits, 8);
    EXPECT_EQ(td->narrowRange, 1);
    EXPECT_FLOAT_EQ(td->quantMin, 1.0f);
    EXPECT_FLOAT_EQ(td->quantMax, 255.0f);
}

// case 6: 单核退化 (1, 8)
TEST_F(FakeQuantWithMinMaxVarsPerChannelTiling, tiling_single_core_tiny)
{
    int64_t M = 1, N = 8;
    gert::StorageShape x_shape = {{M, N}, {M, N}};
    gert::StorageShape mn_shape = {{N}, {N}};
    gert::StorageShape mx_shape = {{N}, {N}};
    gert::StorageShape y_shape = {{M, N}, {M, N}};

    SETUP_TILING_CTX(x_shape, mn_shape, mx_shape, y_shape, 8, false);

    const auto* td = ExtractTilingData(tiling_context);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->dim0, M);
    EXPECT_EQ(td->dim1, N);
    EXPECT_EQ(td->numCore, 1);
    EXPECT_EQ(td->blockFactor, td->blockTailFactor);
    EXPECT_GT(td->baseLen, 0);
    EXPECT_EQ(static_cast<uint32_t>(td->numCore), tiling_context->GetBlockDim());
}

// case 7: 3D 输入合轴 (..., d_last) → (M=6, N=16)
TEST_F(FakeQuantWithMinMaxVarsPerChannelTiling, tiling_3d_merge_shape)
{
    gert::StorageShape x_shape = {{2, 3, 16}, {2, 3, 16}};
    gert::StorageShape mn_shape = {{16}, {16}};
    gert::StorageShape mx_shape = {{16}, {16}};
    gert::StorageShape y_shape = {{2, 3, 16}, {2, 3, 16}};

    SETUP_TILING_CTX(x_shape, mn_shape, mx_shape, y_shape, 8, false);

    const auto* td = ExtractTilingData(tiling_context);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->dim0, 6);
    EXPECT_EQ(td->dim1, 16);
    EXPECT_EQ(td->headNum, 6);
    EXPECT_EQ(td->tailDim, 16);
}
