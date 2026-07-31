/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/grouped_dynamic_mx_quant_with_dual_axis_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;

class GroupedDynamicMxQuantWithDualAxisTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupedDynamicMxQuantWithDualAxisTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupedDynamicMxQuantWithDualAxisTilingTest TearDown" << std::endl; }
};

static void ExecuteTestCase(ge::DataType xDtype, ge::DataType y1Dtype, ge::DataType y1ScaleDtype, ge::DataType y2Dtype,
                            ge::DataType y2ScaleDtype, gert::StorageShape xShape, gert::StorageShape groupIndexShape,
                            gert::StorageShape y1Shape, gert::StorageShape y1ScaleShape, gert::StorageShape y2Shape,
                            gert::StorageShape y2ScaleShape, const string& roundMode, int64_t scaleAlg,
                            int64_t dstDtype, float maxDtypeValue, ge::graphStatus status = ge::GRAPH_SUCCESS,
                            ge::DataType groupIndexDtype = ge::DT_INT64)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::GroupedDynamicMxQuantWithDualAxisCompileInfo compile_info;

    std::string op_type("GroupedDynamicMxQuantWithDualAxis");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 4)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &groupIndexShape})
                      .OutputShapes({&y1Shape, &y1ScaleShape, &y2Shape, &y2ScaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, groupIndexDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, y1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, y1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, y2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, y2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(scaleAlg)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dstDtype)},
                                  {"max_dtype_value", Ops::NN::AnyValue::CreateFrom<float>(maxDtypeValue)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), status);
}

// ======================= Positive tests =======================

// kernel branch: half (FP16) input type, single group, fp8_e4m3fn output
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_fp16_to_fp8_e4m3fn)
{
    gert::StorageShape xShape = {{8, 8192}, {8, 8192}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape y1ScaleShape = {{8, 128, 2}, {8, 128, 2}};
    gert::StorageShape y2Shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape y2ScaleShape = {{1, 8192, 2}, {1, 8192, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// kernel branch: bfloat16 input type, single group, fp8_e4m3fn output
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_bf16_to_fp8_e4m3fn)
{
    gert::StorageShape xShape = {{4, 2048}, {4, 2048}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape y1ScaleShape = {{4, 32, 2}, {4, 32, 2}};
    gert::StorageShape y2Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape y2ScaleShape = {{1, 2048, 2}, {1, 2048, 2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// kernel branch: dstDtype=FP8_E5M2 path in GetInvDstMaxValue
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_fp16_to_fp8_e5m2)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 35, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// kernel branch: multi-group (groupNum > 1), scale2 shape with extra offset
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_bf16_to_fp8_e4m3fn_multi_group)
{
    gert::StorageShape xShape = {{256, 1024}, {256, 1024}};
    gert::StorageShape groupIndexShape = {{4}, {4}};
    gert::StorageShape y1Shape = {{256, 1024}, {256, 1024}};
    gert::StorageShape y1ScaleShape = {{256, 16, 2}, {256, 16, 2}};
    gert::StorageShape y2Shape = {{256, 1024}, {256, 1024}};
    gert::StorageShape y2ScaleShape = {{8, 1024, 2}, {8, 1024, 2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// kernel branch: partial last row block (calcRow < rowBlockSize), small shape
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_partial_row_block)
{
    gert::StorageShape xShape = {{1, 64}, {1, 64}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{1, 64}, {1, 64}};
    gert::StorageShape y1ScaleShape = {{1, 1, 2}, {1, 1, 2}};
    gert::StorageShape y2Shape = {{1, 64}, {1, 64}};
    gert::StorageShape y2ScaleShape = {{1, 64, 2}, {1, 64, 2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// kernel branch: multiple column blocks (n > colBlockSize=256)
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_multi_col_block)
{
    gert::StorageShape xShape = {{2, 512}, {2, 512}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{2, 512}, {2, 512}};
    gert::StorageShape y1ScaleShape = {{2, 8, 2}, {2, 8, 2}};
    gert::StorageShape y2Shape = {{2, 512}, {2, 512}};
    gert::StorageShape y2ScaleShape = {{1, 512, 2}, {1, 512, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// M non-uniform groups (3 groups with sizes 32, 64, 64) covering partial + full row blocks
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_non_uniform_groups)
{
    gert::StorageShape xShape = {{160, 1024}, {160, 1024}};
    gert::StorageShape groupIndexShape = {{3}, {3}};
    gert::StorageShape y1Shape = {{160, 1024}, {160, 1024}};
    gert::StorageShape y1ScaleShape = {{160, 16, 2}, {160, 16, 2}};
    gert::StorageShape y2Shape = {{160, 1024}, {160, 1024}};
    gert::StorageShape y2ScaleShape = {{5, 1024, 2}, {5, 1024, 2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// N=640 = 2*256+128: partial last col block (colBlocks=3, last block 128 cols)
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_partial_last_col_block)
{
    gert::StorageShape xShape = {{128, 640}, {128, 640}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{128, 640}, {128, 640}};
    gert::StorageShape y1ScaleShape = {{128, 10, 2}, {128, 10, 2}};
    gert::StorageShape y2Shape = {{128, 640}, {128, 640}};
    gert::StorageShape y2ScaleShape = {{3, 640, 2}, {3, 640, 2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// Large M single group: 8 full row blocks (512/64=8), single col block (N=256)
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_large_m_single_group)
{
    gert::StorageShape xShape = {{512, 256}, {512, 256}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{512, 256}, {512, 256}};
    gert::StorageShape y1ScaleShape = {{512, 4, 2}, {512, 4, 2}};
    gert::StorageShape y2Shape = {{512, 256}, {512, 256}};
    gert::StorageShape y2ScaleShape = {{9, 256, 2}, {9, 256, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// M exactly fits multiple full row blocks (128=2*64), N=128 (64-aligned, < colBlockSize)
TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_full_row_blocks_small_n)
{
    gert::StorageShape xShape = {{128, 128}, {128, 128}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{128, 128}, {128, 128}};
    gert::StorageShape y1ScaleShape = {{128, 2, 2}, {128, 2, 2}};
    gert::StorageShape y2Shape = {{128, 128}, {128, 128}};
    gert::StorageShape y2ScaleShape = {{3, 128, 2}, {3, 128, 2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

// ======================= Negative: input shape validation =======================

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_x_dim)
{
    gert::StorageShape xShape = {{4, 4096, 2}, {4, 4096, 2}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_group_index_dim)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{2, 2}, {2, 2}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_x_not_64_aligned)
{
    gert::StorageShape xShape = {{4, 100}, {4, 100}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 100}, {4, 100}};
    gert::StorageShape y1ScaleShape = {{4, 1, 2}, {4, 1, 2}};
    gert::StorageShape y2Shape = {{4, 100}, {4, 100}};
    gert::StorageShape y2ScaleShape = {{1, 100, 2}, {1, 100, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_zero_m)
{
    gert::StorageShape xShape = {{0, 64}, {0, 64}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{0, 64}, {0, 64}};
    gert::StorageShape y1ScaleShape = {{0, 1, 2}, {0, 1, 2}};
    gert::StorageShape y2Shape = {{0, 64}, {0, 64}};
    gert::StorageShape y2ScaleShape = {{1, 64, 2}, {1, 64, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_SUCCESS);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_group_index_empty)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{0}, {0}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

// ======================= Negative: dtype validation =======================

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_x_dtype)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_INT32, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_group_index_dtype)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED, ge::DT_INT32);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_y1_dtype)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape,
                    groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_scale1_dtype)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_INT32, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_y2_dtype_mismatch_y1)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    // y1=E4M3FN, y2=E5M2: both valid individually but y1 != y2 → CheckDtype fails
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_scale2_dtype)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    // scale2=FP32 while scale1=E8M0 → CheckDtype fails (scale2 != E8M0)
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_INT32,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

// ======================= Negative: attr validation =======================

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_round_mode)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "floor", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_scale_alg)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 0, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_dst_dtype)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 99, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_invalid_max_dtype_value)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 1.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_dst_dtype_mismatch_y_outputs)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    // dstDtype=35 (FP8_E5M2) but both y output dtypes are DT_FLOAT8_E4M3FN (36).
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 35, 0.0f,
                    ge::GRAPH_FAILED);
}

// ======================= Negative: output shape validation =======================

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_y1_shape_mismatch)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{2, 4096}, {2, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_scale1_shape_mismatch)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 32, 2}, {4, 32, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicMxQuantWithDualAxisTilingTest, test_tiling_scale2_shape_mismatch)
{
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    gert::StorageShape y1Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1ScaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape y2Shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y2ScaleShape = {{2, 4096, 2}, {2, 4096, 2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                    xShape, groupIndexShape, y1Shape, y1ScaleShape, y2Shape, y2ScaleShape, "rint", 1, 36, 0.0f,
                    ge::GRAPH_FAILED);
}
