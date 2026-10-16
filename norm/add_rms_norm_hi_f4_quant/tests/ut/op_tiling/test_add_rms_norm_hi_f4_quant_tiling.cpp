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
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "test_add_rms_norm_hi_f4_quant_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../../common/inc/op_host/op_const_def_nn.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AddRmsNormHiF4QuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddRmsNormHiF4QuantTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddRmsNormHiF4QuantTilingTest TearDown" << std::endl; }
};

static void RunTilingFullTest(gert::StorageShape& x1Shape, gert::StorageShape& x2Shape, gert::StorageShape& gammaShape,
                              gert::StorageShape* betaShape, gert::StorageShape& outYShape,
                              gert::StorageShape& outXShape, gert::StorageShape& outScaleShape,
                              gert::StorageShape& outRstdShape, float epsilon, bool outputRstd, ge::DataType x1Dtype,
                              ge::DataType x2Dtype, ge::DataType gammaDtype, ge::DataType yDtype,
                              ge::DataType outXDtype, ge::DataType scaleDtype, ge::DataType rstdDtype,
                              ge::graphStatus expectedTilingResult, uint64_t expectedTilingKey)
{
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", )"
                                 R"("Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, )"
                                 R"("Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, )"
                                 R"("UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, )"
                                 R"("L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, )"
                                 R"("CORE_NUM": 64, "socVersion": "Ascend950"}})";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AddRmsNormHiF4QuantCompileInfo compile_info;

    std::string op_type("AddRmsNormHiF4Quant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(4, 4)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1Shape, &x2Shape, &gammaShape, betaShape})
                      .OutputShapes({&outYShape, &outXShape, &outScaleShape, &outRstdShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, gammaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, gammaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, outXDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, rstdDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)},
                                  {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(outputRstd)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), expectedTilingResult);
    if (expectedTilingResult == ge::GRAPH_SUCCESS && expectedTilingKey != 0) {
        auto tiling_key = tiling_context->GetTilingKey();
        EXPECT_EQ(tiling_key, expectedTilingKey);
        auto tilingData = tiling_context->GetRawTilingData();
        ASSERT_NE(tilingData, nullptr);
    }
}

TEST_F(AddRmsNormHiF4QuantTilingTest, full_load_small_col_key101)
{
    gert::StorageShape x1_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x2_shape = {{8, 64}, {8, 64}};
    gert::StorageShape gamma_shape = {{64}, {64}};

    gert::StorageShape out_y_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_scale_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_SUCCESS, 101);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, full_load_128col_key101)
{
    gert::StorageShape x1_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x2_shape = {{8, 128}, {8, 128}};
    gert::StorageShape gamma_shape = {{128}, {128}};

    gert::StorageShape out_y_shape = {{8, 128}, {8, 128}};
    gert::StorageShape out_x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape out_scale_shape = {{8, 2, 1}, {8, 2, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_SUCCESS, 101);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, full_load_4kcol_key101)
{
    gert::StorageShape x1_shape = {{8, 4096}, {8, 4096}};
    gert::StorageShape x2_shape = {{8, 4096}, {8, 4096}};
    gert::StorageShape gamma_shape = {{4096}, {4096}};

    gert::StorageShape out_y_shape = {{8, 4096}, {8, 4096}};
    gert::StorageShape out_x_shape = {{8, 4096}, {8, 4096}};
    gert::StorageShape out_scale_shape = {{8, 64, 1}, {8, 64, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_SUCCESS, 101);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, split_d_large_col_key201)
{
    gert::StorageShape x1_shape = {{8, 32768}, {8, 32768}};
    gert::StorageShape x2_shape = {{8, 32768}, {8, 32768}};
    gert::StorageShape gamma_shape = {{32768}, {32768}};

    gert::StorageShape out_y_shape = {{8, 32768}, {8, 32768}};
    gert::StorageShape out_x_shape = {{8, 32768}, {8, 32768}};
    gert::StorageShape out_scale_shape = {{8, 512, 1}, {8, 512, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_SUCCESS, 201);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, split_d_very_large_col_key201)
{
    gert::StorageShape x1_shape = {{4, 65536}, {4, 65536}};
    gert::StorageShape x2_shape = {{4, 65536}, {4, 65536}};
    gert::StorageShape gamma_shape = {{65536}, {65536}};

    gert::StorageShape out_y_shape = {{4, 65536}, {4, 65536}};
    gert::StorageShape out_x_shape = {{4, 65536}, {4, 65536}};
    gert::StorageShape out_scale_shape = {{4, 1024, 1}, {4, 1024, 1}};
    gert::StorageShape out_rstd_shape = {{4, 1}, {4, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_SUCCESS, 201);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, no_rstd_output_key101)
{
    gert::StorageShape x1_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x2_shape = {{8, 64}, {8, 64}};
    gert::StorageShape gamma_shape = {{64}, {64}};

    gert::StorageShape out_y_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_scale_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, false, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_SUCCESS, 101);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, x1_x2_shape_not_equal_fail)
{
    gert::StorageShape x1_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x2_shape = {{8, 63}, {8, 63}};
    gert::StorageShape gamma_shape = {{64}, {64}};

    gert::StorageShape out_y_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_scale_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_FAILED, 0);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, gamma_shape_not_1_dim_fail)
{
    gert::StorageShape x1_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x2_shape = {{8, 64}, {8, 64}};
    gert::StorageShape gamma_shape = {{64, 2}, {64, 2}};

    gert::StorageShape out_y_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_scale_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_FAILED, 0);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, gamma_shape_invalid_fail)
{
    gert::StorageShape x1_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x2_shape = {{8, 64}, {8, 64}};
    gert::StorageShape gamma_shape = {{6}, {6}};

    gert::StorageShape out_y_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_scale_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_FAILED, 0);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, x1_x2_dtype_not_same_fail)
{
    gert::StorageShape x1_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x2_shape = {{8, 64}, {8, 64}};
    gert::StorageShape gamma_shape = {{64}, {64}};

    gert::StorageShape out_y_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_scale_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_FAILED, 0);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, y_dtype_invalid_fail)
{
    gert::StorageShape x1_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x2_shape = {{8, 64}, {8, 64}};
    gert::StorageShape gamma_shape = {{64}, {64}};

    gert::StorageShape out_y_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_scale_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_FAILED, 0);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, scale_dtype_invalid_fail)
{
    gert::StorageShape x1_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x2_shape = {{8, 64}, {8, 64}};
    gert::StorageShape gamma_shape = {{64}, {64}};

    gert::StorageShape out_y_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape out_scale_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT, Ops::NN::DT_FLOAT, ge::GRAPH_FAILED,
                      0);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, numCol_not_multiple_of_64_fail)
{
    gert::StorageShape x1_shape = {{8, 96}, {8, 96}};
    gert::StorageShape x2_shape = {{8, 96}, {8, 96}};
    gert::StorageShape gamma_shape = {{96}, {96}};

    gert::StorageShape out_y_shape = {{8, 96}, {8, 96}};
    gert::StorageShape out_x_shape = {{8, 96}, {8, 96}};
    gert::StorageShape out_scale_shape = {{8, 2, 1}, {8, 2, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_FAILED, 0);
}

TEST_F(AddRmsNormHiF4QuantTilingTest, numCol_48_not_multiple_of_64_fail)
{
    gert::StorageShape x1_shape = {{8, 48}, {8, 48}};
    gert::StorageShape x2_shape = {{8, 48}, {8, 48}};
    gert::StorageShape gamma_shape = {{48}, {48}};

    gert::StorageShape out_y_shape = {{8, 48}, {8, 48}};
    gert::StorageShape out_x_shape = {{8, 48}, {8, 48}};
    gert::StorageShape out_scale_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape out_rstd_shape = {{8, 1}, {8, 1}};

    RunTilingFullTest(x1_shape, x2_shape, gamma_shape, nullptr, out_y_shape, out_x_shape, out_scale_shape,
                      out_rstd_shape, 1e-6f, true, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16,
                      Ops::NN::DT_HIFLOAT4, Ops::NN::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_FLOAT,
                      ge::GRAPH_FAILED, 0);
}
