/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_proto.h
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class RepeatInterleaveGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RepeatInterleaveGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RepeatInterleaveGradTiling TearDown" << std::endl;
    }
};

struct RepeatInterleaveGradCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
    int64_t blockSize;
    bool isAscendC;
    uint32_t clSize;
    uint32_t vRegSize;
};

static void RunTilingTestInternal(gert::StorageShape input_shape, gert::StorageShape repeats_shape,
                                  gert::StorageShape out_shape, ge::DataType grad_dtype, ge::DataType index_dtype,
                                  int64_t axis, bool is_ascend_c,
                                  map<string, string> soc_version_infos)
{
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    RepeatInterleaveGradCompileInfo compile_info;
    compile_info.isAscendC = is_ascend_c;
    compile_info.ubSize = 245760;
    compile_info.clSize = 256;
    compile_info.vRegSize = 256;
    compile_info.blockSize = 32;

    std::string op_type("RepeatInterleaveGrad");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &repeats_shape})
                      .OutputShapes({&out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, grad_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, index_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, grad_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

static void RunTilingTest(gert::StorageShape input_shape, gert::StorageShape repeats_shape,
                          gert::StorageShape out_shape, ge::DataType grad_dtype, ge::DataType index_dtype,
                          int64_t axis, bool is_ascend_c)
{
    RunTilingTestInternal(input_shape, repeats_shape, out_shape, grad_dtype, index_dtype, axis, false, {});
}

static void RunTilingTest(gert::StorageShape input_shape, gert::StorageShape repeats_shape,
                          gert::StorageShape out_shape, ge::DataType grad_dtype, ge::DataType index_dtype, int64_t axis)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    RunTilingTestInternal(input_shape, repeats_shape, out_shape, grad_dtype, index_dtype, axis, true, soc_version_infos);
}


TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_fp16_int32_axis1)
{
    gert::StorageShape input_shape = {{2, 32, 16}, {2, 32, 16}};
    gert::StorageShape repeats_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{2, 16, 16}, {2, 16, 16}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT16, ge::DT_FLOAT16, 1, false);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_fp32_int32_axis1)
{
    gert::StorageShape input_shape = {{2, 32, 16}, {2, 32, 16}};
    gert::StorageShape repeats_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{2, 16, 16}, {2, 16, 16}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT, ge::DT_FLOAT, 1);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_fp16_int32_axis0)
{
    gert::StorageShape input_shape = {{4, 8, 16}, {4, 8, 16}};
    gert::StorageShape repeats_shape = {{4}, {4}};
    gert::StorageShape out_shape = {{2, 8, 16}, {2, 8, 16}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT16, ge::DT_FLOAT16, 0);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_fp16_int32_axis_last)
{
    gert::StorageShape input_shape = {{2, 8, 64}, {2, 8, 64}};
    gert::StorageShape repeats_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{2, 8, 32}, {2, 8, 32}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT16, ge::DT_FLOAT16, 2);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_fp16_scalar_repeat)
{
    gert::StorageShape input_shape = {{2, 64, 16}, {2, 64, 16}};
    gert::StorageShape repeats_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{2, 32, 16}, {2, 32, 16}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT16, ge::DT_FLOAT16, 1);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_fp32_scalar_repeat)
{
    gert::StorageShape input_shape = {{2, 64, 16}, {2, 64, 16}};
    gert::StorageShape repeats_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{2, 32, 16}, {2, 32, 16}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT, ge::DT_FLOAT, 1);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_fp32_int32_axis0)
{
    gert::StorageShape input_shape = {{8, 16, 32}, {8, 16, 32}};
    gert::StorageShape repeats_shape = {{8}, {8}};
    gert::StorageShape out_shape = {{4, 16, 32}, {4, 16, 32}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT, ge::DT_FLOAT, 0);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_fp16_1d_input)
{
    gert::StorageShape input_shape = {{64}, {64}};
    gert::StorageShape repeats_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{32}, {32}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT16, ge::DT_FLOAT16, 0);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_fp32_1d_input)
{
    gert::StorageShape input_shape = {{128}, {128}};
    gert::StorageShape repeats_shape = {{4}, {4}};
    gert::StorageShape out_shape = {{32}, {32}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT, ge::DT_FLOAT, 0);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_large_shape)
{
    gert::StorageShape input_shape = {{8, 128, 256}, {8, 128, 256}};
    gert::StorageShape repeats_shape = {{128}, {128}};
    gert::StorageShape out_shape = {{8, 64, 256}, {8, 64, 256}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT, ge::DT_FLOAT, 1);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_small_batch)
{
    gert::StorageShape input_shape = {{1, 16, 8}, {1, 16, 8}};
    gert::StorageShape repeats_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{1, 8, 8}, {1, 8, 8}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT16, ge::DT_FLOAT16, 1);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_negative_axis)
{
    gert::StorageShape input_shape = {{2, 32, 16}, {2, 32, 16}};
    gert::StorageShape repeats_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{2, 16, 16}, {2, 16, 16}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT16, ge::DT_FLOAT16, -2);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_4d_input)
{
    gert::StorageShape input_shape = {{2, 4, 16, 8}, {2, 4, 16, 8}};
    gert::StorageShape repeats_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{2, 4, 8, 8}, {2, 4, 8, 8}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT16, ge::DT_FLOAT16, 2);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_high_core_num)
{
    gert::StorageShape input_shape = {{16, 128, 64}, {16, 128, 64}};
    gert::StorageShape repeats_shape = {{128}, {128}};
    gert::StorageShape out_shape = {{16, 64, 64}, {16, 64, 64}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT, ge::DT_FLOAT, 1, 128);
}

TEST_F(RepeatInterleaveGradTiling, repeat_interleave_grad_tiling_low_core_num)
{
    gert::StorageShape input_shape = {{2, 16, 8}, {2, 16, 8}};
    gert::StorageShape repeats_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{2, 8, 8}, {2, 8, 8}};
    RunTilingTest(input_shape, repeats_shape, out_shape, ge::DT_FLOAT, ge::DT_FLOAT, 1, 4);
}