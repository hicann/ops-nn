/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class DeepNormGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DeepNormGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DeepNormGradTiling TearDown" << std::endl;
    }
};
struct DeepNormGradCompileInfo {
};

inline void runTest(
    gert::StorageShape& input_shape, gert::StorageShape& input1_shape, gert::StorageShape& input2_shape,
    gert::StorageShape& input3_shape, gert::StorageShape& input4_shape, gert::StorageShape& input5_shape,
    gert::StorageShape& out_shape, gert::StorageShape& out1_shape, gert::StorageShape& out2_shape,
    gert::StorageShape& out3_shape, ge::DataType data_type, uint32_t tiling_key_expect)
{
    //dlog_setlevel(0, 0, 0);

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct DeepNormCompileInfo {
    };
    DeepNormGradCompileInfo compile_info;

    std::string op_type("DeepNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(6, 4)
            .IrInstanceNum({6})
            .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape, &input5_shape})
            .OutputShapes({&out_shape, &out1_shape, &out2_shape, &out3_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(0.3)}})
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, tiling_key_expect);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_0)
{ // tiling key 0
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(
        input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
        out2_shape, out3_shape, ge::DT_FLOAT, 0);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_1)
{ // tiling key 1
    gert::StorageShape input_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input2_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(
        input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
        out2_shape, out3_shape, ge::DT_FLOAT, 1);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_2)
{ // tiling key 2
    gert::StorageShape input_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape input1_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape input2_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape input3_shape = {{9}, {9}};
    gert::StorageShape input4_shape = {{1, 10, 1}, {1, 10, 1}};
    gert::StorageShape input5_shape = {{1, 10, 1}, {1, 10, 1}};
    gert::StorageShape out_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape out1_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape out2_shape = {{9}, {9}};
    gert::StorageShape out3_shape = {{9}, {9}};

    runTest(
        input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
        out2_shape, out3_shape, ge::DT_FLOAT, 2);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_10)
{ // tiling key 10
    gert::StorageShape input_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input1_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input2_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape out1_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(
        input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
        out2_shape, out3_shape, ge::DT_FLOAT16, 10);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_11)
{ // tiling key 11
    gert::StorageShape input_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input2_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(
        input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
        out2_shape, out3_shape, ge::DT_FLOAT16, 11);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_12)
{ // tiling key 12
    gert::StorageShape input_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input1_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input2_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input3_shape = {{133}, {133}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape out1_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape out2_shape = {{133}, {133}};
    gert::StorageShape out3_shape = {{133}, {133}};

    runTest(
        input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
        out2_shape, out3_shape, ge::DT_FLOAT16, 12);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_20)
{ // tiling key 20
    gert::StorageShape input_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input1_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input2_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape out1_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(
        input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
        out2_shape, out3_shape, ge::DT_BF16, 20);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_21)
{ // tiling key 21
    gert::StorageShape input_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input2_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(
        input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
        out2_shape, out3_shape, ge::DT_BF16, 21);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_22)
{ // tiling key 22
    gert::StorageShape input_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input1_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input2_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input3_shape = {{133}, {133}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape out1_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape out2_shape = {{133}, {133}};
    gert::StorageShape out3_shape = {{133}, {133}};

    runTest(
        input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
        out2_shape, out3_shape, ge::DT_BF16, 22);
}