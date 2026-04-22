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
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_common.h"
#include "../../../op_host/op_tiling/rotate_quant_tiling.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"

using namespace std;

struct RotateQuantData {
    // inputs
    gert::StorageShape x_shape{{4, 64}, {4, 64}};
    gert::StorageShape rot_shape{{64, 64}, {64, 64}};

    // outputs
    gert::StorageShape y_shape{{4, 64}, {4, 64}};
    gert::StorageShape scale_shape{{4}, {4}};

    // data type
    ge::DataType xDataType{ge::DT_BF16};
    ge::DataType yDataType{ge::DT_INT8};

    // test debug info
    string debug_info{"tiling_info:"};

    // expect
    ge::graphStatus expect_status{ge::GRAPH_SUCCESS};
    uint64_t expect_tiling_key{0};
    // others
    bool check_tiling_key{true};
};

class TilingRotateQuant : public ::testing::TestWithParam<RotateQuantData> {
protected:
    void SetUp() override
    {
        std::cout << "TilingRotateQuant SetUp" << std::endl;
    }

    void TearDown() override
    {
        std::cout << "TilingRotateQuant TearDown" << std::endl;
    }
};

TEST_P(TilingRotateQuant, rotate_quant_tiling)
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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::RotateQuantCompileInfo compile_info;

    string op_type("RotateQuant");
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

    auto test_params = GetParam();
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder =
        gert::TilingContextFaker()
            .SetOpType("RotateQuant")
            .NodeIoNum(2, 2)
            .IrInstanceNum({1, 1})
            .InputShapes({&test_params.x_shape, &test_params.rot_shape})
            .OutputShapes({&test_params.y_shape, &test_params.scale_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, test_params.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, test_params.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, test_params.yDataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // check tiling result
    ge::graphStatus actual_staus = tiling_func(tiling_context);
    EXPECT_EQ(actual_staus, test_params.expect_status) << test_params.debug_info;

    if (test_params.check_tiling_key) {
        auto actual_tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(actual_tiling_key, test_params.expect_tiling_key) << test_params.debug_info;
    }
}

const auto RotateQuantTestCases = ::testing::Values(
    // 0: BF16 + INT8, small shape
    RotateQuantData{
        {{4, 256}, {4, 256}},
        {{64, 64}, {64, 64}},
        {{4, 256}, {4, 256}},
        {{4}, {4}},
        ge::DT_BF16,
        ge::DT_INT8,
        "bf16_int8_4_64",
        ge::GRAPH_SUCCESS,
        0,  // BF16 tiling key
        true
    },
    // 1: FP16 + INT8
    RotateQuantData{
        {{4, 256}, {4, 256}},
        {{64, 64}, {64, 64}},
        {{4, 256}, {4, 256}},
        {{4}, {4}},
        ge::DT_FLOAT16,
        ge::DT_INT8,
        "fp16_int8_4_64",
        ge::GRAPH_SUCCESS,
        0,  // FP16 tiling key
        true
    },
    // 2: BF16 + INT8, larger shape
    RotateQuantData{
        {{16, 256}, {16, 256}},
        {{256, 256}, {256, 256}},
        {{16, 256}, {16, 256}},
        {{16}, {16}},
        ge::DT_BF16,
        ge::DT_INT8,
        "bf16_int8_16_256",
        ge::GRAPH_SUCCESS,
        0,
        true
    },
    // 3: BF16 + INT8, K=1024
    RotateQuantData{
        {{8, 1024}, {8, 1024}},
        {{1024, 1024}, {1024, 1024}},
        {{8, 1024}, {8, 1024}},
        {{8}, {8}},
        ge::DT_BF16,
        ge::DT_INT8,
        "bf16_int8_8_1024",
        ge::GRAPH_SUCCESS,
        0,
        true
    },
    // 4: BF16 + INT8, K=1024
    RotateQuantData{
        {{8, 1024}, {8, 1024}},
        {{1024, 1024}, {1024, 1024}},
        {{8, 1024}, {8, 1024}},
        {{8}, {8}},
        ge::DT_BF16,
        ge::DT_INT8,
        "bf16_int8_8_1024",
        ge::GRAPH_SUCCESS,
        0,
        true
    },
    // 5: BF16 + INT8, K=16
    RotateQuantData{
        {{32, 256}, {32, 256}},
        {{16, 16}, {16, 16}},
        {{32, 256}, {32, 256}},
        {{32}, {32}},
        ge::DT_BF16,
        ge::DT_INT8,
        "bf16_int8_32_16",
        ge::GRAPH_SUCCESS,
        0,
        true
    },
    // 6: BF16 + INT8, N not divisible by K (should fail)
    RotateQuantData{
        {{4, 100}, {4, 100}},
        {{64, 64}, {64, 64}},
        {{4, 100}, {4, 100}},
        {{4}, {4}},
        ge::DT_BF16,
        ge::DT_INT8,
        "bf16_int8_n_not_divisible_by_k",
        ge::GRAPH_FAILED,
        0,
        false
    },
    // 7: FP16 + INT8, M=1
    RotateQuantData{
        {{1, 256}, {1, 256}},
        {{64, 64}, {64, 64}},
        {{1, 256}, {1, 256}},
        {{1}, {1}},
        ge::DT_FLOAT16,
        ge::DT_INT8,
        "fp16_int8_1_64",
        ge::GRAPH_SUCCESS,
        0,
        true
    },
    // 8: BF16 + INT8, N=K*2 (numBlocks=2)
    RotateQuantData{
        {{4, 128}, {4, 128}},
        {{64, 64}, {64, 64}},
        {{4, 128}, {4, 128}},
        {{4}, {4}},
        ge::DT_BF16,
        ge::DT_INT8,
        "bf16_int8_4_128_blocks2",
        ge::GRAPH_SUCCESS,
        0,
        true
    }
);

INSTANTIATE_TEST_SUITE_P(RotateQuantTilingCases, TilingRotateQuant, RotateQuantTestCases);
