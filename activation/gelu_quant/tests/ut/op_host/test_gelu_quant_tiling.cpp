/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_host/gelu_quant_tiling_def.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"

using namespace std;

// static uint32_t GetEndAxisLenFunc(optiling::GeluQuantTilingData& tilingData)
// {
//     auto dataPtr = tilingData.data_ptr_;
//     auto offset = tilingData.endAxisLen_offset_;
//     return *((uint32_t*)(dataPtr + offset));
// }

struct GeluQuantCompileInfo {
    int32_t vectorCoreNum{0};
    uint64_t ubSize{0};
};

struct GeluQuantData {
    // attrs
    string approximate{"none"};
    string quant_mode{"dynamic"};
    int64_t dst_type{2};
    string round_mode{"rint"};

    // inputs
    gert::StorageShape x_shape{{1, 1024, 6912}, {1, 1024, 6912}};
    gert::StorageShape input_scale_shape{{6912}, {6912}};
    gert::StorageShape input_offset_shape{{6912}, {6912}};

    // outputs
    gert::StorageShape y_shape{{1, 1024, 6912}, {1, 1024, 6912}};
    gert::StorageShape out_scale_shape{{1, 1024}, {1, 1024}};

    // data type
    ge::DataType dataType1{ge::DT_FLOAT};
    ge::DataType dataType2{ge::DT_FLOAT};

    // test debug info
    string debug_info{"tiling_info:"};

    // expect
    ge::graphStatus expect_status{ge::GRAPH_FAILED};
    uint64_t expect_tiling_key{1013};
    uint32_t expect_endAxisLen{6912};
    // others
    bool check_tiling_key{true};
    bool check_tiling_data{false};
};

class TilingGeluQuant : public ::testing::TestWithParam<GeluQuantData> {
protected:
    void SetUp() override
    {
        std::cout << "TilingGeluQuant SetUp" << std::endl;
    }

    void TearDown() override
    {
        std::cout << "TilingGeluQuant TearDown" << std::endl;
    }
};

class TilingGeluQuantRegbase : public ::testing::TestWithParam<GeluQuantData> {
protected:
    void SetUp() override
    {
        std::cout << "TilingGeluQuant SetUp" << std::endl;
    }

    void TearDown() override
    {
        std::cout << "TilingGeluQuant TearDown" << std::endl;
    }
};

TEST_P(TilingGeluQuantRegbase, gelu_quant_tiling)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    GeluQuantCompileInfo compile_info;

    string op_type("GeluQuant");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto test_params = GetParam();
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder =
        gert::TilingContextFaker()
            .SetOpType("GeluQuant")
            .NodeIoNum(3, 2)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&test_params.x_shape, &test_params.input_scale_shape, &test_params.input_offset_shape})
            .OutputShapes({&test_params.y_shape, &test_params.out_scale_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, test_params.dataType1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, test_params.dataType2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, test_params.dataType2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, test_params.dataType1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"approximate", Ops::NN::AnyValue::CreateFrom<string>(test_params.approximate)},
                 {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>(test_params.quant_mode)},
                 {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.dst_type)},
                 {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(test_params.round_mode)}})
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // check tiling result
    ge::graphStatus actual_staus = tiling_func(tiling_context);
    EXPECT_EQ(actual_staus, test_params.expect_status) << test_params.debug_info;

    if (test_params.check_tiling_key) {
        auto actual_tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(actual_tiling_key, test_params.expect_tiling_key) << test_params.debug_info;
    }

    // if (test_params.check_tiling_data) {
    //     optiling::GeluQuantTilingData tilingDataStruct;
    //     tilingDataStruct.SetDataPtr(tiling_context->GetRawTilingData()->GetData());
    //     auto actual_endAxisLen = GetEndAxisLenFunc(tilingDataStruct);
    //     ASSERT_EQ(actual_endAxisLen, test_params.expect_endAxisLen) << test_params.debug_info;
    // }
}

TEST_P(TilingGeluQuant, gelu_quant_tiling)
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
    GeluQuantCompileInfo compile_info;

    string op_type("GeluQuant");
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
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder =
        gert::TilingContextFaker()
            .SetOpType("GeluQuant")
            .NodeIoNum(3, 2)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&test_params.x_shape, &test_params.input_scale_shape, &test_params.input_offset_shape})
            .OutputShapes({&test_params.y_shape, &test_params.out_scale_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, test_params.dataType1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, test_params.dataType2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, test_params.dataType2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, test_params.dataType1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"approximate", Ops::NN::AnyValue::CreateFrom<string>(test_params.approximate)},
                 {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>(test_params.quant_mode)},
                 {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.dst_type)},
                 {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(test_params.round_mode)}})
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

    // if (test_params.check_tiling_data) {
    //     optiling::GeluQuantTilingData tilingDataStruct;
    //     tilingDataStruct.SetDataPtr(tiling_context->GetRawTilingData()->GetData());
    //     auto actual_endAxisLen = GetEndAxisLenFunc(tilingDataStruct);
    //     ASSERT_EQ(actual_endAxisLen, test_params.expect_endAxisLen) << test_params.debug_info;
    // }
}

const auto GeluQuantTestCases = ::testing::Values(
    // 0
    GeluQuantData{
        "none",   // approximate
        "static", // quant mode
        2,        // dst type
        "rint",   // round mode attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_none_1_1024_6912_normal_normal", // debug info
        ge::GRAPH_SUCCESS,
        1013,
        6912, // expect
        true,
        true // others
    },
    // 1
    GeluQuantData{
        "tanh",
        "dynamic",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                       // dtype
        "dynamic_fp32_fp32_tanh_1_1024_6912_normal_normal", // debug info
        ge::GRAPH_SUCCESS,
        1033,
        6912, // expect
        true,
        true // others
    },
    // 2
    GeluQuantData{
        "tanh",
        "dynamic",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT16,
        ge::DT_FLOAT16,                                     // dtype
        "dynamic_fp16_fp16_tanh_1_1024_6912_normal_normal", // debug info
        ge::GRAPH_SUCCESS,
        1031,
        6912, // expect
        true,
        true // others
    },
    // 3
    GeluQuantData{
        "none",
        "static",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{0}, {0}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                     // dtype
        "static_fp32_fp32_none_1_1024_6912_normal_empty", // debug info
        ge::GRAPH_FAILED,
        1003,
        6912, // expect
        false,
        false // others
    },
    // 4
    GeluQuantData{
        "tanh",
        "dynamic",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{0}, {0}},
        {{0}, {0}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_BF16,
        ge::DT_BF16,                                // dtype
        "dynamic_bf16_bf16_tanh_1_1024_6912_empty", // debug info
        ge::GRAPH_FAILED,
        1013,
        6912, // expect
        false,
        false // others
    },
    // 5
    GeluQuantData{
        "tanh",
        "static",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1}, {1}},
        {{6912}, {6912}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_BF16,
        ge::DT_BF16,                                       // dtype
        "static_bf16_bf16_tanh_1_1024_6912_scalar_normal", // debug info
        ge::GRAPH_FAILED,
        0,
        0, // expect
        false,
        false // others
    },
    // 6
    GeluQuantData{
        "tanh",
        "static",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{1}, {1}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_BF16,
        ge::DT_BF16,                                       // dtype
        "static_bf16_bf16_tanh_1_1024_6912_normal_scalar", // debug info
        ge::GRAPH_FAILED,
        0,
        0, // expect
        false,
        false // others
    },
    // 7
    GeluQuantData{
        "none",
        "static",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1}, {1}},
        {{1}, {1}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_none_1_1024_6912_scalar_scalar", // debug info
        ge::GRAPH_SUCCESS,
        1003,
        6912, // expect
        true,
        true // others
    },
    // 8
    GeluQuantData{
        "none",
        "static",
        2,
        "rint", // attr
        {{100, 1024, 69}, {100, 1024, 69}},
        {{69}, {69}},
        {{69}, {69}}, // input shape
        {{100, 1024, 69}, {100, 1024, 69}},
        {{100, 1024, 69}, {100, 1024, 69}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                  // dtype
        "static_fp32_fp32_full_kernel_small_end_axis", // debug info
        ge::GRAPH_SUCCESS,
        1023,
        69, // expect
        true,
        true // others
    },
    // 9
    GeluQuantData{
        "none",
        "static",
        2,
        "rint", // attr
        {{1, 1, 6912}, {1, 1, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1, 6912}, {1, 1, 6912}},
        {{1, 1, 6912}, {1, 1, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_not_full_kernel_split_end_axis", // debug info
        ge::GRAPH_SUCCESS,
        1023,
        6912, // expect
        true,
        true // others
    },
    // 10
    GeluQuantData{
        "tanh",
        "dynamic",
        2,
        "rint", // attr
        {{1, 1024, 69120}, {1, 1024, 69120}},
        {{0}, {0}},
        {{0}, {0}}, // input shape
        {{1, 1024, 69120}, {1, 1024, 69120}},
        {{1, 1024}, {1, 1024}}, // output shape
        ge::DT_BF16,
        ge::DT_BF16,         // dtype
        "dynamic_workspace", // debug info
        ge::GRAPH_FAILED,
        1043,
        69120, // expect
        false,
        false // others
    });

// Regbase
const auto GeluQuantRegbaseTestCases = ::testing::Values(
    // 0
    GeluQuantData{
        "none",   // approximate
        "static", // quant mode
        2,        // dst type
        "rint",   // round mode attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_none_1_1024_6912_normal_normal", // debug info
        ge::GRAPH_SUCCESS,
        1013,
        6912, // expect
        true,
        true // others
    },
    // 1
    GeluQuantData{
        "tanh",
        "dynamic",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                       // dtype
        "dynamic_fp32_fp32_tanh_1_1024_6912_normal_normal", // debug info
        ge::GRAPH_SUCCESS,
        1043,
        6912, // expect
        true,
        true // others
    },
    // 2
    GeluQuantData{
        "tanh",
        "dynamic",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT16,
        ge::DT_FLOAT16,                                     // dtype
        "dynamic_fp16_fp16_tanh_1_1024_6912_normal_normal", // debug info
        ge::GRAPH_SUCCESS,
        1041,
        6912, // expect
        true,
        true // others
    },
    // 3
    GeluQuantData{
        "none",
        "static",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{0}, {0}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                     // dtype
        "static_fp32_fp32_none_1_1024_6912_normal_empty", // debug info
        ge::GRAPH_FAILED,
        1003,
        6912, // expect
        false,
        false // others
    },
    // 4
    GeluQuantData{
        "tanh",
        "dynamic",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{0}, {0}},
        {{0}, {0}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_BF16,
        ge::DT_BF16,                                // dtype
        "dynamic_bf16_bf16_tanh_1_1024_6912_empty", // debug info
        ge::GRAPH_FAILED,
        1013,
        6912, // expect
        false,
        false // others
    },
    // 5
    GeluQuantData{
        "tanh",
        "static",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1}, {1}},
        {{6912}, {6912}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_BF16,
        ge::DT_BF16,                                       // dtype
        "static_bf16_bf16_tanh_1_1024_6912_scalar_normal", // debug info
        ge::GRAPH_FAILED,
        0,
        0, // expect
        false,
        false // others
    },
    // 6
    GeluQuantData{
        "tanh",
        "static",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{6912}, {6912}},
        {{1}, {1}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_BF16,
        ge::DT_BF16,                                       // dtype
        "static_bf16_bf16_tanh_1_1024_6912_normal_scalar", // debug info
        ge::GRAPH_FAILED,
        0,
        0, // expect
        false,
        false // others
    },
    // 7
    GeluQuantData{
        "none",
        "static",
        2,
        "rint", // attr
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1}, {1}},
        {{1}, {1}}, // input shape
        {{1, 1024, 6912}, {1, 1024, 6912}},
        {{1, 1024, 6912}, {1, 1024, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_none_1_1024_6912_scalar_scalar", // debug info
        ge::GRAPH_SUCCESS,
        1003,
        6912, // expect
        true,
        true // others
    },
    // 8
    GeluQuantData{
        "none",
        "static",
        2,
        "rint", // attr
        {{100, 1024, 69}, {100, 1024, 69}},
        {{69}, {69}},
        {{69}, {69}}, // input shape
        {{100, 1024, 69}, {100, 1024, 69}},
        {{100, 1024, 69}, {100, 1024, 69}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                  // dtype
        "static_fp32_fp32_full_kernel_small_end_axis", // debug info
        ge::GRAPH_SUCCESS,
        1023,
        69, // expect
        true,
        true // others
    },
    // 9
    GeluQuantData{
        "none",
        "static",
        2,
        "rint", // attr
        {{1, 1, 6912}, {1, 1, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1, 6912}, {1, 1, 6912}},
        {{1, 1, 6912}, {1, 1, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_not_full_kernel_split_end_axis", // debug info
        ge::GRAPH_SUCCESS,
        1023,
        6912, // expect
        true,
        true // others
    },
    // 10
    GeluQuantData{
        "tanh",
        "dynamic",
        2,
        "rint", // attr
        {{1, 1024, 69120}, {1, 1024, 69120}},
        {{0}, {0}},
        {{0}, {0}}, // input shape
        {{1, 1024, 69120}, {1, 1024, 69120}},
        {{1, 1024}, {1, 1024}}, // output shape
        ge::DT_BF16,
        ge::DT_BF16,         // dtype
        "dynamic_workspace", // debug info
        ge::GRAPH_FAILED,
        1043,
        69120, // expect
        false,
        false // others
    },
    // 11
    GeluQuantData{
        "none",
        "static",
        34,
        "round", // attr
        {{1, 1, 6912}, {1, 1, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1, 6912}, {1, 1, 6912}},
        {{1, 1, 6912}, {1, 1, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_not_full_kernel_split_end_axis", // debug info
        ge::GRAPH_SUCCESS,
        1023,
        6912, // expect
        true,
        true // others
    },
    // 12
    GeluQuantData{
        "none",
        "static",
        34,
        "hybrid", // attr
        {{1, 1, 6912}, {1, 1, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1, 6912}, {1, 1, 6912}},
        {{1, 1, 6912}, {1, 1, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_not_full_kernel_split_end_axis", // debug info
        ge::GRAPH_SUCCESS,
        1023,
        6912, // expect
        true,
        true // others
    },
    // 13
    GeluQuantData{
        "none",
        "static",
        35,
        "rint", // attr
        {{1, 1, 6912}, {1, 1, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1, 6912}, {1, 1, 6912}},
        {{1, 1, 6912}, {1, 1, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_not_full_kernel_split_end_axis", // debug info
        ge::GRAPH_SUCCESS,
        1023,
        6912, // expect
        true,
        true // others
    },
    // 14
    GeluQuantData{
        "none",
        "static",
        3,
        "rint", // attr
        {{1, 1, 6912}, {1, 1, 6912}},
        {{6912}, {6912}},
        {{6912}, {6912}}, // input shape
        {{1, 1, 6912}, {1, 1, 6912}},
        {{1, 1, 6912}, {1, 1, 6912}}, // output shape
        ge::DT_FLOAT,
        ge::DT_FLOAT,                                      // dtype
        "static_fp32_fp32_not_full_kernel_split_end_axis", // debug info
        ge::GRAPH_FAILED,
        0,
        0, // expect
        false,
        false // others
    });

INSTANTIATE_TEST_SUITE_P(GeluQuantTilingCases, TilingGeluQuant, GeluQuantTestCases);
INSTANTIATE_TEST_SUITE_P(GeluQuantTilingCases, TilingGeluQuantRegbase, GeluQuantRegbaseTestCases);