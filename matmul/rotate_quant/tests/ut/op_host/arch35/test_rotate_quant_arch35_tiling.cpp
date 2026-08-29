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
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_common.h"
#include "../../../../op_host/arch35/rotate_quant_tiling.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"

using namespace std;

struct RotateQuantArch35Data {
    string case_name;
    string compile_info;

    gert::StorageShape x_shape;
    gert::StorageShape rot_shape;
    gert::StorageShape y_shape;
    gert::StorageShape scale_shape;

    ge::DataType xDataType{ge::DT_BF16};
    ge::DataType yDataType{ge::DT_FLOAT8_E4M3FN};
    ge::DataType scaleDataType{ge::DT_FLOAT8_E8M0};

    int64_t axis{-1};
    string roundMode{"rint"};
    int64_t scaleAlg{0};
    float dstTypeMax{0.0f};
    bool trans{false};

    ge::graphStatus expect_status{ge::GRAPH_SUCCESS};
};

class TilingRotateQuantArch35 : public ::testing::TestWithParam<RotateQuantArch35Data> {
protected:
    void SetUp() override { std::cout << "TilingRotateQuantArch35 SetUp" << std::endl; }
    void TearDown() override { std::cout << "TilingRotateQuantArch35 TearDown" << std::endl; }
};

TEST_P(TilingRotateQuantArch35, rotate_quant_arch35_tiling)
{
    auto test_params = GetParam();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(test_params.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    Ops::NN::RotateQuant::RotateQuantAptCompileInfo compile_info;

    string op_type("RotateQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(test_params.compile_info.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
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
                      .NodeOutputTd(1, test_params.scaleDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.yDataType)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.axis)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(test_params.roundMode)},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.scaleAlg)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(test_params.dstTypeMax)},
                                  {"trans", Ops::NN::AnyValue::CreateFrom<bool>(test_params.trans)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ge::graphStatus actual_status = tiling_func(tiling_context);
    EXPECT_EQ(actual_status, test_params.expect_status) << test_params.case_name;
}

// Ascend950 platform: 32 AIC + 64 AIV (aicNum:aivNum = 1:2)
static const string COMPILE_INFO_950 = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "socVersion": "Ascend950",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 64}
})";

// Ascend950 with non-1:2 aic:aiv ratio (cube=32, vector=48)
static const string COMPILE_INFO_950_BAD_RATIO = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "socVersion": "Ascend950",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 48}
})";

static RotateQuantArch35Data rotate_quant_arch35_cases[] = {
    // 0: BF16 + FLOAT8_E4M3FN, 950 with 1:2 core ratio -> 控核通过, tiling成功
    {"bf16_fp8e4m3_950_ratio_1to2",
     COMPILE_INFO_950,
     {{128, 128}, {128, 128}},
     {{64, 64}, {64, 64}},
     {{128, 128}, {128, 128}},
     {{128, 2}, {128, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT8_E4M3FN,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_SUCCESS},
    // 1: same shape/dtype but aic:aiv not 1:2 (cube=32, vector=48) -> 控核校验失败 (IsCapable false,
    // arch注册表最终返回GRAPH_FAILED)
    {"bf16_fp8e4m3_950_ratio_not_1to2_fail",
     COMPILE_INFO_950_BAD_RATIO,
     {{128, 128}, {128, 128}},
     {{64, 64}, {64, 64}},
     {{128, 128}, {128, 128}},
     {{128, 2}, {128, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT8_E4M3FN,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
};

INSTANTIATE_TEST_SUITE_P(RotateQuantArch35TilingCases, TilingRotateQuantArch35,
                         testing::ValuesIn(rotate_quant_arch35_cases));
