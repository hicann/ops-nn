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
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_common.h"
#include "../../../op_host/op_tiling/arch35/matmul_emu_split_weight_compile_info.h"
#include "../../../op_host/op_tiling/arch35/matmul_emu_split_weight_tiling.h"
#include "../../../op_kernel/matmul_emu_split_weight_tiling_data.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"

using namespace std;

struct MatmulEmuSplitWeightTilingDataParam {
    string case_name;
    string compile_info;

    // inputs
    gert::StorageShape x_shape;
    gert::StorageShape wHigh_shape;
    gert::StorageShape wLow_shape;

    // outputs
    gert::StorageShape y_shape;

    // data type
    ge::DataType xDataType{ge::DT_BF16};
    ge::DataType yDataType{ge::DT_FLOAT};

    // attrs
    float wLowScale{0.00390625f};
    bool transX{false};
    bool transW{false};
    int32_t yDtype{0};

    // expect
    ge::graphStatus expect_status{ge::GRAPH_SUCCESS};
};

class TilingMatmulEmuSplitWeight : public ::testing::TestWithParam<MatmulEmuSplitWeightTilingDataParam> {
protected:
    void SetUp() override { std::cout << "TilingMatmulEmuSplitWeight SetUp" << std::endl; }

    void TearDown() override { std::cout << "TilingMatmulEmuSplitWeight TearDown" << std::endl; }
};

static string get_map_string(const std::map<string, string>& m, const string& key)
{
    auto it = m.find(key);
    return (it != m.end()) ? it->second : "0";
}

TEST_P(TilingMatmulEmuSplitWeight, matmul_emu_split_weight_tiling)
{
    auto param = GetParam();

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::MatmulEmuSplitWeightCompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(param.compile_info.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulEmuSplitWeight")->tiling_parse;

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tiling_data = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("MatmulEmuSplitWeight")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&param.x_shape, &param.wHigh_shape, &param.wLow_shape})
                      .OutputShapes({&param.y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, param.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, param.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.yDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"w_low_scale", Ops::NN::AnyValue::CreateFrom<float>(param.wLowScale)},
                                  {"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(param.transX)},
                                  {"transpose_w", Ops::NN::AnyValue::CreateFrom<bool>(param.transW)},
                                  {"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(param.yDtype)}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ge::graphStatus actual_status = tiling_func(tiling_context);
    EXPECT_EQ(actual_status, param.expect_status) << param.case_name;

    if (param.expect_status == ge::GRAPH_SUCCESS) {
        auto raw_tiling_data = tiling_context->GetRawTilingData();
        ASSERT_NE(raw_tiling_data, nullptr);
        auto tilingDataPtr = static_cast<MatmulEmuSplitWeightTilingData*>(raw_tiling_data->GetData());
        ASSERT_NE(tilingDataPtr, nullptr);
        EXPECT_EQ(tilingDataPtr->m, static_cast<uint32_t>(param.x_shape.GetOriginShape().GetDim(0)));
        EXPECT_EQ(tilingDataPtr->n, static_cast<uint32_t>(param.wHigh_shape.GetOriginShape().GetDim(1)));
        EXPECT_EQ(tilingDataPtr->k, static_cast<uint32_t>(param.x_shape.GetOriginShape().GetDim(1)));
        const bool isA2A3 = (param.compile_info.find("Ascend910B") != string::npos) ||
                            (param.compile_info.find("ASCEND910_93") != string::npos);
        if (!isA2A3) {
            EXPECT_GT(tilingDataPtr->baseM, 0U);
            EXPECT_GT(tilingDataPtr->baseN, 0U);
            EXPECT_GT(tilingDataPtr->baseK, 0U);
        }
        EXPECT_GT(tilingDataPtr->usedCoreNum, 0U);
        EXPECT_EQ(tiling_context->GetBlockDim(), tilingDataPtr->usedCoreNum);
    }
}

static const string COMPILE_INFO_950 = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown",
    "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false,
    "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "socVersion": "Ascend950",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 64}
})";

// Ascend950 with non-1:2 aic:aiv ratio (cube=32, vector=48), used to verify the 控核 restriction
static const string COMPILE_INFO_950_BAD_RATIO = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown",
    "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false,
    "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "socVersion": "Ascend950",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 48}
})";

static const string COMPILE_INFO_910B = R"({
    "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown",
    "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false,
    "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true,
    "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 24, "socVersion": "Ascend910B"}
})";

static const string COMPILE_INFO_910_93 = R"({
    "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown",
    "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false,
    "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true,
    "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 24, "socVersion": "ASCEND910_93"}
})";

static MatmulEmuSplitWeightTilingDataParam tiling_cases[] = {
    // 0: BF16 input + FP32 output, normal shape
    {"bf16_fp32_128_256_128",
     COMPILE_INFO_950,
     {{128, 256}, {128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 128}, {256, 128}},
     {{128, 128}, {128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_SUCCESS},
    // 1: small shape
    {"bf16_fp32_small_16_64",
     COMPILE_INFO_950,
     {{16, 64}, {16, 64}},
     {{64, 64}, {64, 64}},
     {{64, 64}, {64, 64}},
     {{16, 64}, {16, 64}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_SUCCESS},
    // 2: large shape
    {"bf16_fp32_large_1024",
     COMPILE_INFO_950,
     {{1024, 1024}, {1024, 1024}},
     {{1024, 1024}, {1024, 1024}},
     {{1024, 1024}, {1024, 1024}},
     {{1024, 1024}, {1024, 1024}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_SUCCESS},
    // 3: K mismatch (should fail)
    {"k_mismatch_fail",
     COMPILE_INFO_950,
     {{128, 256}, {128, 256}},
     {{255, 128}, {255, 128}},
     {{255, 128}, {255, 128}},
     {{128, 128}, {128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_FAILED},
    // 4: wLow shape mismatch (should fail)
    {"wlow_shape_mismatch_fail",
     COMPILE_INFO_950,
     {{128, 256}, {128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 127}, {256, 127}},
     {{128, 128}, {128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_FAILED},
    // 5: invalid scale (should fail)
    {"invalid_scale_fail",
     COMPILE_INFO_950,
     {{128, 256}, {128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 128}, {256, 128}},
     {{128, 128}, {128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.01f,
     false,
     false,
     0,
     ge::GRAPH_FAILED},
    // 6: unsupported yDtype (should fail)
    {"invalid_ydtype_fail",
     COMPILE_INFO_950,
     {{128, 256}, {128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 128}, {256, 128}},
     {{128, 128}, {128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     1,
     ge::GRAPH_FAILED},
    // 7: unsupported dtype FP16 (should fail)
    {"fp16_unsupported_fail",
     COMPILE_INFO_950,
     {{128, 256}, {128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 128}, {256, 128}},
     {{128, 128}, {128, 128}},
     ge::DT_FLOAT16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_FAILED},
    // 8: 3D shape (should fail)
    {"3d_shape_fail",
     COMPILE_INFO_950,
     {{2, 128, 256}, {2, 128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 128}, {256, 128}},
     {{2, 128, 128}, {2, 128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_FAILED},
    // 9: aicNum:aivNum not 1:2 (cube=32, vector=48) -> 控核校验失败 (IsCapable returns false)
    {"core_ratio_not_1to2_fail",
     COMPILE_INFO_950_BAD_RATIO,
     {{128, 256}, {128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 128}, {256, 128}},
     {{128, 128}, {128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_PARAM_INVALID},
    // 10: Atlas A2 / 910B success
    {"a2_bf16_fp32_128_256_128",
     COMPILE_INFO_910B,
     {{128, 256}, {128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 128}, {256, 128}},
     {{128, 128}, {128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_SUCCESS},
    // 11: Atlas A3 / 910_93 success
    {"a3_bf16_fp32_128_256_128",
     COMPILE_INFO_910_93,
     {{128, 256}, {128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 128}, {256, 128}},
     {{128, 128}, {128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.00390625f,
     false,
     false,
     0,
     ge::GRAPH_SUCCESS},
    // 12: Atlas A2 invalid scale
    {"a2_invalid_scale_fail",
     COMPILE_INFO_910B,
     {{128, 256}, {128, 256}},
     {{256, 128}, {256, 128}},
     {{256, 128}, {256, 128}},
     {{128, 128}, {128, 128}},
     ge::DT_BF16,
     ge::DT_FLOAT,
     0.01f,
     false,
     false,
     0,
     ge::GRAPH_FAILED},
};

INSTANTIATE_TEST_SUITE_P(MatmulEmuSplitWeightTilingCases, TilingMatmulEmuSplitWeight, testing::ValuesIn(tiling_cases));
