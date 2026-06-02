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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/group_norm_swish_grad_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class GroupNormSwishGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupNormSwishGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupNormSwishGradTiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

namespace {
constexpr uint64_t FP32_TILING_KEY = 0;
constexpr uint64_t FP32_DETERMINISTIC_TILING_KEY = 10;
constexpr uint64_t FP16_DETERMINISTIC_TILING_KEY = 11;
constexpr uint64_t A2_A3_UB_SIZE = 196608;
constexpr uint64_t A2_A3_CORE_NUM = 48;
constexpr uint64_t ASCEND_950_UB_SIZE = 245760;
constexpr uint64_t ASCEND_950_CORE_NUM = 64;
constexpr uint64_t TILING_DATA_TASK_NUM_PER_CORE_INDEX = 7;
constexpr uint64_t TILING_DATA_TASK_NUM_PER_TAIL_CORE_INDEX = 8;
constexpr uint64_t TILING_DATA_TAIL_CORE_INDEX = 9;
constexpr uint64_t TILING_DATA_STAGE2_CORE_USED_INDEX = 17;
constexpr uint64_t TILING_DATA_CAST_ELE_NUM_INDEX = 18;
constexpr uint64_t TILING_DATA_TAIL_CAST_NUM_INDEX = 19;
constexpr uint64_t TILING_DATA_CORE_BATCH_PARTS_INDEX = 20;
constexpr uint64_t TILING_DATA_CORE_BATCH_PARTS_TAIL_REPEAT_INDEX = 21;
constexpr uint64_t TILING_DATA_REPEAT_TIME4_STAGE2_INDEX = 22;
constexpr uint64_t STAGE_ISOLATION_C = 256000;
constexpr uint64_t STAGE_ISOLATION_G = 64;
constexpr uint64_t STAGE_ISOLATION_A2_A3_CAST_ELE_NUM = 10688;
constexpr uint64_t STAGE_ISOLATION_ASCEND_950_CAST_ELE_NUM = 7616;

struct TilingRunResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint64_t taskNumPerCore = 0;
    uint64_t taskNumPerTailCore = 0;
    uint64_t tailCore = 0;
    uint64_t stage2CoreUsed = 0;
    uint64_t castEleNum = 0;
    uint64_t tailCastNum = 0;
    uint64_t coreBatchParts = 0;
    uint64_t coreBatchPartsTailRepeat = 0;
    uint64_t repeatTime4Stage2 = 0;
    size_t workspaceSize = 0;
    string tilingData;
};

static string BuildCompileInfoString(const string& socVersion)
{
    const uint64_t ubSize = (socVersion == "Ascend950") ? ASCEND_950_UB_SIZE : A2_A3_UB_SIZE;
    const uint64_t coreNum = (socVersion == "Ascend950") ? ASCEND_950_CORE_NUM : A2_A3_CORE_NUM;
    return R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": )" +
           std::to_string(ubSize) +
           R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": )" +
           std::to_string(coreNum) + R"(, "socVersion": ")" + socVersion + R"("}
                          })";
}

static void SetPlatformResource(const string& compileInfoString, fe::PlatFormInfos& platformInfo)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> version;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, version);

    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platformInfo.SetPlatformRes("version", version);
}

static TilingRunResult RunIsolatedSocTilingCase(
    const string& socVersion, bool setDeterministic, int32_t deterministic, uint32_t c = 64, uint32_t g = 32,
    uint32_t h = 128, uint32_t w = 128, uint32_t n = 12, ge::DataType dtype = ge::DT_FLOAT)
{
    std::string opType("GroupNormSwishGrad");
    string compileInfoString = BuildCompileInfoString(socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::GroupNormSwishGradCompileInfo compileInfo;

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr) {
        return {};
    }
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    auto kernelHolder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
            .Outputs({&compileInfo})
            .Build();
    auto parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr || !parseContext->GetPlatformInfo()->Init()) {
        return {};
    }
    SetPlatformResource(compileInfoString, *parseContext->GetPlatformInfo());
    if (tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return {};
    }

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    if (param == nullptr || workspaceSize == nullptr) {
        return {};
    }

    gert::StorageShape dyShape = {{n, c, h, w}, {n, c, h, w}};
    gert::StorageShape meanShape = {{n, g}, {n, g}};
    gert::StorageShape rstdShape = {{n, g}, {n, g}};
    gert::StorageShape xShape = {{n, c, h, w}, {n, c, h, w}};
    gert::StorageShape gammaShape = {{c}, {c}};
    gert::StorageShape betaShape = {{c}, {c}};
    gert::StorageShape dxShape = {{n, c, h, w}, {n, c, h, w}};
    gert::StorageShape dgammaShape = {{c}, {c}};
    gert::StorageShape dbetaShape = {{c}, {c}};

    gert::TilingContextFaker contextFaker;
    contextFaker.NodeIoNum(6, 3)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .InputShapes({&dyShape, &meanShape, &rstdShape, &xShape, &gammaShape, &betaShape})
        .OutputShapes({&dxShape, &dgammaShape, &dbetaShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs(
            {{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(g)},
             {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
             {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
             {"dgamma_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)},
             {"dbeta_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)}});
    if (setDeterministic) {
        contextFaker.DeterministicInfo(deterministic);
    }

    auto holder = contextFaker.TilingData(param.get()).Workspace(workspaceSize).Build();
    auto tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        return {};
    }
    SetPlatformResource(compileInfoString, *tilingContext->GetPlatformInfo());

    TilingRunResult result;
    result.status = tilingFunc(tilingContext);
    result.tilingKey = tilingContext->GetTilingKey();
    result.tilingData = TilingData2Str(tilingContext->GetRawTilingData());
    auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData != nullptr &&
        rawTilingData->GetDataSize() > TILING_DATA_REPEAT_TIME4_STAGE2_INDEX * sizeof(uint64_t)) {
        auto tilingData = reinterpret_cast<const uint64_t*>(rawTilingData->GetData());
        result.taskNumPerCore = tilingData[TILING_DATA_TASK_NUM_PER_CORE_INDEX];
        result.taskNumPerTailCore = tilingData[TILING_DATA_TASK_NUM_PER_TAIL_CORE_INDEX];
        result.tailCore = tilingData[TILING_DATA_TAIL_CORE_INDEX];
        result.stage2CoreUsed = tilingData[TILING_DATA_STAGE2_CORE_USED_INDEX];
        result.castEleNum = tilingData[TILING_DATA_CAST_ELE_NUM_INDEX];
        result.tailCastNum = tilingData[TILING_DATA_TAIL_CAST_NUM_INDEX];
        result.coreBatchParts = tilingData[TILING_DATA_CORE_BATCH_PARTS_INDEX];
        result.coreBatchPartsTailRepeat = tilingData[TILING_DATA_CORE_BATCH_PARTS_TAIL_REPEAT_INDEX];
        result.repeatTime4Stage2 = tilingData[TILING_DATA_REPEAT_TIME4_STAGE2_INDEX];
    }
    auto workspaceSizes = tilingContext->GetWorkspaceSizes(1);
    result.workspaceSize = (workspaceSizes == nullptr) ? 0 : workspaceSizes[0];
    return result;
}
} // namespace

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_A2_A3_default_fast_path)
{
    const vector<string> socVersions = {"Ascend910B", "ASCEND910_93"};
    for (const auto& socVersion : socVersions) {
        auto result = RunIsolatedSocTilingCase(socVersion, false, 0);
        EXPECT_EQ(result.status, ge::GRAPH_SUCCESS) << socVersion;
        EXPECT_EQ(result.tilingKey, FP32_TILING_KEY) << socVersion;
    }
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_A2_A3_explicit_deterministic)
{
    const vector<string> socVersions = {"Ascend910B", "ASCEND910_93"};
    for (const auto& socVersion : socVersions) {
        auto result = RunIsolatedSocTilingCase(socVersion, true, 1, STAGE_ISOLATION_C, STAGE_ISOLATION_G, 1, 1);
        EXPECT_EQ(result.status, ge::GRAPH_SUCCESS) << socVersion;
        EXPECT_EQ(result.tilingKey, FP32_DETERMINISTIC_TILING_KEY) << socVersion;
        EXPECT_EQ(result.castEleNum, STAGE_ISOLATION_A2_A3_CAST_ELE_NUM) << socVersion;
        EXPECT_GT(result.coreBatchParts, 0) << socVersion;
        EXPECT_GT(result.workspaceSize, 0) << socVersion;
    }
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_Ascend950_default_deterministic)
{
    auto result = RunIsolatedSocTilingCase("Ascend950", false, 0, STAGE_ISOLATION_C, STAGE_ISOLATION_G, 1, 1);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, FP32_DETERMINISTIC_TILING_KEY);
    EXPECT_EQ(result.castEleNum, STAGE_ISOLATION_ASCEND_950_CAST_ELE_NUM);
    EXPECT_GT(result.coreBatchParts, 0);
    EXPECT_GT(result.workspaceSize, 0);
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_Ascend950_fp16_default_deterministic)
{
    auto result =
        RunIsolatedSocTilingCase("Ascend950", false, 0, STAGE_ISOLATION_C, STAGE_ISOLATION_G, 1, 1, 12, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, FP16_DETERMINISTIC_TILING_KEY);
    EXPECT_GT(result.stage2CoreUsed, 0);
    EXPECT_GT(result.castEleNum, 0);
    EXPECT_GT(result.tailCastNum, 0);
    EXPECT_GT(result.coreBatchParts, 0);
    EXPECT_GT(result.repeatTime4Stage2, 0);
    EXPECT_GT(result.workspaceSize, 0);
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_Ascend950_stage2_tail_repeat)
{
    constexpr uint32_t tailRepeatC = 128000;
    constexpr uint32_t tailRepeatN = 41;
    auto result = RunIsolatedSocTilingCase("Ascend950", true, 1, tailRepeatC, STAGE_ISOLATION_G, 1, 1, tailRepeatN);
    EXPECT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, FP32_DETERMINISTIC_TILING_KEY);
    EXPECT_EQ(result.castEleNum, 4032);
    EXPECT_EQ(result.coreBatchParts, 4);
    EXPECT_EQ(result.coreBatchPartsTailRepeat, 3);
    EXPECT_EQ(result.repeatTime4Stage2, 3);
    EXPECT_GT(result.workspaceSize, 0);
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_A2_A3_tail_core_distribution)
{
    const vector<string> socVersions = {"Ascend910B", "ASCEND910_93"};
    for (const auto& socVersion : socVersions) {
        auto result = RunIsolatedSocTilingCase(socVersion, false, 0, 100, 10, 1, 1, 10);
        EXPECT_EQ(result.status, ge::GRAPH_SUCCESS) << socVersion;
        EXPECT_EQ(result.tilingKey, FP32_TILING_KEY) << socVersion;
        EXPECT_EQ(result.taskNumPerCore, 3) << socVersion;
        EXPECT_EQ(result.taskNumPerTailCore, 2) << socVersion;
        EXPECT_GT(result.tailCore, 0) << socVersion;
    }
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_0)
{
    std::string op_type("GroupNormSwishGrad");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
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
    optiling::GroupNormSwishGradCompileInfo compile_info;

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

    uint32_t N = 12;
    uint32_t C = 64;
    uint32_t H = 128;
    uint32_t W = 128;
    uint32_t G = 32;
    gert::StorageShape dy_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N, G}, {N, G}};
    gert::StorageShape rstd_shape = {{N, G}, {N, G}};
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};

    gert::StorageShape dx_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape dgamma_shape = {{C}, {C}};
    gert::StorageShape dbeta_shape = {{C}, {C}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &mean_shape, &rstd_shape, &x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape, &dbeta_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                           {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"dgamma_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"dbeta_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_data_result << std::endl;
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_1)
{
    std::string op_type("GroupNormSwishGrad");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
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
    optiling::GroupNormSwishGradCompileInfo compile_info;

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

    uint32_t N = 1;
    uint32_t C = 64;
    uint32_t H = 128;
    uint32_t W = 128;
    uint32_t G = 32;
    gert::StorageShape dy_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N, G}, {N, G}};
    gert::StorageShape rstd_shape = {{N, G}, {N, G}};
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};

    gert::StorageShape dx_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape dgamma_shape = {{C}, {C}};
    gert::StorageShape dbeta_shape = {{C}, {C}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &mean_shape, &rstd_shape, &x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape, &dbeta_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                           {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"dgamma_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"dbeta_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_key << std::endl;
    std::cout << tiling_data_result << std::endl;
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_2)
{
    std::string op_type("GroupNormSwishGrad");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
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
    optiling::GroupNormSwishGradCompileInfo compile_info;

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

    uint32_t N = 1;
    uint32_t C = 64;
    uint32_t H = 128;
    uint32_t W = 128;
    uint32_t G = 32;
    gert::StorageShape dy_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N, G}, {N, G}};
    gert::StorageShape rstd_shape = {{N, G}, {N, G}};
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};

    gert::StorageShape dx_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape dgamma_shape = {{C}, {C}};
    gert::StorageShape dbeta_shape = {{C}, {C}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &mean_shape, &rstd_shape, &x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape, &dbeta_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                           {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"dgamma_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"dbeta_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_key << std::endl;
    std::cout << tiling_data_result << std::endl;
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_3)
{
    std::string op_type("GroupNormSwishGrad");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
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
    optiling::GroupNormSwishGradCompileInfo compile_info;

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

    uint32_t N = 1;
    uint32_t C = 2130;
    uint32_t H = 1;
    uint32_t W = 1664;
    uint32_t G = 71;
    gert::StorageShape dy_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N, G}, {N, G}};
    gert::StorageShape rstd_shape = {{N, G}, {N, G}};
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};

    gert::StorageShape dx_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape dgamma_shape = {{C}, {C}};
    gert::StorageShape dbeta_shape = {{C}, {C}};
    int32_t deterministic = 1;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &mean_shape, &rstd_shape, &x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape, &dbeta_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                           {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"dgamma_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"dbeta_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .DeterministicInfo(reinterpret_cast<int32_t*>(deterministic))
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_key << std::endl;
    std::cout << tiling_data_result << std::endl;
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_4)
{
    std::string op_type("GroupNormSwishGrad");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
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
    optiling::GroupNormSwishGradCompileInfo compile_info;
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

    uint32_t N = 32;
    uint32_t C = 64;
    uint32_t H = 1;
    uint32_t W = 1;
    uint32_t G = 64;
    gert::StorageShape dy_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N, G}, {N, G}};
    gert::StorageShape rstd_shape = {{N, G}, {N, G}};
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};

    gert::StorageShape dx_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape dgamma_shape = {{C}, {C}};
    gert::StorageShape dbeta_shape = {{C}, {C}};
    int32_t deterministic = 1;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &mean_shape, &rstd_shape, &x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape, &dbeta_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                           {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"dgamma_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"dbeta_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .DeterministicInfo(reinterpret_cast<int32_t*>(deterministic))
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_key << std::endl;
    std::cout << tiling_data_result << std::endl;
}

TEST_F(GroupNormSwishGradTiling, GroupNormSwishGrad_tiling_5)
{
    std::string op_type("GroupNormSwishGrad");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
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
    optiling::GroupNormSwishGradCompileInfo compile_info;
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

    uint32_t N = 32;
    uint32_t C = 64;
    uint32_t H = 1;
    uint32_t W = 1;
    uint32_t G = 64;
    gert::StorageShape dy_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N, G}, {N, G}};
    gert::StorageShape rstd_shape = {{N, G}, {N, G}};
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};

    gert::StorageShape dx_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape dgamma_shape = {{C}, {C}};
    gert::StorageShape dbeta_shape = {{C}, {C}};
    int32_t deterministic = 1;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &mean_shape, &rstd_shape, &x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape, &dbeta_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                           {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"dgamma_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"dbeta_is_require", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .DeterministicInfo(reinterpret_cast<int32_t*>(deterministic))
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_key << std::endl;
    std::cout << tiling_data_result << std::endl;
}
