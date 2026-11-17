/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "base/err_mgr.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "../../../../op_host/flat_quant_tiling.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "op_common/op_host/util/op_const_def.h"
#include "../../../../../../common/inc/op_host/op_const_def_nn.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
// ascend960dt/pr(DAV_9201/DAV_9202)与ascend950(DAV_3510)共用arch35 kernel路径, tiling按compileInfo.npuArch分流:
//   INT4 -> key4(FlatQuantHigh); MXFP4对齐 -> key6(CMCT/Blaze); MXFP4非对齐 -> 920x回退key4(FlatQuantHighV2);
//   HIFLOAT4 -> key6(仅920x支持); group_list与920x非对齐空p2的MXFP4被拒绝。
// 控核约束: DAV_3510要求AIC:AIV=1:2, DAV_9201/DAV_9202(v121)要求AIC:AIV=1:1, 由TilingParse校验。
static constexpr int64_t MM_HIGH_MODE_KEY = 4;
static constexpr int64_t MM_HIGH_MODE_ALIGN_KEY = 6;
static constexpr uint32_t AIC_NUM = 32;

static void GetPlatFormInfosDavAware(const string& compileInfo, map<string, string>& socInfos,
                                     map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                                     map<string, string>& socVersion)
{
    GetPlatFormInfos(compileInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    if (compileInfo.find(R"("npuArch": "dav3510")") != string::npos ||
        compileInfo.find(R"("npuArch":"dav3510")") != string::npos) {
        socVersion["NpuArch"] = "3510";
    } else if (compileInfo.find(R"("npuArch": "dav9201")") != string::npos ||
               compileInfo.find(R"("npuArch":"dav9201")") != string::npos) {
        socVersion["NpuArch"] = "9201";
    } else if (compileInfo.find(R"("npuArch": "dav9202")") != string::npos ||
               compileInfo.find(R"("npuArch":"dav9202")") != string::npos) {
        socVersion["NpuArch"] = "9202";
    }
}

// ascend950(DAV_3510): cube=32, vector=64, AIC:AIV=1:2
static const string COMPILE_INFO_DAV_3510 = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "npuArch": "dav3510", "socVersion": "Ascend950",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 64}
})";

// v121(ascend960dt/pr): cube=32, vector=32, AIC:AIV=1:1
static string MakeCompileInfo920Series(const char* npuArch, const char* socVersion, int64_t vectorCoreCnt)
{
    string compileInfo = COMPILE_INFO_DAV_3510;
    const string oldArch = "dav3510";
    compileInfo.replace(compileInfo.find(oldArch), oldArch.size(), npuArch);
    const string oldSoc = "Ascend950";
    compileInfo.replace(compileInfo.find(oldSoc), oldSoc.size(), socVersion);
    const string oldCores = "\"vector_core_cnt\": 64";
    compileInfo.replace(compileInfo.find(oldCores), oldCores.size(),
                        "\"vector_core_cnt\": " + std::to_string(vectorCoreCnt));
    return compileInfo;
}

static const string COMPILE_INFO_960DT = MakeCompileInfo920Series("dav9201", "Ascend960DT", 32);
static const string COMPILE_INFO_960PR = MakeCompileInfo920Series("dav9202", "Ascend960PR", 32);

struct FlatQuantArch35Data {
    string case_name;
    string compile_info;

    gert::StorageShape x_shape;
    gert::StorageShape p1_shape;
    gert::StorageShape p2_shape;
    gert::StorageShape group_list_shape;
    gert::StorageShape out_shape;
    gert::StorageShape scale_shape;

    ge::DataType xDataType{ge::DT_FLOAT16};
    ge::DataType outDataType{ge::DT_INT4};
    ge::DataType scaleDataType{ge::DT_FLOAT};

    bool hasGroupList{false};
    float clipRatio{1.0f};
    int64_t dstDtype{ge::DT_INT4};
    float dstTypeMax{0.0f};
    int64_t groupListType{0};

    ge::graphStatus parse_status{ge::GRAPH_SUCCESS};
    ge::graphStatus expect_status{ge::GRAPH_SUCCESS};
    int64_t expected_key{-1};
    int64_t expected_workspace{-1};
};

// 通用用例构造: x=[K,M,N], p1=[M,M], p2=[N,N](emptyP2时为[0,0]), out/scale按输出dtype契约推导
static FlatQuantArch35Data MakeArch35Case(const string& caseName, const string& compileInfo, ge::DataType xDtype,
                                          ge::DataType outDtype, ge::DataType scaleDtype, int64_t dstAttr, int64_t k,
                                          int64_t m, int64_t n, bool emptyP2 = false, int64_t expectedKey = -1,
                                          int64_t expectedWorkspace = -1, ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    FlatQuantArch35Data data;
    data.case_name = caseName;
    data.compile_info = compileInfo;
    data.x_shape = {{k, m, n}, {k, m, n}};
    data.p1_shape = {{m, m}, {m, m}};
    data.p2_shape = emptyP2 ? gert::StorageShape{{0, 0}, {0, 0}} : gert::StorageShape{{n, n}, {n, n}};
    if (outDtype == ge::DT_FLOAT4_E2M1 || outDtype == Ops::NN::DT_HIFLOAT4) {
        const int64_t scaleLastDim = (outDtype == Ops::NN::DT_HIFLOAT4) ? 1 : 2;
        data.out_shape = {{k, m * n}, {k, m * n}};
        data.scale_shape = {{k, (m * n + 63) / 64, scaleLastDim}, {k, (m * n + 63) / 64, scaleLastDim}};
    } else {
        data.out_shape = {{k, m, n}, {k, m, n}};
        data.scale_shape = {{k}, {k}};
    }
    data.xDataType = xDtype;
    data.outDataType = outDtype;
    data.scaleDataType = scaleDtype;
    data.dstDtype = dstAttr;
    data.expected_key = expectedKey;
    data.expected_workspace = expectedWorkspace;
    data.expect_status = status;
    return data;
}

class FlatQuantArch35Tiling : public ::testing::TestWithParam<FlatQuantArch35Data> {
protected:
    void SetUp() override { std::cout << "FlatQuantArch35Tiling SetUp" << std::endl; }
    void TearDown() override { std::cout << "FlatQuantArch35Tiling TearDown" << std::endl; }
};

TEST_P(FlatQuantArch35Tiling, flat_quant_arch35_tiling)
{
    auto test_params = GetParam();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfosDavAware(test_params.compile_info, soc_infos, aicore_spec, intrinsics, soc_version);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::FlatQuantCompileInfo compile_info;

    string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_NE(tiling_parse_func, nullptr);

    // tilingParseFunc simulate: 控核比例与npuArch由platform info注入
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), test_params.parse_status)
        << test_params.case_name;
    if (test_params.parse_status != ge::GRAPH_SUCCESS) {
        return;
    }

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::TilingContextFaker faker;
    faker.SetOpType("FlatQuant")
        .OutputShapes({&test_params.out_shape, &test_params.scale_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, test_params.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, test_params.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, test_params.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, test_params.outDataType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, test_params.scaleDataType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(test_params.clipRatio)},
                    {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.dstDtype)},
                    {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(test_params.dstTypeMax)},
                    {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.groupListType)}})
        .TilingData(param.get())
        .Workspace(ws_size);
    if (test_params.hasGroupList) {
        faker.NodeIoNum(4, 2)
            .IrInstanceNum({1, 1, 1})
            .InputShapes(
                {&test_params.x_shape, &test_params.p1_shape, &test_params.p2_shape, &test_params.group_list_shape})
            .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    } else {
        faker.NodeIoNum(3, 2).IrInstanceNum({1, 1, 1}).InputShapes(
            {&test_params.x_shape, &test_params.p1_shape, &test_params.p2_shape});
    }
    auto holder = faker.Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_func(tiling_context), test_params.expect_status) << test_params.case_name;
    if (test_params.expect_status != ge::GRAPH_SUCCESS) {
        return;
    }
    if (test_params.expected_key >= 0) {
        EXPECT_EQ(static_cast<int64_t>(tiling_context->GetTilingKey()), test_params.expected_key)
            << test_params.case_name;
    }
    if (test_params.expected_workspace >= 0) {
        ASSERT_NE(tiling_context->GetWorkspaceSizes(1), nullptr);
        EXPECT_EQ(static_cast<int64_t>(tiling_context->GetWorkspaceSizes(1)[0]), test_params.expected_workspace)
            << test_params.case_name;
    }
    EXPECT_EQ(tiling_context->GetBlockDim(), AIC_NUM) << test_params.case_name;
}

// ==================== DAV_9201/DAV_9202 TilingParse 控核比例 ====================
static const vector<FlatQuantArch35Data> MakeParseRatioCases()
{
    vector<FlatQuantArch35Data> cases;
    // v121要求AIC:AIV=1:1: vector=64或48时TilingParse失败
    FlatQuantArch35Data badRatioDt = MakeArch35Case("parse_960dt_bad_ratio_fail",
                                                    MakeCompileInfo920Series("dav9201", "Ascend960DT", 64), ge::DT_BF16,
                                                    ge::DT_INT4, ge::DT_FLOAT, ge::DT_INT4, 16, 64, 64);
    badRatioDt.parse_status = ge::GRAPH_FAILED;
    cases.push_back(badRatioDt);

    FlatQuantArch35Data badRatioPr = MakeArch35Case("parse_960pr_bad_ratio_fail",
                                                    MakeCompileInfo920Series("dav9202", "Ascend960PR", 48), ge::DT_BF16,
                                                    ge::DT_INT4, ge::DT_FLOAT, ge::DT_INT4, 16, 64, 64);
    badRatioPr.parse_status = ge::GRAPH_FAILED;
    cases.push_back(badRatioPr);

    // DAV_3510要求AIC:AIV=1:2: vector=32时TilingParse失败
    FlatQuantArch35Data badRatio950 = MakeArch35Case("parse_950_bad_ratio_fail",
                                                     MakeCompileInfo920Series("dav3510", "Ascend950", 32), ge::DT_BF16,
                                                     ge::DT_INT4, ge::DT_FLOAT, ge::DT_INT4, 16, 64, 64);
    badRatio950.parse_status = ge::GRAPH_FAILED;
    cases.push_back(badRatio950);
    return cases;
}

static const vector<FlatQuantArch35Data> MakeTilingCases()
{
    // workspace公式(DAV系列, FP4/HIF4与INT4共用主体): useAivNum*(K_PER_VEC*m*n*2 +
    // 2*K_PER_VEC*mAlign*n*4) + 16MiB(INT4) / 48MiB(FP4与HIF4), K_PER_VEC=4, useAivNum=ceil(K/4)与aivNum取小
    constexpr int64_t WS_16MB = 16 * 1024 * 1024;
    constexpr int64_t WS_48MB = 48 * 1024 * 1024;
    vector<FlatQuantArch35Data> cases;

    // ==================== ascend960dt/pr INT4: key4(FlatQuantHigh) ====================
    cases.push_back(MakeArch35Case("960dt_int4_fp16_key4", COMPILE_INFO_960DT, ge::DT_FLOAT16, ge::DT_INT4,
                                   ge::DT_FLOAT, ge::DT_INT4, 16, 64, 64, false, MM_HIGH_MODE_KEY,
                                   4 * 163840 + WS_16MB));
    cases.push_back(MakeArch35Case("960pr_int4_bf16_key4", COMPILE_INFO_960PR, ge::DT_BF16, ge::DT_INT4, ge::DT_FLOAT,
                                   ge::DT_INT4, 16, 64, 64, false, MM_HIGH_MODE_KEY, 4 * 163840 + WS_16MB));

    // ==================== ascend960dt/pr MXFP4对齐: key6(CMCT) ====================
    cases.push_back(MakeArch35Case("960dt_mxfp4_align_fp16_key6", COMPILE_INFO_960DT, ge::DT_FLOAT16,
                                   ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT4_E2M1, 16, 64, 64, false,
                                   MM_HIGH_MODE_ALIGN_KEY, 4 * 163840 + WS_48MB));
    cases.push_back(MakeArch35Case("960pr_mxfp4_align_bf16_key6", COMPILE_INFO_960PR, ge::DT_BF16, ge::DT_FLOAT4_E2M1,
                                   ge::DT_FLOAT8_E8M0, ge::DT_FLOAT4_E2M1, 16, 64, 64, false, MM_HIGH_MODE_ALIGN_KEY,
                                   4 * 163840 + WS_48MB));
    cases.push_back(MakeArch35Case("960pr_mxfp4_align_empty_p2_key6", COMPILE_INFO_960PR, ge::DT_FLOAT16,
                                   ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT4_E2M1, 16, 64, 64, true,
                                   MM_HIGH_MODE_ALIGN_KEY, 4 * 163840 + WS_48MB));

    // ==================== ascend960dt/pr MXFP4非对齐: 回退key4(FlatQuantHighV2) ====================
    cases.push_back(MakeArch35Case("960dt_mxfp4_noalign_key4", COMPILE_INFO_960DT, ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1,
                                   ge::DT_FLOAT8_E8M0, ge::DT_FLOAT4_E2M1, 16, 120, 64, false, MM_HIGH_MODE_KEY,
                                   4 * 323584 + WS_48MB));
    cases.push_back(MakeArch35Case("960pr_mxfp4_noalign_key4", COMPILE_INFO_960PR, ge::DT_BF16, ge::DT_FLOAT4_E2M1,
                                   ge::DT_FLOAT8_E8M0, ge::DT_FLOAT4_E2M1, 16, 120, 64, false, MM_HIGH_MODE_KEY,
                                   4 * 323584 + WS_48MB));
    // 非对齐且空p2: 920x上拒绝(FlatQuantHighV2回退路径要求p2存在)
    cases.push_back(MakeArch35Case("960dt_mxfp4_noalign_empty_p2_fail", COMPILE_INFO_960DT, ge::DT_FLOAT16,
                                   ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT4_E2M1, 16, 120, 64, true, -1,
                                   -1, ge::GRAPH_FAILED));

    // ==================== ascend960dt/pr HIF4: 恒key6(CMCT), 仅v121支持 ====================
    cases.push_back(MakeArch35Case("960dt_hif4_align_fp16_key6", COMPILE_INFO_960DT, ge::DT_FLOAT16,
                                   Ops::NN::DT_HIFLOAT4, Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_HIFLOAT4, 16, 64, 64,
                                   false, MM_HIGH_MODE_ALIGN_KEY, 4 * 163840 + WS_48MB));
    cases.push_back(MakeArch35Case("960pr_hif4_align_bf16_key6", COMPILE_INFO_960PR, ge::DT_BF16, Ops::NN::DT_HIFLOAT4,
                                   Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_HIFLOAT4, 16, 64, 64, false,
                                   MM_HIGH_MODE_ALIGN_KEY, 4 * 163840 + WS_48MB));
    // HIF4要求M/N 16对齐
    cases.push_back(MakeArch35Case("960dt_hif4_noalign_fail", COMPILE_INFO_960DT, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4,
                                   Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_HIFLOAT4, 16, 120, 64, false, -1, -1,
                                   ge::GRAPH_FAILED));

    // ==================== ascend960系列group_list拒绝 ====================
    FlatQuantArch35Data groupListCase = MakeArch35Case("960dt_group_list_fail", COMPILE_INFO_960DT, ge::DT_BF16,
                                                       ge::DT_INT4, ge::DT_FLOAT, ge::DT_INT4, 16, 64, 64);
    groupListCase.hasGroupList = true;
    groupListCase.group_list_shape = {{1}, {1}};
    groupListCase.groupListType = 1;
    groupListCase.expect_status = ge::GRAPH_FAILED;
    cases.push_back(groupListCase);

    // ==================== ascend950对照: 同形状非对齐MXFP4走key6(Blaze), 与920x的key4回退互为区分 ====================
    cases.push_back(MakeArch35Case("950_mxfp4_noalign_key6_contrast", COMPILE_INFO_DAV_3510, ge::DT_FLOAT16,
                                   ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT4_E2M1, 16, 120, 64, false,
                                   MM_HIGH_MODE_ALIGN_KEY, 4 * 323584 + WS_48MB));
    return cases;
}

static const vector<FlatQuantArch35Data> parseRatioCases = MakeParseRatioCases();
static const vector<FlatQuantArch35Data> tilingCases = MakeTilingCases();

INSTANTIATE_TEST_SUITE_P(FlatQuantArch35ParseRatioCases, FlatQuantArch35Tiling, testing::ValuesIn(parseRatioCases));
INSTANTIATE_TEST_SUITE_P(FlatQuantArch35TilingCases, FlatQuantArch35Tiling, testing::ValuesIn(tilingCases));
} // namespace
