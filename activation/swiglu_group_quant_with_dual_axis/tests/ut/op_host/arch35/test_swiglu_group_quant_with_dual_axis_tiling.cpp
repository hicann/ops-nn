/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/swiglu_group_quant_with_dual_axis_tiling.h"
#include "../../../../op_kernel/arch35/swiglu_group_quant_with_dual_axis_tiling_key.h"

namespace {
struct TilingCase {
    gert::StorageShape x = {{65, 768}, {65, 768}};
    gert::StorageShape weight = {{65}, {65}};
    gert::StorageShape group = {{3}, {3}};
    gert::StorageShape y1 = {{65, 384}, {65, 384}};
    gert::StorageShape scale1 = {{65, 6, 2}, {65, 6, 2}};
    gert::StorageShape y2 = {{65, 384}, {65, 384}};
    gert::StorageShape scale2 = {{2, 384, 2}, {2, 384, 2}};
    gert::StorageShape origin = {{0}, {0}};
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_FLOAT;
    ge::DataType yType = ge::DT_FLOAT8_E4M3FN;
    int64_t dstType = ge::DT_FLOAT8_E4M3FN;
    int64_t quantMode = 1;
    bool outputOrigin = false;
    bool hasWeight = false;
    bool hasGroup = false;
    float clampLimit = 7.0f;
    float alpha = 1.702f;
    float bias = 1.0f;
    ge::graphStatus expected = ge::GRAPH_SUCCESS;
};

void RunCase(const TilingCase& tc)
{
    const std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64}})";
    std::map<std::string, std::string> socInfo;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> versions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfo, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::SwigluGroupQuantWithDualAxisCompileInfo compileInfo;
    const std::string opType = "SwigluGroupQuantWithDualAxis";
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    ASSERT_NE(impl, nullptr);

    std::vector<char> compileInfoBuffer(compileInfoString.begin(), compileInfoString.end());
    compileInfoBuffer.push_back('\0');
    auto parseHolder = gert::KernelRunContextFaker()
                           .KernelIONum(2, 1)
                           .Inputs({compileInfoBuffer.data(), &platformInfo})
                           .Outputs({&compileInfo})
                           .Build();
    auto* parsePlatform = parseHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo();
    ASSERT_TRUE(parsePlatform->Init());
    parsePlatform->SetPlatformRes("SoCInfo", socInfo);
    parsePlatform->SetPlatformRes("AICoreSpec", aicoreSpec);
    parsePlatform->SetCoreNumByCoreType("AICore");
    parsePlatform->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parsePlatform->SetPlatformRes("version", versions);
    ASSERT_EQ(impl->tiling_parse(parseHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto x = tc.x;
    auto weight = tc.weight;
    auto group = tc.group;
    auto y1 = tc.y1;
    auto scale1 = tc.scale1;
    auto y2 = tc.y2;
    auto scale2 = tc.scale2;
    auto origin = tc.origin;

    gert::TilingContextFaker faker;
    faker.SetOpType(opType)
        .NodeIoNum(3, 5)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&x, tc.hasWeight ? &weight : nullptr, tc.hasGroup ? &group : nullptr})
        .OutputShapes({&y1, &scale1, &y2, &scale2, &origin})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, tc.xType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, tc.yType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, tc.yType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(4, tc.xType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(tc.dstType)},
                    {"quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(tc.quantMode)},
                    {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(tc.clampLimit)},
                    {"output_origin", Ops::NN::AnyValue::CreateFrom<bool>(tc.outputOrigin)},
                    {"alpha", Ops::NN::AnyValue::CreateFrom<float>(tc.alpha)},
                    {"bias", Ops::NN::AnyValue::CreateFrom<float>(tc.bias)}})
        .TilingData(tilingData.get())
        .Workspace(workspace);
    if (tc.hasWeight) {
        faker.NodeInputTd(1, tc.weightType, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (tc.hasGroup) {
        faker.NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    auto* platform = context->GetPlatformInfo();
    ASSERT_NE(platform, nullptr);
    platform->SetPlatformRes("SoCInfo", socInfo);
    platform->SetPlatformRes("AICoreSpec", aicoreSpec);
    platform->SetCoreNumByCoreType("AICore");
    platform->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platform->SetPlatformRes("version", versions);

    ASSERT_EQ(impl->tiling(context), tc.expected);
    if (tc.expected == ge::GRAPH_SUCCESS) {
        const int64_t h = tc.x.GetStorageShape().GetDim(tc.x.GetStorageShape().GetDimNum() - 1) / 2;
        const uint64_t mode = (h + 255) / 256 < 64 ? TPL_MODE_ROTATE : TPL_MODE_BLOCK;
        const uint64_t hasClamp = tc.clampLimit > 0.0f;
        const uint64_t hasAttrs = hasClamp || tc.alpha != 1.0f || tc.bias != 0.0f;
        EXPECT_EQ(context->GetTilingKey(), GET_TPL_TILING_KEY(mode, tc.hasGroup, hasAttrs, hasClamp));
    }
}

TEST(SwigluGroupQuantWithDualAxisTiling, NonGroup) { RunCase(TilingCase{}); }

TEST(SwigluGroupQuantWithDualAxisTiling, GroupWeightOrigin)
{
    TilingCase tc;
    tc.hasWeight = true;
    tc.hasGroup = true;
    tc.xType = ge::DT_BF16;
    tc.weightType = ge::DT_FLOAT;
    tc.scale2 = {{4, 384, 2}, {4, 384, 2}};
    tc.origin = {{65, 384}, {65, 384}};
    tc.outputOrigin = true;
    RunCase(tc);
}

TEST(SwigluGroupQuantWithDualAxisTiling, RejectNonGroupWeight)
{
    TilingCase tc;
    tc.hasWeight = true;
    tc.expected = ge::GRAPH_FAILED;
    RunCase(tc);
}

TEST(SwigluGroupQuantWithDualAxisTiling, RejectModeFive)
{
    TilingCase tc;
    tc.quantMode = 5;
    tc.expected = ge::GRAPH_FAILED;
    RunCase(tc);
}

TEST(SwigluGroupQuantWithDualAxisTiling, RejectWrongScaleShape)
{
    TilingCase tc;
    tc.scale2 = {{3, 384, 2}, {3, 384, 2}};
    tc.expected = ge::GRAPH_FAILED;
    RunCase(tc);
}
TEST(SwigluGroupQuantWithDualAxisTiling, RejectsNegativeWeightDimensions)
{
    TilingCase tc;
    tc.hasWeight = true;
    tc.hasGroup = true;
    tc.weight = {{-1, -65}, {-1, -65}};
    tc.expected = ge::GRAPH_FAILED;
    RunCase(tc);
}

TEST(SwigluGroupQuantWithDualAxisTiling, RejectsWeightElementCountOverflow)
{
    TilingCase tc;
    tc.hasWeight = true;
    tc.hasGroup = true;
    constexpr int64_t LARGE_DIM = int64_t{1} << 32;
    tc.weight = {{LARGE_DIM, LARGE_DIM}, {LARGE_DIM, LARGE_DIM}};
    tc.expected = ge::GRAPH_FAILED;
    RunCase(tc);
}
} // namespace

TEST(SwigluGroupQuantWithDualAxisTiling, TemplateKeyCombinations)
{
    for (const int64_t h : {384, 16384}) {
        for (const bool hasGroup : {false, true}) {
            for (int attrs = 0; attrs < 3; ++attrs) {
                TilingCase tc;
                tc.x = {{64, h * 2}, {64, h * 2}};
                tc.y1 = {{64, h}, {64, h}};
                tc.y2 = tc.y1;
                tc.scale1 = {{64, h / 64, 2}, {64, h / 64, 2}};
                tc.hasGroup = hasGroup;
                const int64_t pairRows = hasGroup ? 4 : 1;
                tc.scale2 = {{pairRows, h, 2}, {pairRows, h, 2}};
                tc.alpha = attrs == 1 ? 1.702f : 1.0f;
                tc.bias = 0.0f;
                tc.clampLimit = attrs == 2 ? 7.0f : -1.0f;
                RunCase(tc);
            }
        }
    }
}
