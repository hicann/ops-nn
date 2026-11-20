/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "op_impl_registry.h"
#include "../../../op_host/add_rms_norm_dynamic_mx_quant_tiling.h"
namespace {
gert::KernelRunContextHolder MakeMxContext(optiling::AddRmsNormDynamicMxQuantCompileInfo* compileInfo)
{
    static auto tilingData = gert::TilingData::CreateCap(4096);
    reinterpret_cast<gert::TilingData*>(tilingData.get())->SetDataSize(0);
    static auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    static fe::PlatFormInfos platform;
    platform.Init();
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    static gert::StorageShape x1 = gert::StorageShape{{8, 64}, {8, 64}};
    static gert::StorageShape x2 = gert::StorageShape{{8, 64}, {8, 64}};
    static gert::StorageShape gamma = gert::StorageShape{{64}, {64}};
    static gert::StorageShape beta = gert::StorageShape{{64}, {64}};
    static gert::StorageShape x3 = gert::StorageShape{{8, 64}, {8, 64}};
    static gert::StorageShape y = gert::StorageShape{{8, 64}, {8, 64}};
    static gert::StorageShape xout = gert::StorageShape{{8, 64}, {8, 64}};
    static gert::StorageShape mxscale = gert::StorageShape{{8, 1, 2}, {8, 1, 2}};
    static gert::StorageShape rstd = gert::StorageShape{{8, 1}, {8, 1}};

    return gert::TilingContextFaker()
        .SetOpType("AddRmsNormDynamicMxQuant")
        .NodeIoNum(5, 4)
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &gamma, &beta, &x3})
        .OutputShapes({&y, &xout, &mxscale, &rstd})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6f)},
            {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
            {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>(std::string("rint"))},
            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)},
            {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(true)},
        })
        .CompileInfo(compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .TilingData(tilingData.get())
        .Workspace(workspace)
        .Build();
}
} // namespace
TEST(AddRmsNormDynamicMxQuantPlatformErrorReporting, MissingPlatformReportsOneFormalError)
{
    optiling::AddRmsNormDynamicMxQuantCompileInfo info{};
    auto holder = MakeMxContext(&info);
    auto* context = holder.GetContext<gert::TilingContext>();
    auto* kernel = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetComputeNodeInfo(), nullptr);
    const size_t slot = context->GetComputeNodeInfo()->GetInputsNum() + context->GetComputeNodeInfo()->GetOutputsNum() +
                        1U;
    ASSERT_NE(kernel->MutableInput(slot), nullptr);
    kernel->MutableInput(slot)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetPlatformInfo(), nullptr);
    ASSERT_NE(kernel->MutableInput(slot - 1U), nullptr);
    kernel->MutableInput(slot - 1U)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetCompileInfo(), nullptr);
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicMxQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->tiling(context), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing platform metadata and cached compile information");
}

class AddRmsNormDynamicMxQuantCachedResourcesTest : public testing::TestWithParam<bool> {};

TEST_P(AddRmsNormDynamicMxQuantCachedResourcesTest, MissingPlatformReportsOneFormalError)
{
    optiling::AddRmsNormDynamicMxQuantCompileInfo info{};
    info.totalCoreNum = 40;
    info.totalUbSize = 196608;
    info.vRegSize = 256;
    info.ubBlockSize = GetParam() ? 32 : 0;
    info.isRegbase = true;
    auto holder = MakeMxContext(&info);
    auto* context = holder.GetContext<gert::TilingContext>();
    auto* kernel = holder.GetContext<gert::KernelContext>();
    const size_t slot = context->GetComputeNodeInfo()->GetInputsNum() + context->GetComputeNodeInfo()->GetOutputsNum() +
                        1U;
    ASSERT_NE(kernel->MutableInput(slot), nullptr);
    kernel->MutableInput(slot)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetPlatformInfo(), nullptr);
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicMxQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->tiling(context), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing platform metadata for vector resources");
}

TEST_P(AddRmsNormDynamicMxQuantCachedResourcesTest, ValidPlatformUsesRuntimeResources)
{
    optiling::AddRmsNormDynamicMxQuantCompileInfo info{};
    info.totalCoreNum = 40;
    info.totalUbSize = 196608;
    info.vRegSize = 256;
    info.ubBlockSize = GetParam() ? 32 : 0;
    info.isRegbase = true;
    auto holder = MakeMxContext(&info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetPlatformInfo(), nullptr);
    std::map<std::string, std::string> versionInfo{{"NpuArch", "3510"}};
    context->GetPlatformInfo()->SetPlatformRes("version", versionInfo);
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicMxQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->tiling(context), ge::GRAPH_SUCCESS);
    EXPECT_GT(context->GetBlockDim(), 0U);
    EXPECT_GT(context->GetRawTilingData()->GetDataSize(), 0U);
    EXPECT_TRUE(manager.GetRawErrorMessages().empty());
}

INSTANTIATE_TEST_SUITE_P(ExplicitAndDefaultUbBlock, AddRmsNormDynamicMxQuantCachedResourcesTest,
                         testing::Values(true, false));

TEST(AddRmsNormDynamicMxQuantPlatformErrorReporting, MissingPlatformWithInvalidCachedVectorReportsOneFormalError)
{
    for (uint64_t vectorSize = 1; vectorSize < optiling::FP32_SIZE; ++vectorSize) {
        SCOPED_TRACE(vectorSize);
        optiling::AddRmsNormDynamicMxQuantCompileInfo info{};
        info.totalCoreNum = 40;
        info.totalUbSize = 196608;
        info.vRegSize = vectorSize;
        info.ubBlockSize = 32;
        info.isRegbase = true;
        auto holder = MakeMxContext(&info);
        auto* context = holder.GetContext<gert::TilingContext>();
        auto* kernel = holder.GetContext<gert::KernelContext>();
        const size_t slot = context->GetComputeNodeInfo()->GetInputsNum() +
                            context->GetComputeNodeInfo()->GetOutputsNum() + 1U;
        ASSERT_NE(kernel->MutableInput(slot), nullptr);
        kernel->MutableInput(slot)->Set(nullptr, nullptr);
        ASSERT_EQ(context->GetPlatformInfo(), nullptr);
        const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicMxQuant");
        ASSERT_NE(impl, nullptr);
        ASSERT_NE(impl->tiling, nullptr);
        auto& manager = ErrorManager::GetInstance();
        ASSERT_EQ(manager.Init(), 0);
        manager.GetRawErrorMessages();
        EXPECT_EQ(impl->tiling(context), ge::GRAPH_FAILED);
        const auto errors = manager.GetRawErrorMessages();
        ASSERT_EQ(errors.size(), 1U);
        EXPECT_EQ(errors[0].error_id, "EZ1008");
        EXPECT_EQ(errors[0].args_map.at("reason"), "Missing platform metadata for vector resources");
    }
}

TEST(AddRmsNormDynamicMxQuantPlatformErrorReporting, MissingCachedVectorReportsOneFormalError)
{
    optiling::AddRmsNormDynamicMxQuantCompileInfo info{};
    info.totalCoreNum = 40;
    info.totalUbSize = 196608;
    info.ubBlockSize = 32;
    info.isRegbase = true;
    auto holder = MakeMxContext(&info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    auto* kernel = holder.GetContext<gert::KernelContext>();
    const size_t slot = context->GetComputeNodeInfo()->GetInputsNum() + context->GetComputeNodeInfo()->GetOutputsNum() +
                        1U;
    ASSERT_NE(kernel->MutableInput(slot), nullptr);
    kernel->MutableInput(slot)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetPlatformInfo(), nullptr);
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicMxQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->tiling(context), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing platform metadata for vector resources");
}

TEST(AddRmsNormDynamicMxQuantPlatformErrorReporting, MissingCachedVectorUsesPlatformResources)
{
    optiling::AddRmsNormDynamicMxQuantCompileInfo info{};
    info.totalCoreNum = 40;
    info.totalUbSize = 196608;
    info.isRegbase = true;
    auto holder = MakeMxContext(&info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetPlatformInfo(), nullptr);
    std::map<std::string, std::string> versionInfo{{"NpuArch", "3510"}};
    context->GetPlatformInfo()->SetPlatformRes("version", versionInfo);
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicMxQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->tiling(context), ge::GRAPH_SUCCESS);
    EXPECT_GT(context->GetBlockDim(), 0U);
    EXPECT_GT(context->GetRawTilingData()->GetDataSize(), 0U);
    EXPECT_TRUE(manager.GetRawErrorMessages().empty());
}
