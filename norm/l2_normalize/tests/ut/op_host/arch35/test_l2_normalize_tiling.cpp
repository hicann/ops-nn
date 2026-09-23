/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/l2_normalize_tiling_arch35.h"
#include "../../../../op_kernel/arch35/l2_normalize_struct.h"

namespace {

constexpr const char* OP_TYPE = "L2Normalize";
constexpr const char* COMPILE_INFO = R"({
    "hardware_info": {
        "BT_SIZE": 0,
        "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 253952,
        "L2_SIZE": 33554432,
        "L1_SIZE": 524288,
        "L0A_SIZE": 65536,
        "L0B_SIZE": 65536,
        "L0C_SIZE": 131072,
        "CORE_NUM": 56,
        "socVersion": "Ascend950"
    }
})";

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t key = 0;
};

TilingResult RunTiling(const std::vector<int64_t>& inputDims, const std::vector<int64_t>& axis, float eps = 1e-4f,
                       ge::DataType inputType = ge::DT_FLOAT, ge::DataType outputType = ge::DT_FLOAT,
                       ge::Format inputFormat = ge::FORMAT_ND, const std::vector<int64_t>* outputDims = nullptr,
                       ge::Format outputFormat = ge::FORMAT_RESERVED)
{
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> version = {{"NpuArch", "3510"}, {"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics, version);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::L2NormalizeCompileInfo compileInfo;

    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    if (opImpl == nullptr || opImpl->tiling == nullptr || opImpl->tiling_parse == nullptr) {
        return {};
    }

    auto parseHolder = gert::KernelRunContextFaker()
                           .SetOpType(OP_TYPE)
                           .KernelIONum(2, 1)
                           .Inputs({const_cast<char*>(COMPILE_INFO), reinterpret_cast<void*>(&platformInfo)})
                           .Outputs({&compileInfo})
                           .Build();
    auto* parseContext = parseHolder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr || parseContext->GetPlatformInfo() == nullptr ||
        !parseContext->GetPlatformInfo()->Init()) {
        return {};
    }
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parseContext->GetPlatformInfo()->SetPlatformRes("version", version);
    if (opImpl->tiling_parse(parseHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return {};
    }

    gert::StorageShape inputShape = MakeShape(inputDims);
    gert::StorageShape outputShape = MakeShape(outputDims == nullptr ? inputDims : *outputDims);
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (tilingData == nullptr || workspace == nullptr) {
        return {};
    }

    auto holder = gert::TilingContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&inputShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, inputType, inputFormat, inputFormat)
                      .NodeOutputTd(0, outputType, outputFormat == ge::FORMAT_RESERVED ? inputFormat : outputFormat,
                                    outputFormat == ge::FORMAT_RESERVED ? inputFormat : outputFormat)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(axis)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto* context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return {};
    }
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetPlatformInfo()->SetPlatformRes("version", version);

    TilingResult result;
    result.status = opImpl->tiling(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.key = context->GetTilingKey();
    }
    return result;
}

} // namespace

TEST(L2NormalizeTiling, IsRegistered)
{
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    ASSERT_NE(opImpl, nullptr);
    EXPECT_NE(opImpl->tiling, nullptr);
    EXPECT_NE(opImpl->tiling_parse, nullptr);
}

TEST(L2NormalizeTiling, SelectsBaseEmptyAndGroupTemplates)
{
    const auto base = RunTiling({4, 32}, {1});
    ASSERT_EQ(base.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(base.key, GET_TPL_TILING_KEY(0, 0));

    const auto empty = RunTiling({4, 0, 32}, {1});
    ASSERT_EQ(empty.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(empty.key, GET_TPL_TILING_KEY(0, 1));

    const auto group = RunTiling({1, 200000}, {1});
    ASSERT_EQ(group.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(group.key, GET_TPL_TILING_KEY(1, 0));
}

TEST(L2NormalizeTiling, AcceptsContractAxisAndEpsDomains)
{
    EXPECT_EQ(RunTiling({2, 3, 4}, {1}, 1e-4f, ge::DT_FLOAT16, ge::DT_FLOAT16).status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({2, 3, 4}, {}).status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({2, 3, 4}, {-1, 2, 2}).status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({2, 3, 4}, {1}, 0.0f).status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({2, 3, 4}, {1}, -1.0f).status, ge::GRAPH_SUCCESS);
}

TEST(L2NormalizeTiling, RejectsInvalidShapeTypeFormatAndAxis)
{
    EXPECT_EQ(RunTiling({2, 3, 4}, {3}).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({}, {}).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({1, 1, 1, 1, 1, 1, 1, 1, 1}, {0}).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3}, {1}, 1e-4f, ge::DT_INT32, ge::DT_INT32).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3}, {1}, 1e-4f, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_NCHW).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3}, {1}, 1e-4f, ge::DT_FLOAT, ge::DT_FLOAT16).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3}, {1}, 1e-4f, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND, nullptr, ge::FORMAT_NCHW).status,
              ge::GRAPH_FAILED);

    const std::vector<int64_t> mismatchedOutput = {2, 4};
    EXPECT_EQ(RunTiling({2, 3}, {1}, 1e-4f, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND, &mismatchedOutput).status,
              ge::GRAPH_FAILED);
}
