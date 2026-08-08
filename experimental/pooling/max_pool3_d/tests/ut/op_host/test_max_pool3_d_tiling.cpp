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

#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"

namespace {
struct MaxPool3DCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    size_t dataSize = 0;
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
};

void SetPlatformResources(fe::PlatFormInfos* platformInfo, std::map<std::string, std::string>& socInfos,
                          std::map<std::string, std::string>& aicoreSpec,
                          std::map<std::string, std::string>& intrinsics)
{
    platformInfo->SetPlatformRes("SoCInfo", socInfos);
    platformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo->SetCoreNumByCoreType("AICore");
    platformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
}

TilingResult RunTiling(const gert::StorageShape& input, const gert::StorageShape& output, ge::DataType dtype,
                       ge::Format originFormat, const std::string& dataFormat, const std::vector<int64_t>& ksize,
                       const std::vector<int64_t>& strides, const std::string& padding,
                       const std::vector<int64_t>& pads = {0, 0, 0, 0, 0, 0},
                       const std::vector<int64_t>& dilation = {1, 1, 1, 1, 1}, int64_t ceilMode = 0,
                       ge::Format outputOriginFormat = ge::FORMAT_RESERVED,
                       ge::Format outputStorageFormat = ge::FORMAT_ND, ge::Format inputStorageFormat = ge::FORMAT_ND)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3D");
    if (opImpl == nullptr) {
        return {};
    }
    if (opImpl->tiling == nullptr) {
        return {};
    }
    if (opImpl->tiling_parse == nullptr) {
        return {};
    }

    const std::string platformJson = R"({
        "hardware_info": {
            "BT_SIZE": 0, "load3d_constraints": "1",
            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
            "CORE_NUM": 40
        }
    })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(platformJson.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    if (!platformInfo.Init()) {
        return {};
    }
    MaxPool3DCompileInfo compileInfo = {};
    auto parseHolder = gert::KernelRunContextFaker()
                           .KernelIONum(2, 1)
                           .Inputs({const_cast<char*>(platformJson.c_str()), reinterpret_cast<void*>(&platformInfo)})
                           .Outputs({&compileInfo})
                           .Build();
    auto* parseContext = parseHolder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr) {
        return {};
    }
    auto* parsePlatformInfo = parseContext->GetPlatformInfo();
    if (parsePlatformInfo == nullptr) {
        return {};
    }
    if (!parsePlatformInfo->Init()) {
        return {};
    }
    SetPlatformResources(parsePlatformInfo, socInfos, aicoreSpec, intrinsics);
    auto* kernelContext = parseHolder.GetContext<gert::KernelContext>();
    if (kernelContext == nullptr) {
        return {};
    }
    if (opImpl->tiling_parse(kernelContext) != ge::GRAPH_SUCCESS) {
        return {};
    }

    auto tilingData = gert::TilingData::CreateCap(4096);
    if (tilingData == nullptr) {
        return {};
    }
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    if (workspaceHolder == nullptr) {
        return {};
    }
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (outputOriginFormat == ge::FORMAT_RESERVED) {
        outputOriginFormat = originFormat;
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3D")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({const_cast<gert::StorageShape*>(&input)})
                      .OutputShapes({const_cast<gert::StorageShape*>(&output)})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<void*>(&platformInfo))
                      .NodeInputTd(0, dtype, originFormat, inputStorageFormat)
                      .NodeOutputTd(0, dtype, outputOriginFormat, outputStorageFormat)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom(strides)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom(padding)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom(pads)},
                                  {"dilation", Ops::NN::AnyValue::CreateFrom(dilation)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom(ceilMode)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom(dataFormat)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr) {
        return {};
    }
    auto* tilingPlatformInfo = context->GetPlatformInfo();
    if (tilingPlatformInfo == nullptr) {
        return {};
    }
    SetPlatformResources(tilingPlatformInfo, socInfos, aicoreSpec, intrinsics);

    TilingResult result;
    result.status = opImpl->tiling(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        auto* rawTilingData = context->GetRawTilingData();
        if (rawTilingData == nullptr) {
            result.status = ge::GRAPH_FAILED;
            return result;
        }
        result.dataSize = rawTilingData->GetDataSize();
        result.blockDim = context->GetBlockDim();
        const size_t* workspaceSizes = context->GetWorkspaceSizes(1);
        result.workspaceSize = workspaceSizes == nullptr ? 0 : workspaceSizes[0];
    }
    return result;
}

TEST(MaxPool3DTiling, NcdhwValid)
{
    gert::StorageShape input = {{2, 3, 8, 10, 12}, {2, 3, 8, 10, 12}};
    gert::StorageShape output = {{2, 3, 4, 5, 4}, {2, 3, 4, 5, 4}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 3},
                                  {1, 1, 2, 2, 3}, "VALID");
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.dataSize, 0U);
    EXPECT_GT(result.blockDim, 0U);
    EXPECT_LE(result.blockDim, 40U);
    EXPECT_EQ(result.workspaceSize, 0U);
}

TEST(MaxPool3DTiling, NdhwcSame)
{
    gert::StorageShape input = {{2, 8, 10, 12, 3}, {2, 8, 10, 12, 3}};
    gert::StorageShape output = {{2, 4, 5, 4, 3}, {2, 4, 5, 4, 3}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT16, ge::FORMAT_NDHWC, "NDHWC", {1, 2, 2, 3, 1},
                                  {1, 2, 2, 3, 1}, "SAME");
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.dataSize, 0U);
    EXPECT_GT(result.blockDim, 0U);
    EXPECT_LE(result.blockDim, 40U);
    EXPECT_EQ(result.workspaceSize, 0U);
}

TEST(MaxPool3DTiling, CalculatedCeilDilatedBfloat16)
{
    gert::StorageShape input = {{1, 5, 6, 7, 3}, {1, 5, 6, 7, 3}};
    gert::StorageShape output = {{1, 3, 3, 4, 3}, {1, 3, 3, 4, 3}};
    const auto result = RunTiling(input, output, ge::DT_BF16, ge::FORMAT_NDHWC, "NDHWC", {1, 3, 3, 3, 1},
                                  {1, 2, 2, 2, 1}, "CALCULATED", {1, 1, 1, 1, 1, 1}, {1, 1, 2, 1, 1}, 1);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.dataSize, 0U);
    EXPECT_GT(result.blockDim, 0U);
    EXPECT_LE(result.blockDim, 40U);
}

TEST(MaxPool3DTiling, ThreeElementSpatialAttrs)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    gert::StorageShape output = {{1, 2, 2, 2, 3}, {1, 2, 2, 2, 3}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT16, ge::FORMAT_NCDHW, "NCDHW", {2, 3, 2}, {2, 2, 2},
                                  "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1});
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.dataSize, 0U);
}

TEST(MaxPool3DTiling, OneElementSpatialAttrs)
{
    gert::StorageShape input = {{1, 2, 4, 4, 4}, {1, 2, 4, 4, 4}};
    gert::StorageShape output = {{1, 2, 4, 4, 4}, {1, 2, 4, 4, 4}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1}, {1}, "VALID",
                                  {0, 0, 0, 0, 0, 0}, {1});
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.dataSize, 0U);
}

TEST(MaxPool3DTiling, Ndc1hwc0Output)
{
    gert::StorageShape input = {{1, 3, 4, 4, 4}, {1, 3, 4, 4, 4}};
    gert::StorageShape output = {{1, 3, 2, 2, 2}, {1, 2, 1, 2, 2, 8}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 2},
                                  {1, 1, 2, 2, 2}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0, ge::FORMAT_NCDHW,
                                  ge::FORMAT_NDC1HWC0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.dataSize, 0U);
}

TEST(MaxPool3DTiling, Ndc1hwc0InputAndOutput)
{
    gert::StorageShape input = {{1, 3, 4, 4, 4}, {1, 4, 1, 4, 4, 8}};
    gert::StorageShape output = {{1, 3, 2, 2, 2}, {1, 2, 1, 2, 2, 8}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 2},
                                  {1, 1, 2, 2, 2}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0, ge::FORMAT_NCDHW,
                                  ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(result.dataSize, 0U);
}

TEST(MaxPool3DTiling, Ndc1hwc0TinyK3FeatureOutputCapacity32)
{
    gert::StorageShape input = {{5, 11, 7, 3, 9}, {5, 7, 2, 3, 9, 8}};
    gert::StorageShape output = {{5, 11, 5, 1, 7}, {5, 5, 1, 1, 7, 32}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 3, 3, 3},
                                  {1, 1, 1, 1, 1}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0, ge::FORMAT_NCDHW,
                                  ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 5U);
}

TEST(MaxPool3DTiling, Ndc1hwc0MultiBatchStride2FeatureOutputCapacity32)
{
    gert::StorageShape input = {{7, 27, 25, 5, 13}, {7, 25, 4, 5, 13, 8}};
    gert::StorageShape output = {{7, 27, 13, 3, 7}, {7, 13, 1, 3, 7, 32}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 2},
                                  {1, 1, 2, 2, 2}, "SAME", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0, ge::FORMAT_NCDHW,
                                  ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 28U);
}

TEST(MaxPool3DTiling, Ndc1hwc0TinyK3FeatureFloatC016)
{
    gert::StorageShape input = {{5, 11, 7, 3, 9}, {5, 7, 1, 3, 9, 16}};
    gert::StorageShape output = {{5, 11, 5, 1, 7}, {5, 5, 1, 1, 7, 32}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 3, 3, 3},
                                  {1, 1, 1, 1, 1}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0, ge::FORMAT_NCDHW,
                                  ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 5U);
}

TEST(MaxPool3DTiling, Ndc1hwc0MultiBatchStride2FeatureFloatC016)
{
    gert::StorageShape input = {{7, 27, 25, 5, 13}, {7, 25, 2, 5, 13, 16}};
    gert::StorageShape output = {{7, 27, 13, 3, 7}, {7, 13, 1, 3, 7, 32}};
    const auto result = RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 2},
                                  {1, 1, 2, 2, 2}, "SAME", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0, ge::FORMAT_NCDHW,
                                  ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.blockDim, 28U);
}

TEST(MaxPool3DTiling, RejectsMultiBatchStride2OutputCapacity16)
{
    gert::StorageShape input = {{7, 27, 25, 5, 13}, {7, 25, 4, 5, 13, 8}};
    gert::StorageShape output = {{7, 13, 1, 3, 7, 32}, {7, 1, 2, 3, 7, 8}};
    EXPECT_EQ(
        RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}, "SAME",
                  {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
            .status,
        ge::GRAPH_FAILED);
}

TEST(MaxPool3DTiling, RejectsZeroStride)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    gert::StorageShape output = {{1, 2, 2, 3, 3}, {1, 2, 2, 3, 3}};
    EXPECT_EQ(
        RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 2}, {1, 1, 0, 2, 2}, "VALID")
            .status,
        ge::GRAPH_FAILED);
}

TEST(MaxPool3DTiling, RejectsInvalidDataFormat)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    gert::StorageShape output = {{1, 2, 2, 3, 3}, {1, 2, 2, 3, 3}};
    EXPECT_EQ(
        RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCHW", {1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}, "VALID")
            .status,
        ge::GRAPH_FAILED);
}

TEST(MaxPool3DTiling, RejectsNegativeCalculatedPad)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    gert::StorageShape output = {{1, 2, 2, 3, 3}, {1, 2, 2, 3, 3}};
    EXPECT_EQ(RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 2}, {1, 1, 2, 2, 2},
                        "CALCULATED", {-1, 0, 0, 0, 0, 0})
                  .status,
              ge::GRAPH_FAILED);
}

TEST(MaxPool3DTiling, AcceptsFrameworkCalculatedOutputShape)
{
    gert::StorageShape input = {{1, 5, 6, 7, 3}, {1, 5, 6, 7, 3}};
    gert::StorageShape output = {{1, 4, 3, 4, 3}, {1, 4, 3, 4, 3}};
    EXPECT_EQ(RunTiling(input, output, ge::DT_BF16, ge::FORMAT_NDHWC, "NDHWC", {1, 3, 3, 3, 1}, {1, 2, 2, 2, 1},
                        "CALCULATED", {1, 1, 1, 1, 1, 1}, {1, 1, 2, 1, 1}, 1)
                  .status,
              ge::GRAPH_SUCCESS);
}

TEST(MaxPool3DTiling, RejectsWrongValidOutputShape)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    gert::StorageShape output = {{1, 2, 3, 3, 3}, {1, 2, 3, 3, 3}};
    EXPECT_EQ(
        RunTiling(input, output, ge::DT_FLOAT, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}, "VALID")
            .status,
        ge::GRAPH_FAILED);
}

TEST(MaxPool3DTiling, RejectsUnsupportedDtype)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    gert::StorageShape output = {{1, 2, 2, 3, 3}, {1, 2, 2, 3, 3}};
    EXPECT_EQ(
        RunTiling(input, output, ge::DT_INT32, ge::FORMAT_NCDHW, "NCDHW", {1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}, "VALID")
            .status,
        ge::GRAPH_FAILED);
}
} // namespace
