/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../../../op_host/arch35/sigmoid_focal_loss_grad_tiling_arch35.h"
#include "../../../../op_kernel/arch35/sigmoid_focal_loss_grad_struct.h"
#include "base/context_builder/op_tiling_context_builder.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/expand_dims_type.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tiling_data.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "graph/ascend_string.h"
#include "platform/platform_infos_def.h"

namespace optiling {
ge::graphStatus TilingPrepareForSigmoidFocalLossGrad(gert::TilingParseContext* context);
}

namespace {

constexpr int64_t kMinBitsPerCore = 32768;
constexpr int64_t kBlockAlignElems = 512;
constexpr int64_t kUbAlignBytes = 256;
constexpr int64_t kMaxLaunchBlockDim = 65535;
constexpr size_t kTilingCapacity = 4096;

struct CaseSpec {
    std::vector<int64_t> shape{3, 173};
    std::vector<int64_t> targetShape;
    std::vector<int64_t> doutShape;
    std::vector<int64_t> weightShape;
    std::vector<int64_t> gradShape;
    ge::DataType predDtype = ge::DT_FLOAT16;
    ge::DataType targetDtype = ge::DT_INT32;
    ge::DataType doutDtype = ge::DT_FLOAT16;
    ge::DataType weightDtype = ge::DT_FLOAT16;
    ge::DataType gradDtype = ge::DT_FLOAT16;
    ge::Format predFormat = ge::FORMAT_ND;
    ge::Format targetFormat = ge::FORMAT_ND;
    ge::Format doutFormat = ge::FORMAT_ND;
    ge::Format weightFormat = ge::FORMAT_ND;
    ge::Format gradFormat = ge::FORMAT_ND;
    bool hasWeight = true;
    float alpha = 0.25F;
    float gamma = 2.0F;
    std::string reduction = "mean";
    int64_t availableCoreNum = 64;
    int64_t platformUbBytes = 253952;
};

struct DtypeProfileCase {
    ge::DataType predDtype;
    ge::DataType doutDtype;
    ge::DataType weightDtype;
    bool hasWeight;
    const char* reduction;
};

struct OracleResult {
    SigmoidFocalLossGradTilingData td{};
    uint32_t blockDim = 0;
    uint64_t tilingKey = 0;
};

int64_t CeilDiv(int64_t value, int64_t factor) { return (value + factor - 1) / factor; }

int64_t AlignUp(int64_t value, int64_t factor) { return CeilDiv(value, factor) * factor; }

int64_t AlignDown(int64_t value, int64_t factor) { return value / factor * factor; }

int64_t DtypeBytes(ge::DataType dtype) { return dtype == ge::DT_FLOAT16 ? 2 : 4; }

uint64_t ChooseTilingKey(const CaseSpec& c) { return GET_TPL_TILING_KEY(static_cast<uint64_t>(c.hasWeight)); }

OracleResult ComputeOracle(const CaseSpec& c)
{
    OracleResult result;
    const int64_t dim0 = c.shape[0] * c.shape[1];
    const int64_t predBytes = DtypeBytes(c.predDtype);
    const int64_t doutBytes = DtypeBytes(c.doutDtype);
    const int64_t weightBytes = c.hasWeight ? DtypeBytes(c.weightDtype) : 0;
    const bool hasFp16 = c.predDtype == ge::DT_FLOAT16 || c.doutDtype == ge::DT_FLOAT16 ||
                         (c.hasWeight && c.weightDtype == ge::DT_FLOAT16);
    const int64_t minDtypeBits = hasFp16 ? 16 : 32;
    const int64_t requestedCoreNum = CeilDiv(dim0, kMinBitsPerCore / minDtypeBits);
    const int64_t launchCoreNum = std::min(c.availableCoreNum, kMaxLaunchBlockDim);
    const int64_t coreNum = std::max<int64_t>(1, std::min(requestedCoreNum, launchCoreNum));
    const int64_t blockFormer = AlignUp(CeilDiv(dim0, coreNum), kBlockAlignElems);
    const int64_t blockNum = CeilDiv(dim0, blockFormer);

    const int64_t minElemBytes = c.hasWeight ? std::min({predBytes, int64_t{4}, doutBytes, weightBytes}) :
                                               std::min({predBytes, int64_t{4}, doutBytes});
    const int64_t alignElems = kUbAlignBytes / minElemBytes;
    const int64_t perElemBytes = c.hasWeight ? 76 + 2 * predBytes + doutBytes + weightBytes :
                                               72 + 2 * predBytes + doutBytes;
    const int64_t ubFormer = std::min(
        blockFormer, AlignDown(AlignDown(c.platformUbBytes, kUbAlignBytes) / perElemBytes, alignElems));
    const int64_t blockTail = dim0 - (blockNum - 1) * blockFormer;

    result.td.dim0 = dim0;
    result.td.coreNum = static_cast<int32_t>(coreNum);
    result.td.blockFormer = blockFormer;
    result.td.blockNum = blockNum;
    result.td.ubFormer = ubFormer;
    result.td.ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    result.td.ubTailOfFormerBlock = blockFormer - (result.td.ubLoopOfFormerBlock - 1) * ubFormer;
    result.td.ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    result.td.ubTailOfTailBlock = blockTail - (result.td.ubLoopOfTailBlock - 1) * ubFormer;
    result.td.weightDtype = c.hasWeight && c.weightDtype == ge::DT_FLOAT ? 1 : 0;
    result.td.alpha = c.alpha;
    result.td.gamma = c.gamma;
    result.td.reduceMeanCoef = c.reduction == "mean" ? 1.0F / static_cast<float>(dim0) : 1.0F;
    result.blockDim = static_cast<uint32_t>(blockNum);
    result.tilingKey = ChooseTilingKey(c);
    return result;
}

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

const std::vector<int64_t>& ResolveShape(const std::vector<int64_t>& fallback,
                                         const std::vector<int64_t>& overrideShape)
{
    return overrideShape.empty() ? fallback : overrideShape;
}

void InitPlatformInfo(gert::TilingContext* context, int64_t coreNum, int64_t ubBytes)
{
    auto* platformInfo = context->GetPlatformInfo();
    ASSERT_NE(platformInfo, nullptr);

    std::map<std::string, std::string> socInfo;
    socInfo["ai_core_cnt"] = std::to_string(coreNum);
    socInfo["cube_core_cnt"] = std::to_string(std::max<int64_t>(1, coreNum / 2));
    socInfo["vector_core_cnt"] = std::to_string(coreNum);
    socInfo["core_type_list"] = "AICore";
    socInfo["l2_size"] = "33554432";
    platformInfo->SetPlatformRes("SoCInfo", socInfo);

    std::map<std::string, std::string> aicoreSpec;
    aicoreSpec["ub_size"] = std::to_string(ubBytes);
    aicoreSpec["l0_a_size"] = "65536";
    aicoreSpec["l0_b_size"] = "65536";
    aicoreSpec["l0_c_size"] = "131072";
    aicoreSpec["l1_size"] = "524288";
    platformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo->SetCoreNumByCoreType("AICore");

    std::map<std::string, std::string> version;
    version["Short_SoC_version"] = "Ascend950";
    version["NpuArch"] = "3510";
    platformInfo->SetPlatformRes("version", version);
}

class SigmoidFocalLossGradTilingTest : public ::testing::Test {
protected:
    void Invoke(const CaseSpec& c, size_t tilingCapacity = kTilingCapacity)
    {
        const auto predShape = MakeShape(c.shape);
        const auto targetShape = MakeShape(ResolveShape(c.shape, c.targetShape));
        const auto doutShape = MakeShape(ResolveShape(c.shape, c.doutShape));
        const auto weightShape = MakeShape(ResolveShape(c.shape, c.weightShape));
        const auto gradShape = MakeShape(ResolveShape(c.shape, c.gradShape));

        gert::Tensor pred(predShape, gert::StorageFormat(c.predFormat, c.predFormat, gert::ExpandDimsType()),
                          gert::kOnHost, c.predDtype, nullptr);
        gert::Tensor target(targetShape, gert::StorageFormat(c.targetFormat, c.targetFormat, gert::ExpandDimsType()),
                            gert::kOnHost, c.targetDtype, nullptr);
        gert::Tensor dout(doutShape, gert::StorageFormat(c.doutFormat, c.doutFormat, gert::ExpandDimsType()),
                          gert::kOnHost, c.doutDtype, nullptr);
        gert::Tensor weight(weightShape, gert::StorageFormat(c.weightFormat, c.weightFormat, gert::ExpandDimsType()),
                            gert::kOnHost, c.weightDtype, nullptr);
        gert::Tensor grad(gradShape, gert::StorageFormat(c.gradFormat, c.gradFormat, gert::ExpandDimsType()),
                          gert::kOnHost, c.gradDtype, nullptr);

        std::vector<gert::Tensor*> inputs{&pred, &target, &dout};
        if (c.hasWeight) {
            inputs.push_back(&weight);
        }
        std::vector<gert::Tensor*> outputs{&grad};

        auto tilingBuffer = gert::TilingData::CreateCap(tilingCapacity);
        auto workspaceBuffer = gert::ContinuousVector::Create<size_t>(16);
        ASSERT_NE(tilingBuffer, nullptr);
        ASSERT_NE(workspaceBuffer, nullptr);
        auto* rawTiling = reinterpret_cast<gert::TilingData*>(tilingBuffer.get());
        std::memset(rawTiling->GetData(), 0, rawTiling->GetCapacity());
        auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceBuffer.get());
        std::memset(workspace->MutableData(), 0, 16 * sizeof(size_t));

        fe::PlatFormInfos platformInfo;
        ASSERT_TRUE(platformInfo.Init());
        optiling::SigmoidFocalLossGradCompileInfo compileInfo{c.availableCoreNum, c.platformUbBytes};

        gert::OpTilingContextBuilder builder;
        builder.OpType(ge::AscendString("SigmoidFocalLossGrad"))
            .OpName(ge::AscendString("SigmoidFocalLossGrad"))
            .IONum(inputs.size(), outputs.size())
            .InputTensors(inputs)
            .OutputTensors(outputs)
            .CompileInfo(&compileInfo)
            .PlatformInfo(&platformInfo)
            .TilingData(rawTiling)
            .Workspace(workspace);
        builder.AppendAttr(c.alpha);
        builder.AppendAttr(c.gamma);
        builder.AppendAttr(ge::AscendString(c.reduction.c_str()));

        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        InitPlatformInfo(context, std::max<int64_t>(1, c.availableCoreNum), std::max<int64_t>(256, c.platformUbBytes));

        status_ = optiling::SigmoidFocalLossGradTilingFunc(context);
        blockDim_ = context->GetBlockDim();
        tilingKey_ = context->GetTilingKey();
        workspaceBytes_ = context->GetWorkspaceSizes(1)[0];
        if (rawTiling->GetCapacity() >= sizeof(actual_)) {
            std::memcpy(&actual_, rawTiling->GetData(), sizeof(actual_));
        }
    }

    void ExpectMatchesOracle(const CaseSpec& c)
    {
        Invoke(c);
        ASSERT_EQ(status_, ge::GRAPH_SUCCESS);
        const OracleResult expected = ComputeOracle(c);
        EXPECT_EQ(actual_.dim0, expected.td.dim0);
        EXPECT_EQ(actual_.coreNum, expected.td.coreNum);
        EXPECT_EQ(actual_.blockFormer, expected.td.blockFormer);
        EXPECT_EQ(actual_.blockNum, expected.td.blockNum);
        EXPECT_EQ(actual_.ubFormer, expected.td.ubFormer);
        EXPECT_EQ(actual_.ubLoopOfFormerBlock, expected.td.ubLoopOfFormerBlock);
        EXPECT_EQ(actual_.ubTailOfFormerBlock, expected.td.ubTailOfFormerBlock);
        EXPECT_EQ(actual_.ubLoopOfTailBlock, expected.td.ubLoopOfTailBlock);
        EXPECT_EQ(actual_.ubTailOfTailBlock, expected.td.ubTailOfTailBlock);
        EXPECT_EQ(actual_.weightDtype, expected.td.weightDtype);
        EXPECT_FLOAT_EQ(actual_.alpha, expected.td.alpha);
        EXPECT_FLOAT_EQ(actual_.gamma, expected.td.gamma);
        EXPECT_FLOAT_EQ(actual_.reduceMeanCoef, expected.td.reduceMeanCoef);
        EXPECT_EQ(blockDim_, expected.blockDim);
        EXPECT_EQ(tilingKey_, expected.tilingKey);
        EXPECT_GT(workspaceBytes_, 0U);
    }

    ge::graphStatus status_ = ge::GRAPH_FAILED;
    SigmoidFocalLossGradTilingData actual_{};
    uint32_t blockDim_ = 0;
    uint64_t tilingKey_ = 0;
    size_t workspaceBytes_ = 0;
};

class SigmoidFocalLossGradDtypeProfileTest : public SigmoidFocalLossGradTilingTest,
                                             public ::testing::WithParamInterface<DtypeProfileCase> {};

TEST_P(SigmoidFocalLossGradDtypeProfileTest, RoutesAndSerializesExpectedTiling)
{
    const auto& profile = GetParam();
    CaseSpec c;
    c.predDtype = profile.predDtype;
    c.doutDtype = profile.doutDtype;
    c.weightDtype = profile.weightDtype;
    c.gradDtype = profile.predDtype;
    c.hasWeight = profile.hasWeight;
    c.reduction = profile.reduction;
    c.alpha = -0.125F;
    c.gamma = 3.25F;
    ExpectMatchesOracle(c);
    EXPECT_EQ(tilingKey_, GET_TPL_TILING_KEY(static_cast<uint64_t>(profile.hasWeight)));
}

INSTANTIATE_TEST_SUITE_P(
    AllDtypeProfiles, SigmoidFocalLossGradDtypeProfileTest,
    ::testing::Values(DtypeProfileCase{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, true, "mean"},
                      DtypeProfileCase{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, true, "sum"},
                      DtypeProfileCase{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, true, "none"},
                      DtypeProfileCase{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, true, "mean"},
                      DtypeProfileCase{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, true, "sum"},
                      DtypeProfileCase{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, true, "none"},
                      DtypeProfileCase{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, true, "mean"},
                      DtypeProfileCase{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, true, "sum"},
                      DtypeProfileCase{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_UNDEFINED, false, "none"},
                      DtypeProfileCase{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UNDEFINED, false, "mean"},
                      DtypeProfileCase{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_UNDEFINED, false, "sum"},
                      DtypeProfileCase{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UNDEFINED, false, "none"}),
    [](const ::testing::TestParamInfo<DtypeProfileCase>& info) { return "Profile" + std::to_string(info.index); });

TEST_F(SigmoidFocalLossGradTilingTest, MultiCoreAndUbTailMatchOracle)
{
    CaseSpec c;
    c.shape = {1, 8193};
    c.predDtype = ge::DT_FLOAT;
    c.doutDtype = ge::DT_FLOAT;
    c.weightDtype = ge::DT_FLOAT;
    c.gradDtype = ge::DT_FLOAT;
    c.reduction = "mean";
    c.availableCoreNum = 3;
    c.platformUbBytes = 32768;
    ExpectMatchesOracle(c);
    EXPECT_GT(actual_.blockNum, 1);
    EXPECT_GT(actual_.ubLoopOfFormerBlock, 1);
    EXPECT_NE(actual_.ubTailOfTailBlock, actual_.ubFormer);
}

TEST_F(SigmoidFocalLossGradTilingTest, RejectsUnsupportedDtypes)
{
    CaseSpec c;
    c.targetDtype = ge::DT_INT64;
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.predDtype = ge::DT_BF16;
    c.gradDtype = ge::DT_BF16;
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.gradDtype = ge::DT_FLOAT;
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.weightDtype = ge::DT_INT32;
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);
}

TEST_F(SigmoidFocalLossGradTilingTest, RejectsNonNdFormat)
{
    CaseSpec c;
    c.doutFormat = ge::FORMAT_NCHW;
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);
}

TEST_F(SigmoidFocalLossGradTilingTest, RejectsRankEmptyMismatchAndOverflowShapes)
{
    CaseSpec c;
    c.shape = {2, 3, 4};
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.shape = {2, 0};
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.targetShape = {1, 519};
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.doutShape = {1, 519};
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.weightShape = {1, 519};
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.gradShape = {1, 519};
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.shape = {std::numeric_limits<int64_t>::max(), 2};
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);
}

TEST_F(SigmoidFocalLossGradTilingTest, RejectsInvalidAttributes)
{
    CaseSpec c;
    c.alpha = std::numeric_limits<float>::infinity();
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.gamma = std::numeric_limits<float>::quiet_NaN();
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    c.reduction = "invalid";
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);
}

TEST_F(SigmoidFocalLossGradTilingTest, RejectsInsufficientUbAndSmallTilingBuffer)
{
    CaseSpec c;
    c.platformUbBytes = 256;
    Invoke(c);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);

    c = CaseSpec{};
    Invoke(c, sizeof(SigmoidFocalLossGradTilingData) - 1);
    EXPECT_EQ(status_, ge::GRAPH_FAILED);
}

TEST(SigmoidFocalLossGradTilingNullTest, RejectsNullContexts)
{
    EXPECT_EQ(optiling::SigmoidFocalLossGradTilingFunc(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(optiling::TilingPrepareForSigmoidFocalLossGrad(nullptr), ge::GRAPH_FAILED);
}

} // namespace
