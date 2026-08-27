/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_host/arch35/bn_inference_tiling_arch35.h"
#include "../../../op_kernel/arch35/bn_inference_tiling_data.h"
#include "../../../op_kernel/arch35/bn_inference_tiling_key.h"

namespace {
constexpr uint64_t TILING_DATA_SIZE = 4096U;
constexpr uint64_t UB_SIZE = 248U * 1024U;
constexpr uint64_t CORE_NUM = 64U;
constexpr size_t SIMPLIFIED_KEY_CAPACITY = 100U;

struct SimplifiedKeyResult {
    bool callbackRegistered = false;
    ge::graphStatus status = ge::GRAPH_FAILED;
    std::string key;
};

struct AttrPresence {
    bool epsilon = true;
    bool useGlobalStats = true;
    bool mode = true;
};

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

bool RunCase(const std::vector<int64_t>& xDims, ge::Format xFormat, ge::DataType xDtype, ge::DataType statisticsDtype,
             ge::DataType momentumDtype, ge::graphStatus expected, int64_t parameterLength = -1,
             const std::vector<int64_t>& yDims = {}, ge::Format yFormat = ge::FORMAT_RESERVED,
             int64_t expectedTilingKey = -1, size_t expectedBlockNum = 0,
             ge::Format xOriginFormat = ge::FORMAT_RESERVED, ge::Format yOriginFormat = ge::FORMAT_RESERVED,
             int64_t mode = 1, bool hasScale = false, bool hasOffset = false, const AttrPresence& attrPresence = {})
{
    const ge::Format resolvedXOrigin = xOriginFormat == ge::FORMAT_RESERVED ? xFormat : xOriginFormat;
    const ge::Format logicalFormat = xFormat == ge::FORMAT_ND ? resolvedXOrigin : xFormat;
    const bool channelLast = logicalFormat == ge::FORMAT_NHWC || logicalFormat == ge::FORMAT_NDHWC;
    const int64_t rank = static_cast<int64_t>(xDims.size());
    const int64_t channel = parameterLength >= 0 ? parameterLength :
                                                   (rank < 2 ? 1 : (channelLast ? xDims.back() : xDims[1]));
    const gert::StorageShape xShape = MakeShape(xDims);
    const gert::StorageShape parameterShape = MakeShape({channel});
    const gert::StorageShape momentumShape = MakeShape({});
    const gert::StorageShape outputShape = MakeShape(yDims.empty() ? xDims : yDims);
    const ge::Format outputFormat = yFormat == ge::FORMAT_RESERVED ? xFormat : yFormat;
    const ge::Format resolvedYOrigin = yOriginFormat == ge::FORMAT_RESERVED ? resolvedXOrigin : yOriginFormat;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {xShape, xDtype, xFormat, false, nullptr, resolvedXOrigin},
        {parameterShape, statisticsDtype, ge::FORMAT_ND},
        {parameterShape, statisticsDtype, ge::FORMAT_ND},
        {momentumShape, momentumDtype, ge::FORMAT_ND},
    };
    if (hasScale) {
        inputs.push_back({parameterShape, statisticsDtype, ge::FORMAT_ND});
    }
    if (hasOffset) {
        inputs.push_back({parameterShape, statisticsDtype, ge::FORMAT_ND});
    }
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {outputShape, xDtype, outputFormat, false, nullptr, resolvedYOrigin},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    if (attrPresence.epsilon) {
        attrs.push_back({"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5F)});
    }
    if (attrPresence.useGlobalStats) {
        attrs.push_back({"use_global_stats", Ops::NN::AnyValue::CreateFrom<bool>(false)});
    }
    if (attrPresence.mode) {
        attrs.push_back({"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(mode)});
    }
    const std::vector<uint32_t> inputInstances = {1, 1, 1, 1, hasScale ? 1U : 0U, hasOffset ? 1U : 0U};
    const std::vector<uint32_t> outputInstances = {1};
    optiling::BNInferenceCompileInfo compileInfo{static_cast<int64_t>(CORE_NUM), static_cast<int64_t>(UB_SIZE), 256,
                                                 32};
    gert::TilingContextPara para("BNInference", inputs, outputs, attrs, inputInstances, outputInstances, &compileInfo,
                                 CORE_NUM, UB_SIZE, TILING_DATA_SIZE);
    TilingInfo info;
    const bool succeeded = ExecuteTiling(para, info);
    EXPECT_EQ(succeeded, expected == ge::GRAPH_SUCCESS);
    if (succeeded) {
        EXPECT_EQ(info.tilingDataSize, BN_INFERENCE_TILING_DATA_EXPECTED_BYTES);
        if (expectedTilingKey >= 0) {
            EXPECT_EQ(info.tilingKey, expectedTilingKey);
        }
        if (expectedBlockNum > 0) {
            EXPECT_EQ(info.blockNum, expectedBlockNum);
        }
    }
    return succeeded;
}

bool RunModeCase(const std::vector<int64_t>& xDims, ge::Format xFormat, int64_t mode, bool hasScale, bool hasOffset,
                 ge::graphStatus expected, int64_t expectedTilingKey)
{
    return RunCase(xDims, xFormat, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, expected, -1, {}, ge::FORMAT_RESERVED,
                   expectedTilingKey, 0, ge::FORMAT_RESERVED, ge::FORMAT_RESERVED, mode, hasScale, hasOffset);
}

SimplifiedKeyResult RunSimplifiedKeyCase(ge::DataType xDtype, ge::DataType statisticsDtype, ge::DataType momentumDtype,
                                         bool hasScale, ge::DataType scaleDtype, bool hasOffset,
                                         ge::DataType offsetDtype, ge::Format featureFormat = ge::FORMAT_NCHW)
{
    gert::StorageShape xShape = MakeShape({2, 3, 4, 5});
    gert::StorageShape parameterShape = MakeShape({3});
    gert::StorageShape momentumShape = MakeShape({});
    gert::StorageShape outputShape = MakeShape({2, 3, 4, 5});
    optiling::BNInferenceCompileInfo compileInfo{static_cast<int64_t>(CORE_NUM), static_cast<int64_t>(UB_SIZE), 256,
                                                 32};
    SimplifiedKeyResult result;
    fe::PlatFormInfos platformInfo;
    if (!platformInfo.Init()) {
        ADD_FAILURE() << "Failed to initialize platform information";
        return result;
    }
    std::vector<uint32_t> inputInstances = {1, 1, 1, 1, hasScale ? 1U : 0U, hasOffset ? 1U : 0U};
    std::vector<gert::StorageShape*> inputShapes = {&xShape, &parameterShape, &parameterShape, &momentumShape};
    if (hasScale) {
        inputShapes.push_back(&parameterShape);
    }
    if (hasOffset) {
        inputShapes.push_back(&parameterShape);
    }

    gert::TilingContextFaker faker;
    faker.SetOpType("BNInference")
        .NodeIoNum(6, 1)
        .IrInstanceNum(inputInstances)
        .InputShapes(inputShapes)
        .OutputShapes({&outputShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, xDtype, featureFormat, featureFormat)
        .NodeInputTd(1, statisticsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, statisticsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, momentumDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    int32_t physicalOptionalIndex = 4;
    if (hasScale) {
        faker.NodeInputTd(physicalOptionalIndex++, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (hasOffset) {
        faker.NodeInputTd(physicalOptionalIndex, offsetDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    faker.NodeOutputTd(0, xDtype, featureFormat, featureFormat);

    auto holder = faker.Build();
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("BNInference");
    result.callbackRegistered = opImpl != nullptr && opImpl->gen_simplifiedkey != nullptr;
    if (!result.callbackRegistered) {
        return result;
    }
    std::array<ge::char_t, SIMPLIFIED_KEY_CAPACITY> simplifiedKey = {};
    result.status = opImpl->gen_simplifiedkey(holder.GetContext<gert::TilingContext>(), simplifiedKey.data());
    if (result.status == ge::GRAPH_SUCCESS) {
        result.key = simplifiedKey.data();
    }
    return result;
}
} // namespace

TEST(BNInferenceTilingTest, GeneratesD02SimplifiedKeyWithoutAffineInputs)
{
    const SimplifiedKeyResult result = RunSimplifiedKeyCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, false,
                                                            ge::DT_UNDEFINED, false, ge::DT_UNDEFINED);
    ASSERT_TRUE(result.callbackRegistered);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, "diy,0/2/2/2/2/2/0/1/0/0/0/1/1/1");
}

TEST(BNInferenceTilingTest, GeneratesD08SimplifiedKeyWithFloatScaleOnly)
{
    const SimplifiedKeyResult result = RunSimplifiedKeyCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, true,
                                                            ge::DT_FLOAT, false, ge::DT_UNDEFINED);
    ASSERT_TRUE(result.callbackRegistered);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, "diy,0/2/2/2/2/2/0/1/0/0/0/0/0/1");
}

TEST(BNInferenceTilingTest, GeneratesD08SimplifiedKeyWithFloatOffsetOnly)
{
    const SimplifiedKeyResult result = RunSimplifiedKeyCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, false,
                                                            ge::DT_UNDEFINED, true, ge::DT_FLOAT);
    ASSERT_TRUE(result.callbackRegistered);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, "diy,0/2/2/2/2/2/0/1/0/0/0/0/0/1");
}

TEST(BNInferenceTilingTest, RejectsDifferentScaleAndOffsetDtypesForSimplifiedKey)
{
    const SimplifiedKeyResult result = RunSimplifiedKeyCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, true,
                                                            ge::DT_FLOAT16, true, ge::DT_FLOAT);
    ASSERT_TRUE(result.callbackRegistered);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(BNInferenceTilingTest, RejectsUnsupportedRequiredDtypesForSimplifiedKey)
{
    const SimplifiedKeyResult result = RunSimplifiedKeyCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, false,
                                                            ge::DT_UNDEFINED, false, ge::DT_UNDEFINED);
    ASSERT_TRUE(result.callbackRegistered);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST(BNInferenceTilingTest, MapsPublicFiveDimensionalFormatsToNdBinaryKey)
{
    const SimplifiedKeyResult ncdhw = RunSimplifiedKeyCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, false,
                                                           ge::DT_UNDEFINED, false, ge::DT_UNDEFINED, ge::FORMAT_NCDHW);
    const SimplifiedKeyResult ndhwc = RunSimplifiedKeyCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, false,
                                                           ge::DT_UNDEFINED, false, ge::DT_UNDEFINED, ge::FORMAT_NDHWC);
    ASSERT_EQ(ncdhw.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(ndhwc.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(ncdhw.key, "diy,2/2/2/2/2/2/2/0/0/0/0/0/0/0");
    EXPECT_EQ(ndhwc.key, "diy,2/2/2/2/2/2/2/0/0/0/0/0/0/0");
}

TEST(BNInferenceTilingTest, AcceptsNchwWithoutAffine)
{
    EXPECT_TRUE(RunCase({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS,
                        -1, {}, ge::FORMAT_RESERVED, BNInferenceKey::CF_GENERIC_BASE, 6));
}

TEST(BNInferenceTilingTest, UsesDefaultsWhenOptionalAttributesAreOmitted)
{
    const auto run = [](const AttrPresence& attrPresence) {
        return RunCase({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                       {}, ge::FORMAT_RESERVED, BNInferenceKey::CF_GENERIC_BASE, 0, ge::FORMAT_RESERVED,
                       ge::FORMAT_RESERVED, 1, false, false, attrPresence);
    };
    EXPECT_TRUE(run({false, true, true}));
    EXPECT_TRUE(run({true, false, true}));
    EXPECT_TRUE(run({true, true, false}));
    EXPECT_TRUE(run({false, false, false}));
}

TEST(BNInferenceTilingTest, SelectsPreFoldedKeysForAllKernelFamilies)
{
    EXPECT_TRUE(RunModeCase({2, 3, 4, 5}, ge::FORMAT_NCHW, 0, false, false, ge::GRAPH_SUCCESS,
                            BNInferenceKey::CF_GENERIC_PRE_FOLDED_BASE));
    EXPECT_TRUE(RunModeCase({129, 3, 1, 1}, ge::FORMAT_NCHW, 0, true, false, ge::GRAPH_SUCCESS,
                            BNInferenceKey::CF_PACKED_PRE_FOLDED_BASE + 1));
    EXPECT_TRUE(RunModeCase({2, 4, 5, 16}, ge::FORMAT_NHWC, 0, true, true, ge::GRAPH_SUCCESS,
                            BNInferenceKey::CL_GENERIC_PRE_FOLDED_BASE + 3));
    EXPECT_TRUE(RunModeCase({1, 256, 256, 3}, ge::FORMAT_NHWC, 0, false, false, ge::GRAPH_SUCCESS,
                            BNInferenceKey::CL_PACKED_PRE_FOLDED_BASE));
}

TEST(BNInferenceTilingTest, SelectsPreFoldedKeysForPublicFiveDimensionalLayouts)
{
    EXPECT_TRUE(RunModeCase({2, 3, 2, 4, 5}, ge::FORMAT_NCDHW, 0, true, false, ge::GRAPH_SUCCESS,
                            BNInferenceKey::CF_GENERIC_PRE_FOLDED_BASE + 1));
    EXPECT_TRUE(RunModeCase({2, 2, 4, 5, 3}, ge::FORMAT_NDHWC, 0, true, true, ge::GRAPH_SUCCESS,
                            BNInferenceKey::CL_GENERIC_PRE_FOLDED_BASE + 3));
}

TEST(BNInferenceTilingTest, RejectsOffsetOnlyInPreFoldedModeIncludingEmptyTensor)
{
    EXPECT_FALSE(RunModeCase({2, 3, 4, 5}, ge::FORMAT_NCHW, 0, false, true, ge::GRAPH_FAILED, -1));
    EXPECT_FALSE(RunModeCase({0, 3, 4, 5}, ge::FORMAT_NCHW, 0, false, true, ge::GRAPH_FAILED, -1));
}

TEST(BNInferenceTilingTest, KeepsHistoricalNonzeroModeOnFullBnPath)
{
    EXPECT_TRUE(RunModeCase({2, 3, 4, 5}, ge::FORMAT_NCHW, 2, false, false, ge::GRAPH_SUCCESS,
                            BNInferenceKey::CF_GENERIC_BASE));
}

TEST(BNInferenceTilingTest, AcceptsEmptyChannel)
{
    EXPECT_TRUE(RunCase({2, 4, 5, 0}, ge::FORMAT_NHWC, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::EMPTY, 1));
}

TEST(BNInferenceTilingTest, SelectsChannelFirstPackedPath)
{
    EXPECT_TRUE(RunCase({129, 3, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS,
                        -1, {}, ge::FORMAT_RESERVED, BNInferenceKey::CF_PACKED_BASE));
}

TEST(BNInferenceTilingTest, SelectsChannelLastGenericPath)
{
    EXPECT_TRUE(RunCase({2, 4, 5, 16}, ge::FORMAT_NHWC, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS,
                        -1, {}, ge::FORMAT_RESERVED, BNInferenceKey::CL_GENERIC_BASE));
}

TEST(BNInferenceTilingTest, SelectsChannelLastPackedPath)
{
    EXPECT_TRUE(RunCase({1, 256, 256, 3}, ge::FORMAT_NHWC, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                        ge::GRAPH_SUCCESS, -1, {}, ge::FORMAT_RESERVED, BNInferenceKey::CL_PACKED_BASE));
}

TEST(BNInferenceTilingTest, AcceptsNdRank4AsChannelFirst)
{
    EXPECT_TRUE(RunCase({2, 3, 4, 5}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::CF_GENERIC_BASE));
}

TEST(BNInferenceTilingTest, AcceptsNdRank5AsChannelFirst)
{
    EXPECT_TRUE(RunCase({2, 3, 2, 4, 5}, ge::FORMAT_ND, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::CF_GENERIC_BASE));
}

TEST(BNInferenceTilingTest, ResolvesNdStorageFromPublicOriginFormat)
{
    EXPECT_TRUE(RunCase({2, 3, 4, 5}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::CF_GENERIC_BASE, 0, ge::FORMAT_NCHW));
    EXPECT_TRUE(RunCase({2, 4, 5, 3}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::CL_GENERIC_BASE, 0, ge::FORMAT_NHWC));
    EXPECT_TRUE(RunCase({2, 3, 2, 4, 5}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::CF_GENERIC_BASE, 0, ge::FORMAT_NCDHW));
    EXPECT_TRUE(RunCase({2, 2, 4, 5, 3}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::CL_GENERIC_BASE, 0, ge::FORMAT_NDHWC));
}

TEST(BNInferenceTilingTest, NdOriginWinsWhenAxisOneAndLastAreBothChannelSized)
{
    EXPECT_TRUE(RunCase({1, 3, 2, 3}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 3, {},
                        ge::FORMAT_RESERVED, BNInferenceKey::CL_GENERIC_BASE, 0, ge::FORMAT_NHWC));
}

TEST(BNInferenceTilingTest, AcceptsEmptyNdChannelLastOrigin)
{
    EXPECT_TRUE(RunCase({2, 5, 7, 0}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::EMPTY, 1, ge::FORMAT_NHWC));
    EXPECT_TRUE(RunCase({2, 0, 7, 3}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::EMPTY, 1, ge::FORMAT_NHWC));
    EXPECT_TRUE(RunCase({2, 5, 7, 9, 0}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, -1,
                        {}, ge::FORMAT_RESERVED, BNInferenceKey::EMPTY, 1, ge::FORMAT_NDHWC));
}

TEST(BNInferenceTilingTest, RejectsNdOriginRankMismatchAndOutputOriginMismatch)
{
    EXPECT_FALSE(RunCase({2, 3, 4, 5}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED, -1,
                         {}, ge::FORMAT_RESERVED, -1, 0, ge::FORMAT_NCDHW));
    EXPECT_FALSE(RunCase({2, 4, 5, 3}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED, -1,
                         {}, ge::FORMAT_RESERVED, -1, 0, ge::FORMAT_NHWC, ge::FORMAT_NCHW));
}

TEST(BNInferenceTilingTest, AcceptsPublicFiveDimensionalFormats)
{
    EXPECT_TRUE(RunCase({2, 3, 2, 4, 5}, ge::FORMAT_NCDHW, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS,
                        -1, {}, ge::FORMAT_RESERVED, BNInferenceKey::CF_GENERIC_BASE));
    EXPECT_TRUE(RunCase({2, 2, 4, 5, 3}, ge::FORMAT_NDHWC, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS,
                        -1, {}, ge::FORMAT_RESERVED, BNInferenceKey::CL_GENERIC_BASE));
}

TEST(BNInferenceTilingTest, RejectsWrongRankForFormat)
{
    EXPECT_FALSE(RunCase({2, 3, 4, 5, 6}, ge::FORMAT_NCHW, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED));
}

TEST(BNInferenceTilingTest, RejectsWrongRankForNd)
{
    EXPECT_FALSE(RunCase({2, 3, 4}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED));
    EXPECT_FALSE(
        RunCase({2, 3, 4, 5, 6, 7}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED));
}

TEST(BNInferenceTilingTest, RejectsOneDimensionalEmptyFeatureTensor)
{
    EXPECT_FALSE(RunCase({0}, ge::FORMAT_NCHW, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED));
    EXPECT_FALSE(RunCase({0}, ge::FORMAT_NHWC, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED));
    EXPECT_FALSE(RunCase({0}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED));
}

TEST(BNInferenceTilingTest, RejectsNdRank5ParameterLengthFromLastAxis)
{
    EXPECT_FALSE(
        RunCase({2, 3, 4, 5, 7}, ge::FORMAT_ND, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED, 7));
}

TEST(BNInferenceTilingTest, RejectsParameterLengthMismatch)
{
    EXPECT_FALSE(RunCase({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED, 4));
}

TEST(BNInferenceTilingTest, RejectsUnsupportedDtypeCombination)
{
    EXPECT_FALSE(RunCase({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::GRAPH_FAILED));
}

TEST(BNInferenceTilingTest, RejectsOutputShapeMismatch)
{
    EXPECT_FALSE(RunCase({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED, -1,
                         {2, 3, 4, 6}));
}

TEST(BNInferenceTilingTest, RejectsElementCountOverflow)
{
    constexpr int64_t max = std::numeric_limits<int64_t>::max();
    EXPECT_FALSE(RunCase({max, 2, 1, 1}, ge::FORMAT_NCHW, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED));
    EXPECT_FALSE(RunCase({max, 1, 1, 2}, ge::FORMAT_NHWC, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED));
}
