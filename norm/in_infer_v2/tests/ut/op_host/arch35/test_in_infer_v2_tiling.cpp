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
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_shape.h"
#include "op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/arch35/in_infer_v2_tiling_arch35.h"

namespace {

constexpr int64_t DEFAULT_CORE_NUM = 64;
constexpr int64_t DEFAULT_UB_SIZE = 245760;
constexpr int64_t RESERVED_UB = 11840;
constexpr size_t DEFAULT_TILING_CAPACITY = 4096;

struct TilingCase {
    std::vector<int64_t> xDims = {2, 3, 4, 5};
    std::vector<int64_t> gammaDims = {6};
    std::vector<int64_t> betaDims = {6};
    std::vector<int64_t> meanDims = {6};
    std::vector<int64_t> varianceDims = {6};
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::Format xFormat = ge::FORMAT_ND;
    bool hasX = true;
    bool hasGamma = true;
    bool hasBeta = true;
    bool hasMean = true;
    bool hasVariance = true;
    bool hasBatchMean = true;
    bool hasBatchVariance = true;
    bool hasEpsilon = true;
    float epsilon = 1e-4f;
    bool hasPlatformInfo = true;
    bool hasCompileInfo = true;
    bool hasTilingData = true;
    bool hasWorkspace = true;
    int64_t coreNum = DEFAULT_CORE_NUM;
    int64_t ubSize = DEFAULT_UB_SIZE;
    size_t tilingCapacity = DEFAULT_TILING_CAPACITY;
};

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = std::numeric_limits<uint64_t>::max();
    uint64_t blockDim = 0;
    size_t workspaceSize = std::numeric_limits<size_t>::max();
    INInferV2TilingData data{};
    bool hasData = false;
};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape result;
    for (int64_t dim : dims) {
        result.MutableOriginShape().AppendDim(dim);
        result.MutableStorageShape().AppendDim(dim);
    }
    return result;
}

TilingResult RunTiling(const TilingCase& testCase)
{
    gert::StorageShape xShape = MakeStorageShape(testCase.xDims);
    gert::StorageShape gammaShape = MakeStorageShape(testCase.gammaDims);
    gert::StorageShape betaShape = MakeStorageShape(testCase.betaDims);
    gert::StorageShape meanShape = MakeStorageShape(testCase.meanDims);
    gert::StorageShape varianceShape = MakeStorageShape(testCase.varianceDims);
    gert::StorageShape yShape = MakeStorageShape(testCase.xDims);
    gert::StorageShape batchMeanShape = MakeStorageShape(testCase.meanDims);
    gert::StorageShape batchVarianceShape = MakeStorageShape(testCase.varianceDims);

    std::vector<uint32_t> inputInstanceNum = {
        testCase.hasX ? 1U : 0U,    testCase.hasGamma ? 1U : 0U,    testCase.hasBeta ? 1U : 0U,
        testCase.hasMean ? 1U : 0U, testCase.hasVariance ? 1U : 0U,
    };
    std::vector<gert::StorageShape*> inputShapes;
    std::vector<ge::DataType> inputDtypes;
    std::vector<ge::Format> inputFormats;
    auto appendInput = [&](bool present, gert::StorageShape* shape, ge::DataType dtype, ge::Format format) {
        if (present) {
            inputShapes.push_back(shape);
            inputDtypes.push_back(dtype);
            inputFormats.push_back(format);
        }
    };
    appendInput(testCase.hasX, &xShape, testCase.xDtype, testCase.xFormat);
    appendInput(testCase.hasGamma, &gammaShape, ge::DT_FLOAT, ge::FORMAT_ND);
    appendInput(testCase.hasBeta, &betaShape, ge::DT_FLOAT, ge::FORMAT_ND);
    appendInput(testCase.hasMean, &meanShape, ge::DT_FLOAT, ge::FORMAT_ND);
    appendInput(testCase.hasVariance, &varianceShape, ge::DT_FLOAT, ge::FORMAT_ND);

    std::vector<uint32_t> outputInstanceNum = {1U, testCase.hasBatchMean ? 1U : 0U,
                                               testCase.hasBatchVariance ? 1U : 0U};
    // Production contexts may omit trailing zero-instance optional-output
    // entries entirely, making GetIrOutputInstanceInfo() return nullptr.
    while (outputInstanceNum.size() > 1U && outputInstanceNum.back() == 0U) {
        outputInstanceNum.pop_back();
    }
    std::vector<gert::StorageShape*> outputShapes = {&yShape};
    if (testCase.hasBatchMean) {
        outputShapes.push_back(&batchMeanShape);
    }
    if (testCase.hasBatchVariance) {
        outputShapes.push_back(&batchVarianceShape);
    }

    const std::string compileInfoString =
        R"({"hardware_info":{"BT_SIZE":0,"load3d_constraints":"1","Intrinsic_fix_pipe_l0c2out":false,)"
        R"("Intrinsic_data_move_l12ub":true,"Intrinsic_data_move_l0c2ub":true,)"
        R"("Intrinsic_data_move_out2l1_nd2nz":false,"UB_SIZE":)" +
        std::to_string(testCase.ubSize) +
        R"(,"L2_SIZE":33554432,"L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,)"
        R"("L0C_SIZE":131072,"CORE_NUM":)" +
        std::to_string(testCase.coreNum) + R"(,"socVersion":"Ascend950"}})";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersion = {{"NpuArch", "3510"}, {"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    optiling::INInferV2CompileInfo compileInfo{testCase.coreNum, testCase.ubSize};
    auto tilingData = testCase.hasTilingData ? gert::TilingData::CreateCap(testCase.tilingCapacity) : nullptr;
    auto workspaceHolder = testCase.hasWorkspace ? gert::ContinuousVector::Create<size_t>(1) : nullptr;
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    gert::TilingContextFaker faker;
    faker.SetOpType("INInferV2");
    faker.NodeIoNum(inputShapes.size(), outputShapes.size());
    faker.IrInstanceNum(inputInstanceNum, outputInstanceNum);
    faker.InputShapes(inputShapes);
    faker.OutputShapes(outputShapes);
    if (testCase.hasCompileInfo) {
        faker.CompileInfo(&compileInfo);
    }
    if (testCase.hasPlatformInfo) {
        faker.PlatformInfo(&platformInfo);
    } else {
        faker.PlatformInfo(nullptr);
    }
    if (tilingData != nullptr) {
        faker.TilingData(tilingData.get());
    }
    if (workspace != nullptr) {
        faker.Workspace(workspace);
    }
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        faker.NodeInputTd(static_cast<int32_t>(i), inputDtypes[i], inputFormats[i], inputFormats[i]);
    }
    for (size_t i = 0; i < outputShapes.size(); ++i) {
        const ge::DataType dtype = (i == 0) ? testCase.xDtype : ge::DT_FLOAT;
        faker.NodeOutputTd(static_cast<int32_t>(i), dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (testCase.hasEpsilon) {
        faker.NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(testCase.epsilon)}});
    }

    auto holder = faker.Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    TilingResult result;
    if (context == nullptr) {
        return result;
    }
    if (testCase.hasPlatformInfo) {
        auto* contextPlatformInfo = context->GetPlatformInfo();
        if (contextPlatformInfo == nullptr) {
            return result;
        }
        contextPlatformInfo->SetPlatformRes("SoCInfo", socInfos);
        contextPlatformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
        contextPlatformInfo->SetCoreNumByCoreType("AICore");
        contextPlatformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
        contextPlatformInfo->SetPlatformRes("version", socVersion);
    }
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("INInferV2");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return result;
    }
    result.status = opImpl->tiling(context);
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }

    result.tilingKey = context->GetTilingKey();
    result.blockDim = context->GetBlockDim();
    if (context->GetWorkspaceNum() > 0) {
        const size_t* workspaceSizes = context->GetWorkspaceSizes(1);
        if (workspaceSizes != nullptr) {
            result.workspaceSize = workspaceSizes[0];
        }
    }
    const auto* rawTilingData = context->GetRawTilingData();
    if (rawTilingData != nullptr && rawTilingData->GetData() != nullptr &&
        rawTilingData->GetDataSize() >= sizeof(result.data)) {
        result.data = *reinterpret_cast<const INInferV2TilingData*>(rawTilingData->GetData());
        result.hasData = true;
    }
    return result;
}

} // namespace

class INInferV2TilingTest : public testing::Test {};

TEST_F(INInferV2TilingTest, float32FullInputsUsesUnitSplit)
{
    const TilingResult result = RunTiling({});
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.tilingKey, 0U);
    EXPECT_EQ(result.blockDim, 6U);
    EXPECT_EQ(result.workspaceSize, 0U);
    EXPECT_EQ(result.data.innerSize, 20);
    EXPECT_EQ(result.data.units, 6);
    EXPECT_EQ(result.data.formerCoreNum, 0);
    EXPECT_EQ(result.data.formerUnits, 1);
    EXPECT_EQ(result.data.latterUnits, 1);
    EXPECT_EQ(result.data.innerCores, 1);
    EXPECT_EQ(result.data.innerPerCore, 20);
    EXPECT_EQ(result.data.units * result.data.innerSize, 120);
    EXPECT_EQ(result.data.hasGammaBeta, 1);
    EXPECT_EQ(result.data.hasBatchMean, 1);
    EXPECT_EQ(result.data.hasBatchVar, 1);
    EXPECT_FLOAT_EQ(result.data.epsilon, 1e-4f);
    EXPECT_GT(result.data.ubTileSize, 0);
    EXPECT_EQ(result.data.ubTileSize % 64, 0);
}

TEST_F(INInferV2TilingTest, float16WithoutGammaBetaUsesInnerSplitAndDefaultEpsilon)
{
    TilingCase testCase;
    testCase.xDims = {1, 1, 130};
    testCase.gammaDims = {1};
    testCase.betaDims = {1};
    testCase.meanDims = {1};
    testCase.varianceDims = {1};
    testCase.xDtype = ge::DT_FLOAT16;
    testCase.hasGamma = false;
    testCase.hasBeta = false;
    testCase.hasEpsilon = false;
    const TilingResult result = RunTiling(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.blockDim, 2U);
    EXPECT_EQ(result.data.innerCores, 2);
    EXPECT_EQ(result.data.innerPerCore, 65);
    EXPECT_EQ(result.data.hasGammaBeta, 0);
    EXPECT_FLOAT_EQ(result.data.epsilon, 1e-5f);
    EXPECT_GT(result.data.ubTileSize, 0);
    EXPECT_EQ(result.data.ubTileSize % 64, 0);
}

TEST_F(INInferV2TilingTest, moreUnitsThanCoresUsesFormerAndLatterSplit)
{
    TilingCase testCase;
    testCase.xDims = {10, 10, 1};
    testCase.gammaDims = {100};
    testCase.betaDims = {100};
    testCase.meanDims = {100};
    testCase.varianceDims = {100};
    const TilingResult result = RunTiling(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.blockDim, 64U);
    EXPECT_EQ(result.data.formerCoreNum, 36);
    EXPECT_EQ(result.data.formerUnits, 2);
    EXPECT_EQ(result.data.latterUnits, 1);
}

TEST_F(INInferV2TilingTest, emptyTensorFullAxisSpaceUsesDefensivePath)
{
    // rank 2~8 逐 rank、逐轴穷举单零维。空间轴为0时 N*C 仍非0，
    // 会覆盖“y为空，但 mean/variance 仍需透传”的防御路径。
    for (size_t rank = 2; rank <= 8; ++rank) {
        for (size_t zeroAxis = 0; zeroAxis < rank; ++zeroAxis) {
            TilingCase testCase;
            testCase.xDims.assign(rank, 2);
            testCase.xDims[1] = 3;
            testCase.xDims[zeroAxis] = 0;
            const int64_t ncElements = testCase.xDims[0] * testCase.xDims[1];
            testCase.gammaDims = {ncElements};
            testCase.betaDims = {ncElements};
            testCase.meanDims = {ncElements};
            testCase.varianceDims = {ncElements};
            const TilingResult result = RunTiling(testCase);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS) << "rank=" << rank << ", zeroAxis=" << zeroAxis;
            ASSERT_TRUE(result.hasData) << "rank=" << rank << ", zeroAxis=" << zeroAxis;
            EXPECT_EQ(result.blockDim, static_cast<uint64_t>(std::max<int64_t>(ncElements, 1)))
                << "rank=" << rank << ", zeroAxis=" << zeroAxis;
            EXPECT_EQ(result.data.units, ncElements) << "rank=" << rank << ", zeroAxis=" << zeroAxis;
            EXPECT_EQ(result.data.units * result.data.innerSize, 0) << "rank=" << rank << ", zeroAxis=" << zeroAxis;
        }
    }

    TilingCase multiZeroCase;
    multiZeroCase.xDims = {0, 3, 2, 0, 2, 0, 2, 2};
    multiZeroCase.gammaDims = {0};
    multiZeroCase.betaDims = {0};
    multiZeroCase.meanDims = {0};
    multiZeroCase.varianceDims = {0};
    const TilingResult multiZeroResult = RunTiling(multiZeroCase);
    ASSERT_EQ(multiZeroResult.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(multiZeroResult.hasData);
    EXPECT_EQ(multiZeroResult.data.units * multiZeroResult.data.innerSize, 0);
}

TEST_F(INInferV2TilingTest, rankAboveEightIsRejected)
{
    for (const size_t rank : {9U, 16U}) {
        TilingCase testCase;
        testCase.xDims.assign(rank, 1);
        testCase.xDims[1] = 3;
        testCase.xDims.back() = 5;
        testCase.gammaDims = {3};
        testCase.betaDims = {3};
        testCase.meanDims = {3};
        testCase.varianceDims = {3};
        const TilingResult result = RunTiling(testCase);
        EXPECT_EQ(result.status, ge::GRAPH_FAILED) << "rank=" << rank;
    }
}

TEST_F(INInferV2TilingTest, optionalBatchOutputsMayBeAbsent)
{
    for (const auto& outputPresence :
         {std::pair<bool, bool>{false, false}, {true, false}, {false, true}, {true, true}}) {
        TilingCase testCase;
        testCase.hasBatchMean = outputPresence.first;
        testCase.hasBatchVariance = outputPresence.second;
        const TilingResult result = RunTiling(testCase);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS)
            << "hasBatchMean=" << outputPresence.first << ", hasBatchVariance=" << outputPresence.second;
        ASSERT_TRUE(result.hasData);
        EXPECT_EQ(result.data.hasBatchMean, static_cast<int64_t>(outputPresence.first));
        EXPECT_EQ(result.data.hasBatchVar, static_cast<int64_t>(outputPresence.second));
    }
}

TEST_F(INInferV2TilingTest, nchwOriginFormatIsAccepted)
{
    TilingCase testCase;
    testCase.xFormat = ge::FORMAT_NCHW;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_SUCCESS);
}

TEST_F(INInferV2TilingTest, exactMinimumUbIsAccepted)
{
    TilingCase testCase;
    testCase.ubSize = RESERVED_UB + 1024;
    const TilingResult result = RunTiling(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(result.hasData);
    EXPECT_EQ(result.data.ubTileSize, 64);
}

TEST_F(INInferV2TilingTest, rejectsMissingCompileInfo)
{
    TilingCase testCase;
    testCase.hasPlatformInfo = false;
    testCase.hasCompileInfo = false;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsNonPositiveCoreCount)
{
    TilingCase testCase;
    testCase.coreNum = 0;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsNonPositiveUbSize)
{
    TilingCase testCase;
    testCase.ubSize = 0;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsMissingXDescriptor)
{
    TilingCase testCase;
    testCase.hasX = false;
    testCase.hasGamma = false;
    testCase.hasBeta = false;
    testCase.hasMean = false;
    testCase.hasVariance = false;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsUnsupportedXDataType)
{
    TilingCase testCase;
    testCase.xDtype = ge::DT_INT32;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsUnsupportedXFormat)
{
    TilingCase testCase;
    testCase.xFormat = ge::FORMAT_NHWC;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsRankBelowTwo)
{
    TilingCase testCase;
    testCase.xDims = {6};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsRankOneEmptyTensor)
{
    TilingCase testCase;
    testCase.xDims = {0};
    testCase.gammaDims = {0};
    testCase.betaDims = {0};
    testCase.meanDims = {0};
    testCase.varianceDims = {0};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsNegativeDimension)
{
    TilingCase testCase;
    testCase.xDims = {2, 3, -1};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsNcProductOverflow)
{
    TilingCase testCase;
    testCase.xDims = {std::numeric_limits<int64_t>::max(), 2, 1};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsTrailingProductOverflow)
{
    TilingCase testCase;
    testCase.xDims = {1, 1, std::numeric_limits<int64_t>::max(), 2};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsTotalElementOverflow)
{
    TilingCase testCase;
    testCase.xDims = {std::numeric_limits<int64_t>::max() / 2, 1, 3};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsMissingMean)
{
    TilingCase testCase;
    testCase.hasMean = false;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsMeanElementMismatch)
{
    TilingCase testCase;
    testCase.meanDims = {5};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsMissingVariance)
{
    TilingCase testCase;
    testCase.hasVariance = false;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsVarianceElementMismatch)
{
    TilingCase testCase;
    testCase.varianceDims = {5};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsGammaWithoutBeta)
{
    TilingCase testCase;
    testCase.hasBeta = false;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsBetaWithoutGamma)
{
    TilingCase testCase;
    testCase.hasGamma = false;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsGammaElementMismatch)
{
    TilingCase testCase;
    testCase.gammaDims = {5};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsBetaElementMismatch)
{
    TilingCase testCase;
    testCase.betaDims = {5};
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsUbBelowMinimum)
{
    TilingCase testCase;
    testCase.ubSize = RESERVED_UB + 1023;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsMissingRawTilingData)
{
    TilingCase testCase;
    testCase.hasTilingData = false;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsInsufficientRawTilingCapacity)
{
    TilingCase testCase;
    testCase.tilingCapacity = sizeof(INInferV2TilingData) - 1;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2TilingTest, rejectsMissingWorkspaceVector)
{
    TilingCase testCase;
    testCase.hasWorkspace = false;
    EXPECT_EQ(RunTiling(testCase).status, ge::GRAPH_FAILED);
}
