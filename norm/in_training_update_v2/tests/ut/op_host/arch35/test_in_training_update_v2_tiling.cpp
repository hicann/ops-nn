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
#include <securec.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/arch35/in_training_update_v2_tiling_arch35.h"
#include "../../../../op_kernel/arch35/in_training_update_v2_tiling_data.h"

namespace {
constexpr uint64_t CORE_NUM = 64U;
constexpr uint64_t UB_SIZE = 245760U;
constexpr uint64_t TILING_DATA_CAPACITY = 4096U;
constexpr int64_t CHANNEL_TILE = 64;
constexpr int64_t DMA_MAX_BLOCK_COUNT = 4095;
constexpr int64_t DMA_MAX_BLOCK_LENGTH_BYTES = (1LL << 21) - 1;
constexpr uint64_t ARTIFICIAL_LARGE_UB_SIZE_BYTES = 1ULL << 30;
constexpr int64_t FP32_CHANNELS_WITH_STRIDE_ABOVE_UINT32 = (1LL << 30) + CHANNEL_TILE;
constexpr int64_t FP32_CHANNELS_WITH_STRIDE_ABOVE_DMA40 = (1LL << 38) + CHANNEL_TILE;
constexpr int64_t NHWC_C65_FP16_ROW_STRIDE_ELEMS = 48;
constexpr int64_t TOTAL_ELEMENTS_OVERFLOW_DIM = 3037000500LL;
constexpr int64_t BYTE_EXTENT_OVERFLOW_DIM = std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(float)) +
                                             1;
constexpr int64_t NHWC_GUARD_ELEMS = 64;
constexpr int64_t X_Y_QUEUE_BUFFER_COUNT = 4;
constexpr int64_t STAT_BUFFER_COUNT = 8;
constexpr int64_t STAT_CHUNK = 64;

struct CaseConfig {
    std::vector<int64_t> xDims{2, 3, 4, 5};
    std::vector<int64_t> sumDims;
    std::vector<int64_t> squareDims;
    std::vector<int64_t> gammaDims;
    std::vector<int64_t> betaDims;
    std::vector<int64_t> meanDims;
    std::vector<int64_t> varianceDims;
    std::vector<int64_t> yDims;
    std::vector<int64_t> batchMeanDims;
    std::vector<int64_t> batchVarianceDims;
    ge::Format xOrigin = ge::FORMAT_NCHW;
    ge::Format outputOrigin = ge::FORMAT_RESERVED;
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType sumDtype = ge::DT_FLOAT;
    ge::DataType gammaDtype = ge::DT_FLOAT;
    bool hasGamma = false;
    bool hasBeta = false;
    bool hasMean = false;
    bool hasVariance = false;
    bool includeAttrs = true;
    uint64_t coreNum = CORE_NUM;
    uint64_t ubSize = UB_SIZE;
    uint64_t tilingDataCapacity = TILING_DATA_CAPACITY;
};

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

std::vector<int64_t> DefaultStatDims(const CaseConfig& config)
{
    if (config.xDims.size() != 4) {
        return {2, 3, 1, 1};
    }
    if (config.xOrigin == ge::FORMAT_NHWC) {
        return {config.xDims[0], 1, 1, config.xDims[3]};
    }
    return {config.xDims[0], config.xDims[1], 1, 1};
}

std::vector<int64_t> DefaultAffineDims(const CaseConfig& config, bool shared)
{
    const auto stats = DefaultStatDims(config);
    if (!shared) {
        return stats;
    }
    if (config.xOrigin == ge::FORMAT_NHWC) {
        return {1, 1, 1, stats[3]};
    }
    return {1, stats[1], 1, 1};
}

bool RunCase(const CaseConfig& config, TilingInfo& info)
{
    const auto statDims = config.sumDims.empty() ? DefaultStatDims(config) : config.sumDims;
    const auto squareDims = config.squareDims.empty() ? statDims : config.squareDims;
    const auto yDims = config.yDims.empty() ? config.xDims : config.yDims;
    const auto batchMeanDims = config.batchMeanDims.empty() ? statDims : config.batchMeanDims;
    const auto batchVarianceDims = config.batchVarianceDims.empty() ? statDims : config.batchVarianceDims;
    const ge::Format outputOrigin = config.outputOrigin == ge::FORMAT_RESERVED ? config.xOrigin : config.outputOrigin;

    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {MakeShape(config.xDims), config.xDtype, config.xOrigin, false, nullptr, config.xOrigin},
        {MakeShape(statDims), config.sumDtype, ge::FORMAT_ND, false, nullptr, config.xOrigin},
        {MakeShape(squareDims), ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr, config.xOrigin},
    };
    if (config.hasGamma) {
        const auto dims = config.gammaDims.empty() ? DefaultAffineDims(config, true) : config.gammaDims;
        inputs.push_back({MakeShape(dims), config.gammaDtype, ge::FORMAT_ND, false, nullptr, config.xOrigin});
    }
    if (config.hasBeta) {
        const auto dims = config.betaDims.empty() ? DefaultAffineDims(config, false) : config.betaDims;
        inputs.push_back({MakeShape(dims), ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr, config.xOrigin});
    }
    if (config.hasMean) {
        const auto dims = config.meanDims.empty() ? statDims : config.meanDims;
        inputs.push_back({MakeShape(dims), ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr, config.xOrigin});
    }
    if (config.hasVariance) {
        const auto dims = config.varianceDims.empty() ? statDims : config.varianceDims;
        inputs.push_back({MakeShape(dims), ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr, config.xOrigin});
    }

    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {MakeShape(yDims), config.xDtype, config.xOrigin, false, nullptr, outputOrigin},
        {MakeShape(batchMeanDims), ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr, outputOrigin},
        {MakeShape(batchVarianceDims), ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr, outputOrigin},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    if (config.includeAttrs) {
        attrs.push_back({"momentum", Ops::NN::AnyValue::CreateFrom<float>(0.25F)});
        attrs.push_back({"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1.0e-4F)});
    }
    const std::vector<uint32_t> inputInstances = {1,
                                                  1,
                                                  1,
                                                  config.hasGamma ? 1U : 0U,
                                                  config.hasBeta ? 1U : 0U,
                                                  config.hasMean ? 1U : 0U,
                                                  config.hasVariance ? 1U : 0U};
    const std::vector<uint32_t> outputInstances = {1, 1, 1};
    optiling::INTrainingUpdateV2CompileInfo compileInfo{static_cast<int64_t>(config.coreNum),
                                                        static_cast<int64_t>(config.ubSize)};
    gert::TilingContextPara para("INTrainingUpdateV2", inputs, outputs, attrs, inputInstances, outputInstances,
                                 &compileInfo, config.coreNum, config.ubSize, config.tilingDataCapacity);
    return ExecuteTiling(para, info);
}

INTrainingUpdateV2TilingData Data(const TilingInfo& info)
{
    INTrainingUpdateV2TilingData data{};
    if (info.tilingData == nullptr || info.tilingDataSize != sizeof(data)) {
        ADD_FAILURE() << "invalid tiling data buffer";
        return data;
    }
    const errno_t result = memcpy_s(&data, sizeof(data), info.tilingData.get(), sizeof(data));
    EXPECT_EQ(result, EOK);
    return data;
}

TEST(INTrainingUpdateV2Tiling, NchwFullPairsUseIndependentAffineBroadcastStrides)
{
    CaseConfig config;
    config.hasGamma = config.hasBeta = config.hasMean = config.hasVariance = true;
    config.gammaDims = {1, 3, 1, 1};
    config.betaDims = {2, 3, 1, 1};
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 1);
    EXPECT_EQ(info.blockNum, 15U);
    const auto& data = Data(info);
    EXPECT_EQ(data.unitBlocks, 15);
    EXPECT_EQ(data.rCores, 1);
    EXPECT_EQ(data.formerBlockNum, 0);
    EXPECT_EQ(data.formerUnits, 1);
    EXPECT_EQ(data.latterUnits, 1);
    EXPECT_EQ(data.n, 2);
    EXPECT_EQ(data.c, 3);
    EXPECT_EQ(data.r, 20);
    EXPECT_EQ(data.hasAffine, 1);
    EXPECT_EQ(data.hasRunning, 1);
    EXPECT_EQ(data.gammaBatchStride, 0);
    EXPECT_EQ(data.betaBatchStride, 3);
    EXPECT_EQ(data.xyBufferBytes, (data.tileElems + NHWC_GUARD_ELEMS) * static_cast<int64_t>(sizeof(float)));
    EXPECT_EQ(data.statBufferBytes, STAT_CHUNK * static_cast<int64_t>(sizeof(float)));
    EXPECT_LE(X_Y_QUEUE_BUFFER_COUNT * data.xyBufferBytes + STAT_BUFFER_COUNT * data.statBufferBytes,
              static_cast<int64_t>(UB_SIZE));
    const double invRExact = 1.0 / 20.0;
    EXPECT_FLOAT_EQ(data.invR, static_cast<float>(invRExact));
    EXPECT_FLOAT_EQ(data.invRCorrection, static_cast<float>(invRExact - static_cast<double>(data.invR)));
    EXPECT_FLOAT_EQ(data.momentum, 0.25F);
    EXPECT_FLOAT_EQ(data.epsilon, 1.0e-4F);
}

TEST(INTrainingUpdateV2Tiling, NhwcTailUsesTwoDimensionalDmaTile)
{
    CaseConfig config;
    config.xDims = {2, 3, 5, 65};
    config.xOrigin = ge::FORMAT_NHWC;
    config.xDtype = ge::DT_FLOAT16;
    config.hasGamma = config.hasBeta = true;
    config.gammaDims = {2, 1, 1, 65};
    config.betaDims = {1, 1, 1, 65};
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 2);
    const auto& data = Data(info);
    EXPECT_EQ(data.n, 2);
    EXPECT_EQ(data.c, 65);
    EXPECT_EQ(data.r, 15);
    EXPECT_EQ(data.rTile, std::min(DMA_MAX_BLOCK_COUNT, data.tileElems / NHWC_C65_FP16_ROW_STRIDE_ELEMS));
    EXPECT_GE(data.rTile, 1);
    EXPECT_EQ(data.gammaBatchStride, 65);
    EXPECT_EQ(data.betaBatchStride, 0);
}

TEST(INTrainingUpdateV2Tiling, NchwSmallPlanesUseOneOwnerForSharedGmBlock)
{
    CaseConfig config;
    config.xDims = {1, 2, 1, 1};
    config.xDtype = ge::DT_FLOAT;
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 1);
    EXPECT_EQ(info.blockNum, 1U);
    EXPECT_EQ(Data(info).unitBlocks, 1);
    EXPECT_EQ(Data(info).rCores, 1);
}

TEST(INTrainingUpdateV2Tiling, NhwcFp32TailAssignsWholeGmBlocksToCores)
{
    CaseConfig config;
    config.xOrigin = ge::FORMAT_NHWC;
    config.xDims = {1, 1, 1, 65};
    config.xDtype = ge::DT_FLOAT;
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 2);
    EXPECT_EQ(info.blockNum, 9U);
    EXPECT_EQ(Data(info).unitBlocks, 9);
    EXPECT_EQ(Data(info).rCores, 1);
}

TEST(INTrainingUpdateV2Tiling, NhwcFp16TailAssignsWholeGmBlocksToCores)
{
    CaseConfig config;
    config.xOrigin = ge::FORMAT_NHWC;
    config.xDims = {1, 1, 1, 65};
    config.xDtype = ge::DT_FLOAT16;
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 2);
    EXPECT_EQ(info.blockNum, 5U);
    EXPECT_EQ(Data(info).unitBlocks, 5);
    EXPECT_EQ(Data(info).rCores, 1);
}

TEST(INTrainingUpdateV2Tiling, MissingAllOptionalInputsSelectsPlainNormalization)
{
    TilingInfo info;
    ASSERT_TRUE(RunCase(CaseConfig{}, info));
    const auto& data = Data(info);
    EXPECT_EQ(data.hasAffine, 0);
    EXPECT_EQ(data.hasRunning, 0);
}

TEST(INTrainingUpdateV2Tiling, HalfPairsAreAcceptedAndOrphansAreNotShapeCoupled)
{
    CaseConfig config;
    config.xDims = {2, 5, 3, 3};
    config.hasGamma = true;
    config.gammaDims = {7, 5, 1, 1};
    config.hasMean = true;
    config.meanDims = {9, 5, 1, 1};
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    const auto& data = Data(info);
    EXPECT_EQ(data.hasAffine, 0);
    EXPECT_EQ(data.hasRunning, 0);
}

TEST(INTrainingUpdateV2Tiling, EmptyNUsesDedicatedNoAccessTemplate)
{
    CaseConfig config;
    config.xDims = {0, 3, 4, 5};
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(info.blockNum, 1U);
    EXPECT_EQ(Data(info).totalElements, 0);
}

TEST(INTrainingUpdateV2Tiling, EmptyNchwCUsesDedicatedNoAccessTemplate)
{
    CaseConfig config;
    config.xDims = {2, 0, 4, 5};
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(info.blockNum, 1U);
    EXPECT_EQ(Data(info).totalElements, 0);
}

TEST(INTrainingUpdateV2Tiling, EmptyNhwcNUsesDedicatedNoAccessTemplate)
{
    CaseConfig config;
    config.xOrigin = ge::FORMAT_NHWC;
    config.xDims = {0, 3, 4, 5};
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(info.blockNum, 1U);
    EXPECT_EQ(Data(info).totalElements, 0);
}

TEST(INTrainingUpdateV2Tiling, EmptyNchwNAndCUseDedicatedNoAccessTemplate)
{
    CaseConfig config;
    config.xDims = {0, 0, 4, 5};
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(info.blockNum, 1U);
    EXPECT_EQ(Data(info).totalElements, 0);
}

TEST(INTrainingUpdateV2Tiling, EmptyNhwcNAndCUseDedicatedNoAccessTemplate)
{
    CaseConfig config;
    config.xOrigin = ge::FORMAT_NHWC;
    config.xDims = {0, 4, 5, 0};
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(info.blockNum, 1U);
    EXPECT_EQ(Data(info).totalElements, 0);
}

TEST(INTrainingUpdateV2Tiling, EmptyNMayAlsoContainMultipleZeroSpatialAxes)
{
    CaseConfig config;
    config.xDims = {0, 3, 0, 0};
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(Data(info).totalElements, 0);
}

TEST(INTrainingUpdateV2Tiling, MissingAttributesUsePublicDefaults)
{
    CaseConfig config;
    config.includeAttrs = false;
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_FLOAT_EQ(Data(info).momentum, 0.1F);
    EXPECT_FLOAT_EQ(Data(info).epsilon, 1.0e-5F);
}

TEST(INTrainingUpdateV2Tiling, RejectsRankFiveEvenWhenShapesOtherwiseMatch)
{
    CaseConfig config;
    config.xDims = {2, 3, 4, 5, 6};
    config.sumDims = {2, 3, 1, 1};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsFormatOutsidePublicNchwNhwcContract)
{
    CaseConfig config;
    config.xOrigin = ge::FORMAT_NCDHW;
    config.xDims = {2, 3, 4, 5};
    config.sumDims = {2, 3, 1, 1};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsRequiredStatisticShapeMismatch)
{
    CaseConfig config;
    config.sumDims = {1, 3, 1, 1};
    config.squareDims = {2, 3, 1, 1};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsActiveAffineBatchOutsideOneOrN)
{
    CaseConfig config;
    config.xDims = {3, 4, 2, 2};
    config.hasGamma = config.hasBeta = true;
    config.gammaDims = {2, 4, 1, 1};
    config.betaDims = {1, 4, 1, 1};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsActiveRunningStatisticMismatch)
{
    CaseConfig config;
    config.hasMean = config.hasVariance = true;
    config.meanDims = {1, 3, 1, 1};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsInvalidOrphanDtype)
{
    CaseConfig config;
    config.hasGamma = true;
    config.gammaDtype = ge::DT_FLOAT16;
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsZeroSpatialDimensionForNonEmptyNc)
{
    CaseConfig config;
    config.xDims = {2, 3, 0, 5};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsNchwZeroWForNonEmptyNc)
{
    CaseConfig config;
    config.xDims = {2, 3, 4, 0};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsNhwcZeroHForNonEmptyNc)
{
    CaseConfig config;
    config.xOrigin = ge::FORMAT_NHWC;
    config.xDims = {2, 0, 4, 5};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsNhwcZeroWForNonEmptyNc)
{
    CaseConfig config;
    config.xOrigin = ge::FORMAT_NHWC;
    config.xDims = {2, 3, 0, 5};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsRankOneZeroShapeInsteadOfTreatingItAsEmpty)
{
    CaseConfig config;
    config.xDims = {0};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsUnresolvedDynamicDimensionAtTilingTime)
{
    CaseConfig config;
    config.xDims = {2, 3, -1, 5};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsOutputOriginFormatMismatch)
{
    CaseConfig config;
    config.outputOrigin = ge::FORMAT_NHWC;
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsUnsupportedXDataType)
{
    CaseConfig config;
    config.xDtype = ge::DT_BF16;
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsUbTooSmallForOneVectorTile)
{
    CaseConfig config;
    config.ubSize = 14000;
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, ClampsNchwTileToDmaBlockLengthLimit)
{
    CaseConfig config;
    config.ubSize = ARTIFICIAL_LARGE_UB_SIZE_BYTES;
    TilingInfo info;
    ASSERT_TRUE(RunCase(config, info));
    EXPECT_LE(Data(info).tileElems * static_cast<int64_t>(sizeof(float)), DMA_MAX_BLOCK_LENGTH_BYTES);
}

TEST(INTrainingUpdateV2Tiling, AcceptsNhwcGmStrideAboveUint32WhenInt64DmaSupportsIt)
{
    CaseConfig config;
    config.xOrigin = ge::FORMAT_NHWC;
    config.xDims = {1, 1, 1, FP32_CHANNELS_WITH_STRIDE_ABOVE_UINT32};
    TilingInfo info;
    EXPECT_TRUE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsNhwcGmStrideOutsideDmaField)
{
    CaseConfig config;
    config.xOrigin = ge::FORMAT_NHWC;
    config.xDims = {1, 1, 1, FP32_CHANNELS_WITH_STRIDE_ABOVE_DMA40};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsNcProductOutsideInt64)
{
    CaseConfig config;
    config.xDims = {std::numeric_limits<int64_t>::max(), 2, 1, 1};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsSpatialProductOutsideInt64)
{
    CaseConfig config;
    config.xDims = {1, 1, std::numeric_limits<int64_t>::max(), 2};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsTotalElementProductOutsideInt64)
{
    CaseConfig config;
    config.xDims = {TOTAL_ELEMENTS_OVERFLOW_DIM, 1, TOTAL_ELEMENTS_OVERFLOW_DIM, 1};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsXByteExtentOutsideInt64)
{
    CaseConfig config;
    config.xDims = {1, 1, 1, BYTE_EXTENT_OVERFLOW_DIM};
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsStatisticByteExtentOutsideInt64)
{
    CaseConfig config;
    config.xDims = {BYTE_EXTENT_OVERFLOW_DIM, 1, 1, 1};
    config.xDtype = ge::DT_FLOAT16;
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

TEST(INTrainingUpdateV2Tiling, RejectsTilingDataBufferThatIsTooSmall)
{
    CaseConfig config;
    config.tilingDataCapacity = sizeof(INTrainingUpdateV2TilingData) - 1;
    TilingInfo info;
    EXPECT_FALSE(RunCase(config, info));
}

} // namespace
