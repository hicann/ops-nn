/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/arch35/apply_came_part3_tiling_arch35.h"

namespace ApplyCamePart3TilingUT {
namespace {
constexpr uint64_t kUbSize = 248UL * 1024UL;
constexpr uint64_t kCoreNum = 48UL;
constexpr uint64_t kTilingDataSize = 4096UL;
constexpr size_t kSystemWorkspaceSize = 16UL * 1024UL * 1024UL;
constexpr size_t kDetWorkspaceBytes = 1568UL;

struct TilingResult {
    TilingInfo info;
    optiling::ApplyCamePart3TilingData data;
};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

TilingResult RunTiling(int64_t n, int64_t m, ge::DataType momentType, bool useFirstMoment)
{
    const gert::StorageShape matrix = MakeStorageShape({n, m});
    const gert::StorageShape scalar = MakeStorageShape({1});
    const gert::StorageShape globalShape = MakeStorageShape({2});
    const gert::StorageShape sumR = MakeStorageShape({n});
    const gert::StorageShape sumC = MakeStorageShape({m});
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {matrix, ge::DT_FLOAT, ge::FORMAT_ND},      {matrix, momentType, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},      {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},      {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {globalShape, ge::DT_INT64, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {matrix, momentType, ge::FORMAT_ND},
        {sumR, ge::DT_FLOAT, ge::FORMAT_ND},
        {sumC, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"use_first_moment", Ops::NN::AnyValue::CreateFrom<bool>(useFirstMoment)},
    };
    optiling::ApplyCamePart3CompileInfo compileInfo{static_cast<int32_t>(kCoreNum), kUbSize};
    gert::TilingContextPara para("ApplyCamePart3", inputs, outputs, attrs, &compileInfo, kCoreNum, kUbSize,
                                 kTilingDataSize);

    TilingResult result;
    EXPECT_TRUE(ExecuteTiling(para, result.info));
    EXPECT_EQ(result.info.tilingDataSize, sizeof(result.data));
    if (result.info.tilingDataSize == sizeof(result.data)) {
        std::memcpy(&result.data, result.info.tilingData.get(), sizeof(result.data));
    }
    return result;
}

bool RunInvalidTiling(const std::vector<int64_t>& matrixDims, const std::vector<int64_t>& scalarDims,
                      const std::vector<int64_t>& globalShapeDims)
{
    const gert::StorageShape matrix = MakeStorageShape(matrixDims);
    const gert::StorageShape scalar = MakeStorageShape(scalarDims);
    const gert::StorageShape globalShape = MakeStorageShape(globalShapeDims);
    const gert::StorageShape sumR = MakeStorageShape({1});
    const gert::StorageShape sumC = MakeStorageShape({1});
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {matrix, ge::DT_FLOAT, ge::FORMAT_ND},      {matrix, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},      {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},      {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {globalShape, ge::DT_INT64, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {matrix, ge::DT_FLOAT, ge::FORMAT_ND},
        {sumR, ge::DT_FLOAT, ge::FORMAT_ND},
        {sumC, ge::DT_FLOAT, ge::FORMAT_ND},
        {MakeStorageShape({1}), ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"use_first_moment", Ops::NN::AnyValue::CreateFrom<bool>(false)},
    };
    optiling::ApplyCamePart3CompileInfo compileInfo{static_cast<int32_t>(kCoreNum), kUbSize};
    gert::TilingContextPara para("ApplyCamePart3", inputs, outputs, attrs, &compileInfo, kCoreNum, kUbSize,
                                 kTilingDataSize);
    TilingInfo info;
    return ExecuteTiling(para, info);
}

int64_t DivCeil(int64_t value, int64_t factor) { return value / factor + (value % factor == 0 ? 0 : 1); }

size_t ExpectedWorkspaceSize(const optiling::ApplyCamePart3TilingData& data)
{
    const size_t rOneCoreBlocks = static_cast<size_t>(DivCeil(data.rNumCalc, data.baseN));
    const int64_t rTailNumCalc = data.curN - data.rNumCalc * (data.rCoreNum - 1);
    const size_t rBlockCount = static_cast<size_t>(data.rCoreNum - 1) * rOneCoreBlocks +
                               static_cast<size_t>(DivCeil(rTailNumCalc, data.baseN));
    const size_t workspaceCElements = rBlockCount * static_cast<size_t>(data.curM);
    return kSystemWorkspaceSize + kDetWorkspaceBytes + workspaceCElements * sizeof(float);
}
} // namespace

TEST(ApplyCamePart3Tiling, FixedDetBufferParticipatesInUbBudget)
{
    const auto result = RunTiling(1, 1577, ge::DT_FLOAT, true);
    EXPECT_EQ(result.data.cCoreNum, 1);
    EXPECT_EQ(result.data.cNumCalc, 1577);
    EXPECT_EQ(result.data.baseN, 1);
    EXPECT_EQ(result.data.baseM, 832);
    ASSERT_EQ(result.info.workspaceSizes.size(), 1U);
    EXPECT_EQ(result.info.workspaceSizes[0], ExpectedWorkspaceSize(result.data));
}

TEST(ApplyCamePart3Tiling, UnalignedRowsSerializeOnlyWhenWritingFirstMoment)
{
    for (const auto momentType : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        constexpr int64_t m = 65;
        const auto writeMoment = RunTiling(8, m, momentType, true);
        EXPECT_EQ(writeMoment.data.cCoreNum, 1);
        EXPECT_EQ(writeMoment.data.cNumCalc, m);
        EXPECT_EQ(writeMoment.data.usedCoreNum, writeMoment.data.rCoreNum);

        const auto preserveMoment = RunTiling(8, m, momentType, false);
        EXPECT_GT(preserveMoment.data.cCoreNum, 1);
        EXPECT_LT(preserveMoment.data.cNumCalc, m);
        EXPECT_EQ(preserveMoment.data.usedCoreNum, preserveMoment.data.rCoreNum * preserveMoment.data.cCoreNum);
    }
}

TEST(ApplyCamePart3Tiling, AlignedRowsRetainColumnParallelism)
{
    for (const auto momentType : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        const int64_t m = momentType == ge::DT_FLOAT ? 256 : 512;
        for (const bool useFirstMoment : {false, true}) {
            const auto result = RunTiling(1, m, momentType, useFirstMoment);
            EXPECT_GT(result.data.cCoreNum, 1);
            EXPECT_EQ(result.data.usedCoreNum, result.data.cCoreNum);
        }
    }
}

TEST(ApplyCamePart3Tiling, WideShapesUseBoundedTiles)
{
    for (const auto& testCase :
         std::vector<std::pair<int64_t, int64_t>>{{144, 178858}, {1, 28000000}, {2, 28000000}, {1, 103000000}}) {
        const auto result = RunTiling(testCase.first, testCase.second, ge::DT_FLOAT, false);
        EXPECT_GT(result.data.usedCoreNum, 0);
        EXPECT_LE(result.data.usedCoreNum, static_cast<int64_t>(kCoreNum));
        EXPECT_LE(result.data.baseM, 255 * 8);
        EXPECT_GT(result.data.baseN, 0);
        ASSERT_EQ(result.info.workspaceSizes.size(), 1U);
        EXPECT_EQ(result.info.workspaceSizes[0], ExpectedWorkspaceSize(result.data));
    }
}

TEST(ApplyCamePart3Tiling, WorkspaceCUsesFullLogicalRowStride)
{
    const auto oneRow = RunTiling(1, 178858, ge::DT_BF16, false);
    const auto twoRows = RunTiling(2, 178858, ge::DT_BF16, false);
    ASSERT_EQ(oneRow.info.workspaceSizes.size(), 1U);
    ASSERT_EQ(twoRows.info.workspaceSizes.size(), 1U);
    EXPECT_EQ(oneRow.info.workspaceSizes[0], ExpectedWorkspaceSize(oneRow.data));
    EXPECT_EQ(twoRows.info.workspaceSizes[0], ExpectedWorkspaceSize(twoRows.data));

    const size_t oneRowBlocks = static_cast<size_t>(DivCeil(oneRow.data.curN, oneRow.data.baseN));
    const size_t twoRowBlocks = static_cast<size_t>(DivCeil(twoRows.data.curN, twoRows.data.baseN));
    EXPECT_GE(twoRowBlocks, oneRowBlocks);
    EXPECT_GE(twoRows.info.workspaceSizes[0], oneRow.info.workspaceSizes[0]);
}

TEST(ApplyCamePart3Tiling, SupportsDimensionsBeyondInt32WhenTensorBytesAreAddressable)
{
    constexpr int64_t kBeyondInt32 = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    const auto wide = RunTiling(1, kBeyondInt32, ge::DT_FLOAT, false);
    EXPECT_EQ(wide.data.curM, kBeyondInt32);
    EXPECT_GT(wide.data.cCoreNum, 1);

    const auto tall = RunTiling(kBeyondInt32, 1, ge::DT_FLOAT16, false);
    EXPECT_EQ(tall.data.curN, kBeyondInt32);
    EXPECT_GT(tall.data.rCoreNum, 1);
}

TEST(ApplyCamePart3Tiling, RejectsOnlyWhenFp32TensorByteRangeOverflows)
{
    constexpr int64_t kBeyondInt32 = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    EXPECT_FALSE(RunInvalidTiling({kBeyondInt32, kBeyondInt32}, {1}, {2}));
}

TEST(ApplyCamePart3Tiling, InvalidShapesReturnGraphFailed)
{
    EXPECT_FALSE(RunInvalidTiling({0, 4}, {1}, {2}));
    EXPECT_FALSE(RunInvalidTiling({4, 0}, {1}, {2}));
    EXPECT_FALSE(RunInvalidTiling({2, 4}, {2}, {2}));
    EXPECT_FALSE(RunInvalidTiling({2, 4}, {1}, {1}));
    EXPECT_FALSE(RunInvalidTiling({2, 4}, {1}, {3}));
}

} // namespace ApplyCamePart3TilingUT
