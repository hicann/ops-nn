/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_SPARSE_SEGMENT_SUM_UT : public testing::Test {};

auto CreateSparseSegmentSumNodeDef(const vector<vector<int64_t>>& shapes, const vector<DataType>& dataTypes,
                                   const vector<void*>& datas) -> decltype(CpuKernelUtils::CreateNodeDef())
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "SparseSegmentSum", "SparseSegmentSum")
        .Input({"x", dataTypes[0], shapes[0], datas[0]})
        .Input({"indices", dataTypes[1], shapes[1], datas[1]})
        .Input({"segment_ids", dataTypes[2], shapes[2], datas[2]})
        .Output({"y", dataTypes[3], shapes[3], datas[3]});
    return nodeDef;
}

uint64_t NumElements(const vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1;
    }
    return static_cast<uint64_t>(std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>()));
}

template <typename T, typename Index, typename SegmentId>
void RunSparseSegmentSumKernel(const vector<vector<int64_t>>& shapes, const vector<DataType>& dataTypes,
                               const T* inputX, const Index* inputIndices, const SegmentId* inputSegmentIds,
                               const T* expectOutput)
{
    const uint64_t xSize = NumElements(shapes[0]);
    const uint64_t indicesSize = NumElements(shapes[1]);
    const uint64_t segmentIdsSize = NumElements(shapes[2]);
    const uint64_t outputSize = NumElements(shapes[3]);

    auto xData = std::make_unique<T[]>(xSize);
    auto indicesData = std::make_unique<Index[]>(indicesSize);
    auto segmentIdsData = std::make_unique<SegmentId[]>(segmentIdsSize);
    auto outputData = std::make_unique<T[]>(outputSize);
    auto expect = std::make_unique<T[]>(outputSize);

    for (uint64_t i = 0; i < xSize; ++i) {
        xData[i] = inputX[i];
    }
    for (uint64_t i = 0; i < indicesSize; ++i) {
        indicesData[i] = inputIndices[i];
    }
    for (uint64_t i = 0; i < segmentIdsSize; ++i) {
        segmentIdsData[i] = inputSegmentIds[i];
    }
    for (uint64_t i = 0; i < outputSize; ++i) {
        outputData[i] = T();
        expect[i] = expectOutput[i];
    }

    vector<void*> datas = {static_cast<void*>(xData.get()), static_cast<void*>(indicesData.get()),
                           static_cast<void*>(segmentIdsData.get()), static_cast<void*>(outputData.get())};
    auto nodeDef = CreateSparseSegmentSumNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(outputData.get(), expect.get(), outputSize));
}

template <typename T>
void RunSparseSegmentSumTailAndGapCase(DataType dataType)
{
    constexpr int64_t kRows = 3;
    constexpr int64_t kColumns = 9;
    constexpr int64_t kOutputRows = 3;
    constexpr int64_t kIndexCount = 3;
    constexpr size_t kSecondRow = 2U;
    constexpr size_t kValueCycle = 7U;
    constexpr size_t kInputSize = static_cast<size_t>(kRows * kColumns);
    constexpr size_t kOutputSize = static_cast<size_t>(kOutputRows * kColumns);
    vector<T> x(kInputSize);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<T>(i % kValueCycle);
    }
    int32_t indices[kIndexCount] = {0, 2, 1};
    int32_t segmentIds[kIndexCount] = {0, 0, 2};
    vector<T> expected(kOutputSize, static_cast<T>(0));
    for (size_t column = 0; column < static_cast<size_t>(kColumns); ++column) {
        expected[column] = x[column] + x[kSecondRow * static_cast<size_t>(kColumns) + column];
        expected[kSecondRow * static_cast<size_t>(kColumns) + column] = static_cast<T>(0) +
                                                                        x[static_cast<size_t>(kColumns) + column];
    }

    vector<DataType> dataTypes = {dataType, DT_INT32, DT_INT32, dataType};
    vector<vector<int64_t>> shapes = {{kRows, kColumns}, {kIndexCount}, {kIndexCount}, {kOutputRows, kColumns}};
    RunSparseSegmentSumKernel(shapes, dataTypes, x.data(), indices, segmentIds, expected.data());
}

TEST_F(TEST_SPARSE_SEGMENT_SUM_UT, ALL_DTYPES_VECTOR_TAIL_AND_SEGMENT_GAP_SUCC)
{
    RunSparseSegmentSumTailAndGapCase<int8_t>(DT_INT8);
    RunSparseSegmentSumTailAndGapCase<int16_t>(DT_INT16);
    RunSparseSegmentSumTailAndGapCase<int32_t>(DT_INT32);
    RunSparseSegmentSumTailAndGapCase<int64_t>(DT_INT64);
    RunSparseSegmentSumTailAndGapCase<uint8_t>(DT_UINT8);
    RunSparseSegmentSumTailAndGapCase<uint16_t>(DT_UINT16);
    RunSparseSegmentSumTailAndGapCase<uint32_t>(DT_UINT32);
    RunSparseSegmentSumTailAndGapCase<uint64_t>(DT_UINT64);
    RunSparseSegmentSumTailAndGapCase<Eigen::half>(DT_FLOAT16);
    RunSparseSegmentSumTailAndGapCase<float>(DT_FLOAT);
    RunSparseSegmentSumTailAndGapCase<double>(DT_DOUBLE);
}

TEST_F(TEST_SPARSE_SEGMENT_SUM_UT, SIGNED_OVERFLOW_PACKET_AND_TAIL_SUCC)
{
    constexpr int64_t kRows = 2;
    constexpr int64_t kColumns = 5;
    constexpr int64_t kIndexCount = 2;
    constexpr int64_t kOutputRows = 1;
    constexpr int32_t kOne = 1;
    constexpr int32_t kTwo = 2;
    const int32_t max = std::numeric_limits<int32_t>::max();
    const int32_t min = std::numeric_limits<int32_t>::min();
    int32_t x[kRows * kColumns] = {max, max, min, min, kOne, 0, kOne, 0, -kOne, kTwo};
    int32_t indices[kIndexCount] = {0, 1};
    int32_t segmentIds[kIndexCount] = {0, 0};
    int32_t expected[kColumns] = {max, min, min, max, kOne + kTwo};
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{kRows, kColumns}, {kIndexCount}, {kIndexCount}, {kOutputRows, kColumns}};
    RunSparseSegmentSumKernel(shapes, dataTypes, x, indices, segmentIds, expected);
}

TEST_F(TEST_SPARSE_SEGMENT_SUM_UT, DATA_TYPE_DT_FLOAT_INT32_SUCC)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{3, 4}, {3}, {3}, {2, 4}};
    float x[12] = {1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8};
    int32_t indices[3] = {0, 2, 1};
    int32_t segmentIds[3] = {0, 0, 1};
    float expect[8] = {6, 8, 10, 12, 9, 10, 11, 12};
    RunSparseSegmentSumKernel(shapes, dataTypes, x, indices, segmentIds, expect);
}

TEST_F(TEST_SPARSE_SEGMENT_SUM_UT, DATA_TYPE_DT_INT64_INDICES_SEGMENT_IDS_SUCC)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{3, 2}, {3}, {3}, {2, 2}};
    int64_t x[6] = {1, 2, 9, 10, 5, 6};
    int64_t indices[3] = {0, 2, 1};
    int64_t segmentIds[3] = {0, 0, 1};
    int64_t expect[4] = {6, 8, 9, 10};
    RunSparseSegmentSumKernel(shapes, dataTypes, x, indices, segmentIds, expect);
}

TEST_F(TEST_SPARSE_SEGMENT_SUM_UT, FAILED_SHAPE_MISMATCH)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 4}, {2}, {3}, {1, 4}};
    double x[8] = {1.0};
    int32_t indices[2] = {0, 1};
    int32_t segmentIds[3] = {0, 0, 0};
    double y[4] = {0.0};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(y)};
    auto nodeDef = CreateSparseSegmentSumNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_SUM_UT, FAILED_DTYPE_MISMATCH)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 4}, {2}, {2}, {1, 4}};
    double x[8] = {1.0};
    int32_t indices[2] = {0, 1};
    int32_t segmentIds[2] = {0, 0};
    double y[4] = {0.0};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(y)};
    auto nodeDef = CreateSparseSegmentSumNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_SUM_UT, FAILED_UNSUPPORTED_X_DTYPE)
{
    vector<DataType> dataTypes = {DT_BOOL, DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 4}, {2}, {2}, {1, 4}};
    bool x[8] = {true};
    int32_t indices[2] = {0, 1};
    int32_t segmentIds[2] = {0, 0};
    bool y[4] = {false};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(y)};
    auto nodeDef = CreateSparseSegmentSumNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_SUM_UT, FAILED_INPUT_NULL)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 4}, {2}, {2}, {1, 4}};
    double x[8] = {1.0};
    int32_t segmentIds[2] = {0, 0};
    double y[4] = {0.0};
    vector<void*> datas = {static_cast<void*>(x), nullptr, static_cast<void*>(segmentIds), static_cast<void*>(y)};
    auto nodeDef = CreateSparseSegmentSumNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_SUM_UT, FAILED_NEGATIVE_INDEX)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 4}, {2}, {2}, {1, 4}};
    float x[8] = {1.0F};
    int32_t indices[2] = {0, -1};
    int32_t segmentIds[2] = {0, 0};
    float y[4] = {1.0F};
    vector<void*> datas = {x, indices, segmentIds, y};
    auto nodeDef = CreateSparseSegmentSumNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}
