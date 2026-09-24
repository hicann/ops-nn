/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/aicpu_test_utils.h"
#undef private
#undef protected

#include <cstdint>
#include <string>
#include <vector>

#include "Eigen/Core"

using namespace aicpu;
using std::string;
using std::vector;

class TestSparseSegmentMeanAicpu : public testing::Test {};

#define CREATE_NODEDEF(shapes, dataTypes, datas)                            \
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
    NodeDefBuilder(nodeDef.get(), "SparseSegmentMean", "SparseSegmentMean") \
        .Input({"x", (dataTypes)[0], (shapes)[0], (datas)[0]})              \
        .Input({"indices", (dataTypes)[1], (shapes)[1], (datas)[1]})        \
        .Input({"segment_ids", (dataTypes)[2], (shapes)[2], (datas)[2]})    \
        .Output({"y", (dataTypes)[3], (shapes)[3], (datas)[3]})

template <typename XType, typename IndexType, typename SegmentType>
void RunSuccessCase(DataType xType, DataType indexType, DataType segmentType)
{
    vector<DataType> dataTypes = {xType, indexType, segmentType, xType};
    vector<vector<int64_t>> shapes = {{3, 2}, {4}, {4}, {3, 2}};
    XType x[6] = {static_cast<XType>(1), static_cast<XType>(2), static_cast<XType>(3),
                  static_cast<XType>(4), static_cast<XType>(5), static_cast<XType>(6)};
    IndexType indices[4] = {0, 2, 1, 0};
    SegmentType segmentIds[4] = {0, 0, 1, 2};
    XType y[6] = {static_cast<XType>(0)};
    vector<void*> datas = {x, indices, segmentIds, y};
    CREATE_NODEDEF(shapes, dataTypes, datas);

    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);

    XType expect[6] = {static_cast<XType>(3), static_cast<XType>(4), static_cast<XType>(3),
                       static_cast<XType>(4), static_cast<XType>(1), static_cast<XType>(2)};
    EXPECT_EQ(CompareResult<XType>(y, expect, 6), true);
}

template <typename T>
void RunVectorTailAndGapCase(DataType dataType)
{
    constexpr int64_t kRows = 3;
    constexpr int64_t kColumns = 9;
    constexpr int64_t kOutputRows = 3;
    constexpr int64_t kIndexCount = 3;
    constexpr size_t kSecondRow = 2U;
    constexpr size_t kValueCycle = 11U;
    constexpr int32_t kValueOffset = 5;
    constexpr int32_t kMeanDivisor = 2;
    constexpr size_t kInputSize = static_cast<size_t>(kRows * kColumns);
    constexpr size_t kOutputSize = static_cast<size_t>(kOutputRows * kColumns);
    vector<T> x(kInputSize);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<T>(static_cast<int32_t>(i % kValueCycle) - kValueOffset);
    }
    vector<int32_t> indices = {0, 2, 1};
    vector<int32_t> segmentIds = {0, 0, 2};
    vector<T> y(kOutputSize, static_cast<T>(1));
    vector<T> expected(kOutputSize, static_cast<T>(0));
    for (size_t column = 0; column < static_cast<size_t>(kColumns); ++column) {
        expected[column] = (x[column] + x[kSecondRow * static_cast<size_t>(kColumns) + column]) /
                           static_cast<T>(kMeanDivisor);
        expected[kSecondRow * static_cast<size_t>(kColumns) + column] = static_cast<T>(0) +
                                                                        x[static_cast<size_t>(kColumns) + column];
    }

    vector<DataType> dataTypes = {dataType, DT_INT32, DT_INT32, dataType};
    vector<vector<int64_t>> shapes = {{kRows, kColumns}, {kIndexCount}, {kIndexCount}, {kOutputRows, kColumns}};
    vector<void*> datas = {x.data(), indices.data(), segmentIds.data(), y.data()};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(y.data(), expected.data(), y.size()));
}

TEST_F(TestSparseSegmentMeanAicpu, vector_tail_and_segment_gap_success)
{
    RunVectorTailAndGapCase<float>(DT_FLOAT);
    RunVectorTailAndGapCase<double>(DT_DOUBLE);
    RunVectorTailAndGapCase<Eigen::half>(DT_FLOAT16);
}

TEST_F(TestSparseSegmentMeanAicpu, float_int32_int32_success)
{
    RunSuccessCase<float, int32_t, int32_t>(DT_FLOAT, DT_INT32, DT_INT32);
}

TEST_F(TestSparseSegmentMeanAicpu, double_int64_int64_success)
{
    RunSuccessCase<double, int64_t, int64_t>(DT_DOUBLE, DT_INT64, DT_INT64);
}

TEST_F(TestSparseSegmentMeanAicpu, double_int32_int64_success)
{
    RunSuccessCase<double, int32_t, int64_t>(DT_DOUBLE, DT_INT32, DT_INT64);
}

TEST_F(TestSparseSegmentMeanAicpu, float16_int64_int32_success)
{
    RunSuccessCase<Eigen::half, int64_t, int32_t>(DT_FLOAT16, DT_INT64, DT_INT32);
}

TEST_F(TestSparseSegmentMeanAicpu, input_shape_mismatch_fail)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 2}, {2}, {3}, {1, 2}};
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    int32_t indices[2] = {0, 1};
    int32_t segmentIds[3] = {0, 0, 0};
    double y[2] = {0.0};
    vector<void*> datas = {x, indices, segmentIds, y};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TestSparseSegmentMeanAicpu, unsupported_x_dtype_fail)
{
    vector<DataType> dataTypes = {DT_BOOL, DT_INT32, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 2}, {2}, {2}, {1, 2}};
    bool x[4] = {true};
    int32_t indices[2] = {0, 1};
    int32_t segmentIds[2] = {0, 0};
    bool y[2] = {false};
    vector<void*> datas = {x, indices, segmentIds, y};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TestSparseSegmentMeanAicpu, unsupported_indices_dtype_fail)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_DOUBLE, DT_INT64, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 2}, {2}, {2}, {1, 2}};
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double indices[2] = {0.0, 1.0};
    int64_t segmentIds[2] = {0, 0};
    double y[2] = {0.0};
    vector<void*> datas = {x, indices, segmentIds, y};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TestSparseSegmentMeanAicpu, dtype_mismatch_fail)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 2}, {2}, {2}, {1, 2}};
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    int32_t indices[2] = {0, 1};
    int32_t segmentIds[2] = {0, 0};
    float y[2] = {0.0F};
    vector<void*> datas = {x, indices, segmentIds, y};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TestSparseSegmentMeanAicpu, segment_ids_unsorted_fail)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{3, 2}, {3}, {3}, {2, 2}};
    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int32_t indices[3] = {0, 1, 2};
    int32_t segmentIds[3] = {0, 1, 0};
    double y[4] = {0.0};
    vector<void*> datas = {x, indices, segmentIds, y};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TestSparseSegmentMeanAicpu, indices_out_of_range_fail)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 2}, {2}, {2}, {1, 2}};
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    int32_t indices[2] = {0, 2};
    int32_t segmentIds[2] = {0, 0};
    double y[2] = {0.0};
    vector<void*> datas = {x, indices, segmentIds, y};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TestSparseSegmentMeanAicpu, negative_indices_fail)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 2}, {2}, {2}, {1, 2}};
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    int32_t indices[2] = {0, -1};
    int32_t segmentIds[2] = {0, 0};
    double y[2] = {0.0};
    vector<void*> datas = {x, indices, segmentIds, y};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}
