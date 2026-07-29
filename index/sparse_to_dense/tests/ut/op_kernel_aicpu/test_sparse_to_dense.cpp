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

#include <vector>
#include <string>

using namespace aicpu;
using std::string;
using std::vector;

class SparseToDenseAicpuTest : public testing::Test {};

#define CREATE_NODEDEF(shapes, dataTypes, datas, validate)           \
    auto node_def = CpuKernelUtils::CreateNodeDef();                 \
    NodeDefBuilder(node_def.get(), "SparseToDense", "SparseToDense") \
        .Input({"indices", dataTypes[0], shapes[0], datas[0]})       \
        .Input({"output_shape", dataTypes[1], shapes[1], datas[1]})  \
        .Input({"values", dataTypes[2], shapes[2], datas[2]})        \
        .Input({"default_value", dataTypes[3], shapes[3], datas[3]}) \
        .Output({"y", dataTypes[4], shapes[4], datas[4]})            \
        .Attr("validate_indices", validate)

TEST_F(SparseToDenseAicpuTest, Int32IndexFloatValue2D)
{
    int32_t indices[3][2] = {{0, 1}, {1, 0}, {2, 1}};
    int32_t outputShape[2] = {3, 2};
    float values[3] = {1.0F, 2.0F, 3.0F};
    float defaultValue = 9.0F;
    float output[6] = {0.0F};
    float expect[6] = {9.0F, 1.0F, 2.0F, 9.0F, 9.0F, 3.0F};
    vector<vector<int64_t>> shapes = {{3, 2}, {2}, {3}, {}, {3, 2}};
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expect, 6));
}

TEST_F(SparseToDenseAicpuTest, Int64IndexInt64Value)
{
    int64_t indices[2][2] = {{0, 0}, {1, 1}};
    int64_t outputShape[2] = {2, 2};
    int64_t values[2] = {4, 5};
    int64_t defaultValue = -1;
    int64_t output[4] = {0};
    int64_t expect[4] = {4, -1, -1, 5};
    vector<vector<int64_t>> shapes = {{2, 2}, {2}, {2}, {}, {2, 2}};
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expect, 4));
}

TEST_F(SparseToDenseAicpuTest, ScalarValuesBroadcast)
{
    int32_t indices[2] = {0, 2};
    int32_t outputShape[1] = {4};
    int32_t value = 7;
    int32_t defaultValue = 1;
    int32_t output[4] = {0};
    int32_t expect[4] = {7, 1, 7, 1};
    vector<vector<int64_t>> shapes = {{2}, {1}, {}, {}, {4}};
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<void*> datas = {indices, outputShape, &value, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expect, 4));
}

TEST_F(SparseToDenseAicpuTest, BoolValue)
{
    int64_t indices[1][2] = {{1, 0}};
    int64_t outputShape[2] = {2, 2};
    bool values[1] = {true};
    bool defaultValue = false;
    bool output[4] = {true, true, true, true};
    bool expect[4] = {false, false, true, false};
    vector<vector<int64_t>> shapes = {{1, 2}, {2}, {1}, {}, {2, 2}};
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_BOOL, DT_BOOL, DT_BOOL};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, expect, 4));
}

TEST_F(SparseToDenseAicpuTest, InvalidIndicesRank)
{
    int64_t indices[1][1][1] = {{{0}}};
    int64_t outputShape[1] = {1};
    int64_t values[1] = {1};
    int64_t defaultValue = 0;
    int64_t output[1] = {0};
    vector<vector<int64_t>> shapes = {{1, 1, 1}, {1}, {1}, {}, {1}};
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(SparseToDenseAicpuTest, InvalidOutputShapeRank)
{
    int64_t indices[1][2] = {{0, 0}};
    int64_t outputShape[1][2] = {{1, 1}};
    int64_t values[1] = {1};
    int64_t defaultValue = 0;
    int64_t output[1] = {0};
    vector<vector<int64_t>> shapes = {{1, 2}, {1, 2}, {1}, {}, {1, 1}};
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(SparseToDenseAicpuTest, InvalidValuesRank)
{
    int64_t indices[1][2] = {{0, 0}};
    int64_t outputShape[2] = {1, 1};
    int64_t values[1][1] = {{1}};
    int64_t defaultValue = 0;
    int64_t output[1] = {0};
    vector<vector<int64_t>> shapes = {{1, 2}, {2}, {1, 1}, {}, {1, 1}};
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(SparseToDenseAicpuTest, ValuesSizeMismatch)
{
    int32_t indices[2] = {0, 1};
    int32_t outputShape[1] = {2};
    int32_t values[1] = {3};
    int32_t defaultValue = 0;
    int32_t output[2] = {0};
    vector<vector<int64_t>> shapes = {{2}, {1}, {1}, {}, {2}};
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(SparseToDenseAicpuTest, InvalidIndexDtype)
{
    uint64_t indices[1] = {0};
    int64_t outputShape[1] = {1};
    int64_t values[1] = {1};
    int64_t defaultValue = 0;
    int64_t output[1] = {0};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}, {}, {1}};
    vector<DataType> dataTypes = {DT_UINT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(SparseToDenseAicpuTest, OutOfOrderIndices)
{
    int32_t indices[2] = {1, 0};
    int32_t outputShape[1] = {2};
    int32_t values[2] = {3, 4};
    int32_t defaultValue = 0;
    int32_t output[2] = {0};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}, {}, {2}};
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(SparseToDenseAicpuTest, RepeatedIndices)
{
    int32_t indices[2] = {1, 1};
    int32_t outputShape[1] = {2};
    int32_t values[2] = {3, 4};
    int32_t defaultValue = 0;
    int32_t output[2] = {0};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}, {}, {2}};
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(SparseToDenseAicpuTest, OutOfBoundsIndices)
{
    int32_t indices[1] = {2};
    int32_t outputShape[1] = {2};
    int32_t values[1] = {3};
    int32_t defaultValue = 0;
    int32_t output[2] = {0};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}, {}, {2}};
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(SparseToDenseAicpuTest, DefaultValueTypeMismatch)
{
    int32_t indices[1] = {0};
    int32_t outputShape[1] = {1};
    int32_t values[1] = {3};
    float defaultValue = 0.0F;
    int32_t output[1] = {0};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}, {}, {1}};
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(SparseToDenseAicpuTest, NegativeOutputShapeValue)
{
    int32_t indices[1] = {0};
    int64_t outputShape[1] = {-1};
    int32_t values[1] = {3};
    int32_t defaultValue = 0;
    int32_t output[1] = {0};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}, {}, {1}};
    vector<DataType> dataTypes = {DT_INT32, DT_INT64, DT_INT32, DT_INT32, DT_INT32};
    vector<void*> datas = {indices, outputShape, values, &defaultValue, output};
    CREATE_NODEDEF(shapes, dataTypes, datas, true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
