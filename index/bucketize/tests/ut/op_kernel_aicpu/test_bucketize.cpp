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
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_BUCKETIZE_UT : public testing::Test {};

template <typename T, typename T2>
void BucketizeCalcExpect(const NodeDef& nodeDef, T2 expectOut[], const std::vector<float>& bound, bool right)
{
    auto input0 = nodeDef.MutableInputs(0);
    T* input0Data = static_cast<T*>(input0->GetData());
    int64_t input0Num = input0->NumElements();
    for (int64_t i = 0; i < input0Num; ++i) {
        auto firstBiggerIt = right ? std::upper_bound(bound.begin(), bound.end(), input0Data[i]) :
                                     std::lower_bound(bound.begin(), bound.end(), input0Data[i]);
        expectOut[i] = static_cast<T2>(firstBiggerIt - bound.begin());
    }
}

#define CREATE_NODEDEF(shapes, dataTypes, datas, bound, dtype, right) \
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
    NodeDefBuilder(nodeDef.get(), "Bucketize", "Bucketize")           \
        .Input({"x", (dataTypes)[0], (shapes)[0], (datas)[0]})        \
        .Output({"y", (dataTypes)[1], (shapes)[1], (datas)[1]})       \
        .Attr("boundaries", (bound))                                  \
        .Attr("dtype", (dtype))                                       \
        .Attr("right", (right))

template <typename T1, typename T2>
void RunBucketizeKernel(const vector<DataType>& dataTypes, vector<vector<int64_t>>& shapes,
                        const std::vector<float>& bound, DataType dtype, bool right)
{
    uint64_t inputSize = CalTotalElements(shapes, 0);
    auto* input = new T1[inputSize];
    SetRandomValue<T1>(input, inputSize);

    uint64_t outputSize = CalTotalElements(shapes, 1);
    auto* output = new T2[outputSize];
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};

    CREATE_NODEDEF(shapes, dataTypes, datas, bound, dtype, right);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);

    auto* outputExp = new T2[outputSize];
    BucketizeCalcExpect<T1, T2>(*nodeDef.get(), outputExp, bound, right);

    EXPECT_EQ(CompareResult(output, outputExp, outputSize), true);
    delete[] input;
    delete[] output;
    delete[] outputExp;
}

TEST_F(TEST_BUCKETIZE_UT, INT32_TO_INT32_LEFT_SUCC)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    RunBucketizeKernel<int32_t, int32_t>(dataTypes, shapes, {0, 5, 10, 15}, DT_INT32, false);
}

TEST_F(TEST_BUCKETIZE_UT, DOUBLE_TO_INT64_RIGHT_SUCC)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    RunBucketizeKernel<double, int64_t>(dataTypes, shapes, {0, 5, 10, 15}, DT_INT64, true);
}

TEST_F(TEST_BUCKETIZE_UT, FLOAT_TO_INT32_RIGHT_SUCC)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    RunBucketizeKernel<float, int32_t>(dataTypes, shapes, {-1, 0, 1}, DT_INT32, true);
}

TEST_F(TEST_BUCKETIZE_UT, INT64_TO_INT64_LEFT_SUCC)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    RunBucketizeKernel<int64_t, int64_t>(dataTypes, shapes, {0, 5, 10, 15}, DT_INT64, false);
}

TEST_F(TEST_BUCKETIZE_UT, DUPLICATE_BOUNDARIES_LEFT_SUCC)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {3}};
    float input[3] = {1.0F, 2.0F, 3.0F};
    int32_t output[3] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, std::vector<float>({2.0F, 2.0F}), DT_INT32, false);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[3] = {0, 0, 2};
    EXPECT_EQ(CompareResult(output, outputExp, 3), true);
}

TEST_F(TEST_BUCKETIZE_UT, DUPLICATE_BOUNDARIES_RIGHT_SUCC)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {3}};
    float input[3] = {1.0F, 2.0F, 3.0F};
    int32_t output[3] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, std::vector<float>({2.0F, 2.0F}), DT_INT32, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[3] = {0, 2, 2};
    EXPECT_EQ(CompareResult(output, outputExp, 3), true);
}

TEST_F(TEST_BUCKETIZE_UT, UNSORTED_BOUNDARIES_FAIL)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {3}};
    int32_t input[3] = {1, 2, 3};
    int32_t output[3] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, std::vector<float>({3, 1, 2}), DT_INT32, false);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BUCKETIZE_UT, SHAPE_MISMATCH_FAIL)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {2}};
    int32_t input[3] = {1, 2, 3};
    int32_t output[2] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, std::vector<float>({1, 2, 3}), DT_INT32, false);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BUCKETIZE_UT, OUTPUT_TYPE_INVALID_FAIL)
{
    vector<DataType> dataTypes = {DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{3}, {3}};
    int32_t input[3] = {1, 2, 3};
    float output[3] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, std::vector<float>({1, 2, 3}), DT_FLOAT, false);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BUCKETIZE_UT, INPUT_TYPE_INVALID_FAIL)
{
    vector<DataType> dataTypes = {DT_BOOL, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {3}};
    bool input[3] = {true, false, true};
    int32_t output[3] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, std::vector<float>({1, 2, 3}), DT_INT32, false);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BUCKETIZE_UT, EMPTY_BOUNDARIES_SUCCESS)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {3}};
    int32_t input[3] = {-1, 0, 1};
    int32_t output[3] = {-1};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, std::vector<float>({}), DT_INT32, false);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int32_t outputExp[3] = {0, 0, 0};
    EXPECT_EQ(CompareResult(output, outputExp, 3), true);
}

TEST_F(TEST_BUCKETIZE_UT, EMPTY_TENSOR_SUCCESS)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{0}, {0}};
    int32_t input[1] = {0};
    int32_t output[1] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, std::vector<float>({1, 2, 3}), DT_INT32, false);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
}
