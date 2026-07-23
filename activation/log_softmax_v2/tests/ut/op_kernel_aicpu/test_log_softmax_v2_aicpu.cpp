/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
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

class TEST_LOGSOFTMAXV2_AICPU_UT : public testing::Test {};

auto CreateLogSoftmaxV2NodeDef(const std::vector<std::vector<int64_t>>& shapes, const std::vector<DataType>& data_types,
                               const std::vector<void*>& datas,
                               const std::vector<int64_t>& axes) -> decltype(CpuKernelUtils::CreateNodeDef())
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "LogSoftmaxV2", "LogSoftmaxV2")
        .Input({"logits", data_types[0], shapes[0], datas[0]})
        .Output({"logsoftmax", data_types[1], shapes[1], datas[1]})
        .Attr("axes", axes);
    return node_def;
}

template <typename T>
void RunLogSoftmaxV2Kernel(const std::vector<std::vector<int64_t>>& shapes, const std::vector<DataType>& data_types,
                           const std::vector<T>& input_data, const std::vector<T>& expect_output,
                           const std::vector<int64_t>& axes, uint32_t expect_status = KERNEL_STATUS_OK)
{
    auto calc_size = [](const std::vector<int64_t>& shape) -> uint64_t {
        return shape.empty() ? 1 : std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    };

    const uint64_t input_size = calc_size(shapes[0]);
    const uint64_t output_size = calc_size(shapes[1]);

    auto logits_data = std::make_unique<T[]>(input_size);
    auto output_data = std::make_unique<T[]>(output_size);

    for (uint64_t i = 0; i < input_size; ++i) {
        logits_data[i] = input_data[i];
    }
    for (uint64_t i = 0; i < output_size; ++i) {
        output_data[i] = T();
    }

    std::vector<void*> datas = {static_cast<void*>(logits_data.get()), static_cast<void*>(output_data.get())};
    auto node_def = CreateLogSoftmaxV2NodeDef(shapes, data_types, datas, axes);
    RUN_KERNEL(node_def, HOST, expect_status);

    if (expect_status == KERNEL_STATUS_OK) {
        auto expect = std::make_unique<T[]>(output_size);
        for (uint64_t i = 0; i < output_size; ++i) {
            expect[i] = expect_output[i];
        }
        EXPECT_TRUE(CompareResult(output_data.get(), expect.get(), output_size));
    }
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, DATA_TYPE_FLOAT_SUCC)
{
    std::vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    std::vector<std::vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> expect(6);
    for (int row = 0; row < 2; ++row) {
        float max_val = input[row * 3];
        for (int j = 1; j < 3; ++j) {
            max_val = std::max(max_val, input[row * 3 + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < 3; ++j) {
            sum += expf(input[row * 3 + j] - max_val);
        }
        float log_sum = logf(sum);
        for (int j = 0; j < 3; ++j) {
            expect[row * 3 + j] = input[row * 3 + j] - max_val - log_sum;
        }
    }
    RunLogSoftmaxV2Kernel(shapes, data_types, input, expect, {-1});
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, DATA_TYPE_FLOAT16_SUCC)
{
    std::vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
    std::vector<std::vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    std::vector<Eigen::half> input = {Eigen::half(1.0), Eigen::half(2.0), Eigen::half(3.0),
                                      Eigen::half(1.0), Eigen::half(2.0), Eigen::half(3.0)};
    std::vector<Eigen::half> expect(6);
    for (int row = 0; row < 2; ++row) {
        float max_val = static_cast<float>(input[row * 3]);
        for (int j = 1; j < 3; ++j) {
            max_val = std::max(max_val, static_cast<float>(input[row * 3 + j]));
        }
        float sum = 0.0f;
        for (int j = 0; j < 3; ++j) {
            sum += expf(static_cast<float>(input[row * 3 + j]) - max_val);
        }
        float log_sum = logf(sum);
        for (int j = 0; j < 3; ++j) {
            expect[row * 3 + j] = Eigen::half(static_cast<float>(input[row * 3 + j]) - max_val - log_sum);
        }
    }
    RunLogSoftmaxV2Kernel(shapes, data_types, input, expect, {-1});
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, DATA_TYPE_DOUBLE_SUCC)
{
    std::vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    std::vector<std::vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    std::vector<double> input = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    std::vector<double> expect(6);
    for (int row = 0; row < 2; ++row) {
        double max_val = input[row * 3];
        for (int j = 1; j < 3; ++j) {
            max_val = std::max(max_val, input[row * 3 + j]);
        }
        double sum = 0.0;
        for (int j = 0; j < 3; ++j) {
            sum += std::exp(input[row * 3 + j] - max_val);
        }
        double log_sum = std::log(sum);
        for (int j = 0; j < 3; ++j) {
            expect[row * 3 + j] = input[row * 3 + j] - max_val - log_sum;
        }
    }
    RunLogSoftmaxV2Kernel(shapes, data_types, input, expect, {-1});
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, FLOAT_AXIS_0_SUCC)
{
    std::vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    std::vector<std::vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> expect(6);
    for (int col = 0; col < 3; ++col) {
        float max_val = std::max(input[col], input[3 + col]);
        float sum = expf(input[col] - max_val) + expf(input[3 + col] - max_val);
        float log_sum = logf(sum);
        expect[col] = input[col] - max_val - log_sum;
        expect[3 + col] = input[3 + col] - max_val - log_sum;
    }
    RunLogSoftmaxV2Kernel(shapes, data_types, input, expect, {0});
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, FLOAT_LARGE_PARALLEL_PATH)
{
    std::vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    std::vector<std::vector<int64_t>> shapes = {{4, 1025}, {4, 1025}};
    std::vector<float> input(4 * 1025);
    std::vector<float> expect(4 * 1025);
    for (int row = 0; row < 4; ++row) {
        float max_val = 0.0f;
        for (int j = 0; j < 1025; ++j) {
            input[row * 1025 + j] = static_cast<float>(j) * 0.001f;
            max_val = std::max(max_val, input[row * 1025 + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < 1025; ++j) {
            sum += expf(input[row * 1025 + j] - max_val);
        }
        float log_sum = logf(sum);
        for (int j = 0; j < 1025; ++j) {
            expect[row * 1025 + j] = input[row * 1025 + j] - max_val - log_sum;
        }
    }
    RunLogSoftmaxV2Kernel(shapes, data_types, input, expect, {-1});
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, DOUBLE_SCALAR_SUCC)
{
    std::vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    std::vector<std::vector<int64_t>> shapes = {{}, {}};
    std::vector<double> input = {3.14};
    std::vector<double> expect = {0.0};
    RunLogSoftmaxV2Kernel(shapes, data_types, input, expect, {0});
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, AXES_NUM_EXCEPTION)
{
    std::vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    std::vector<std::vector<int64_t>> shapes = {{4, 5, 6}, {4, 5, 6}};
    float input[120] = {1.0f};
    float output[120] = {0.0f};
    std::vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    auto node_def = CreateLogSoftmaxV2NodeDef(shapes, data_types, datas, {5, 3});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, AXES_OUT_OF_RANGE_EXCEPTION)
{
    std::vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    std::vector<std::vector<int64_t>> shapes = {{4, 5, 6}, {4, 5, 6}};
    float input[120] = {1.0f};
    float output[120] = {0.0f};
    std::vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    auto node_def = CreateLogSoftmaxV2NodeDef(shapes, data_types, datas, {3});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, INPUT_DTYPE_EXCEPTION)
{
    std::vector<DataType> data_types = {DT_INT32, DT_INT32};
    std::vector<std::vector<int64_t>> shapes = {{4, 5, 6}, {4, 5, 6}};
    int32_t input[120] = {1};
    int32_t output[120] = {0};
    std::vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    auto node_def = CreateLogSoftmaxV2NodeDef(shapes, data_types, datas, {1});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, INPUT_NULL_EXCEPTION)
{
    std::vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    std::vector<std::vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
    double output[60] = {0.0};
    std::vector<void*> datas = {static_cast<void*>(nullptr), static_cast<void*>(output)};
    auto node_def = CreateLogSoftmaxV2NodeDef(shapes, data_types, datas, {1});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, FLOAT_NEGATIVE_VALUES_AXIS_LAST)
{
    std::vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    std::vector<std::vector<int64_t>> shapes = {{3, 4}, {3, 4}};
    std::vector<float> input = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> expect(12);
    for (int row = 0; row < 3; ++row) {
        float max_val = input[row * 4];
        for (int j = 1; j < 4; ++j) {
            max_val = std::max(max_val, input[row * 4 + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < 4; ++j) {
            sum += expf(input[row * 4 + j] - max_val);
        }
        float log_sum = logf(sum);
        for (int j = 0; j < 4; ++j) {
            expect[row * 4 + j] = input[row * 4 + j] - max_val - log_sum;
        }
    }
    RunLogSoftmaxV2Kernel(shapes, data_types, input, expect, {-1});
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, DOUBLE_3D_MIDDLE_AXIS)
{
    std::vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    std::vector<std::vector<int64_t>> shapes = {{2, 3, 2}, {2, 3, 2}};
    std::vector<double> input = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0};
    std::vector<double> expect(12);
    for (int batch = 0; batch < 2; ++batch) {
        for (int col = 0; col < 2; ++col) {
            double max_val = input[batch * 6 + col];
            for (int j = 1; j < 3; ++j) {
                max_val = std::max(max_val, input[batch * 6 + j * 2 + col]);
            }
            double sum = 0.0;
            for (int j = 0; j < 3; ++j) {
                sum += std::exp(input[batch * 6 + j * 2 + col] - max_val);
            }
            double log_sum = std::log(sum);
            for (int j = 0; j < 3; ++j) {
                expect[batch * 6 + j * 2 + col] = input[batch * 6 + j * 2 + col] - max_val - log_sum;
            }
        }
    }
    RunLogSoftmaxV2Kernel(shapes, data_types, input, expect, {1});
}

TEST_F(TEST_LOGSOFTMAXV2_AICPU_UT, FLOAT_LARGE_PARALLEL_NEGATIVE)
{
    std::vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    std::vector<std::vector<int64_t>> shapes = {{4, 2048}, {4, 2048}};
    std::vector<float> input(4 * 2048);
    std::vector<float> expect(4 * 2048);
    for (int i = 0; i < 4 * 2048; ++i) {
        input[i] = static_cast<float>((i % 100) - 50) * 0.1f;
    }
    for (int row = 0; row < 4; ++row) {
        float max_val = input[row * 2048];
        for (int j = 1; j < 2048; ++j) {
            max_val = std::max(max_val, input[row * 2048 + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < 2048; ++j) {
            sum += expf(input[row * 2048 + j] - max_val);
        }
        float log_sum = logf(sum);
        for (int j = 0; j < 2048; ++j) {
            expect[row * 2048 + j] = input[row * 2048 + j] - max_val - log_sum;
        }
    }
    RunLogSoftmaxV2Kernel(shapes, data_types, input, expect, {-1});
}
