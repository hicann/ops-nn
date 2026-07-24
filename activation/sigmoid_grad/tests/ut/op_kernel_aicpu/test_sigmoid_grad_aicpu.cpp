/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>
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

class TEST_SIGMOIDGRAD_AICPU_UT : public testing::Test {};

auto CreateSigmoidGradNodeDef(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types,
                              const vector<void*>& datas,
                              bool complex_conj) -> decltype(CpuKernelUtils::CreateNodeDef())
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SigmoidGrad", "SigmoidGrad")
        .Input({"y", data_types[0], shapes[0], datas[0]})
        .Input({"dy", data_types[1], shapes[1], datas[1]})
        .Output({"z", data_types[2], shapes[2], datas[2]})
        .Attr("complex_conj", complex_conj);
    return node_def;
}

template <typename T>
void RunSigmoidGradKernel(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types,
                          const vector<T>& input_y, const vector<T>& input_dy, const vector<T>& expect_output,
                          bool complex_conj = false, uint32_t expect_status = KERNEL_STATUS_OK)
{
    auto calc_size = [](const vector<int64_t>& shape) -> uint64_t {
        return shape.empty() ? 1 : accumulate(shape.begin(), shape.end(), 1LL, multiplies<int64_t>());
    };

    const uint64_t y_size = calc_size(shapes[0]);
    const uint64_t dy_size = calc_size(shapes[1]);
    const uint64_t z_size = calc_size(shapes[2]);

    auto y_data = make_unique<T[]>(y_size);
    auto dy_data = make_unique<T[]>(dy_size);
    auto output_data = make_unique<T[]>(z_size);

    for (uint64_t i = 0; i < y_size; ++i) {
        y_data[i] = input_y[i];
    }
    for (uint64_t i = 0; i < dy_size; ++i) {
        dy_data[i] = input_dy[i];
    }
    for (uint64_t i = 0; i < z_size; ++i) {
        output_data[i] = T();
    }

    vector<void*> datas = {static_cast<void*>(y_data.get()), static_cast<void*>(dy_data.get()),
                           static_cast<void*>(output_data.get())};
    auto node_def = CreateSigmoidGradNodeDef(shapes, data_types, datas, complex_conj);
    RUN_KERNEL(node_def, HOST, expect_status);

    if (expect_status == KERNEL_STATUS_OK) {
        auto expect = make_unique<T[]>(z_size);
        for (uint64_t i = 0; i < z_size; ++i) {
            expect[i] = expect_output[i];
        }
        EXPECT_TRUE(CompareResult(output_data.get(), expect.get(), z_size));
    }
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, FLOAT_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<float> y = {2.0f, 5.0f};
    vector<float> dy = {3.0f, 5.0f};
    vector<float> expect = {-6.0f, -100.0f};
    RunSigmoidGradKernel(shapes, data_types, y, dy, expect);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, FLOAT16_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<Eigen::half> y = {Eigen::half(2.0), Eigen::half(3.0)};
    vector<Eigen::half> dy = {Eigen::half(3.0), Eigen::half(2.0)};
    vector<Eigen::half> expect = {Eigen::half(-6.0), Eigen::half(-12.0)};
    RunSigmoidGradKernel(shapes, data_types, y, dy, expect);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, DOUBLE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
    vector<double> y = {0.5, 0.3, 0.8};
    vector<double> dy = {1.0, 2.0, 0.5};
    vector<double> expect(3);
    for (int i = 0; i < 3; ++i) {
        expect[i] = dy[i] * (1.0 - y[i]) * y[i];
    }
    RunSigmoidGradKernel(shapes, data_types, y, dy, expect);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, COMPLEX64_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<complex<float>> y = {{2.0f, 3.0f}, {3.0f, -2.0f}};
    vector<complex<float>> dy = {{2.0f, 3.0f}, {3.0f, -2.0f}};
    vector<complex<float>> expect(2);
    for (int i = 0; i < 2; ++i) {
        expect[i] = dy[i] * (complex<float>(1.0f) - y[i]) * y[i];
    }
    RunSigmoidGradKernel(shapes, data_types, y, dy, expect);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, COMPLEX128_CONJ_SUCC)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
    vector<complex<double>> y = {{4.0, 5.0}, {5.0, -6.0}};
    vector<complex<double>> dy = {{4.0, 5.0}, {5.0, -6.0}};
    vector<complex<double>> expect(2);
    for (int i = 0; i < 2; ++i) {
        expect[i] = conj((complex<double>(1.0) - y[i]) * y[i]) * dy[i];
    }
    RunSigmoidGradKernel(shapes, data_types, y, dy, expect, true);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, FLOAT_LARGE_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4, 2048}, {4, 2048}, {4, 2048}};
    vector<float> y(4 * 2048);
    vector<float> dy(4 * 2048);
    vector<float> expect(4 * 2048);
    for (int i = 0; i < 4 * 2048; ++i) {
        y[i] = static_cast<float>(i % 100) * 0.01f;
        dy[i] = static_cast<float>(2.0f);
        expect[i] = dy[i] * (1.0f - y[i]) * y[i];
    }
    RunSigmoidGradKernel(shapes, data_types, y, dy, expect);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, EMPTY_TENSOR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{0}, {0}, {0}};
    vector<void*> datas = {nullptr, nullptr, nullptr};
    auto node_def = CreateSigmoidGradNodeDef(shapes, data_types, datas, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, INPUT_DTYPE_DISMATCH)
{
    vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}};
    float y[1] = {1.0f};
    double dy[1] = {2.0};
    double output[1] = {0.0};
    vector<void*> datas = {static_cast<void*>(y), static_cast<void*>(dy), static_cast<void*>(output)};
    auto node_def = CreateSigmoidGradNodeDef(shapes, data_types, datas, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, DTYPE_UNSUPPORT)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}};
    bool y[1] = {false};
    bool dy[1] = {true};
    bool output[1] = {false};
    vector<void*> datas = {static_cast<void*>(y), static_cast<void*>(dy), static_cast<void*>(output)};
    auto node_def = CreateSigmoidGradNodeDef(shapes, data_types, datas, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, INPUT_SHAPE_DISMATCH)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 2}, {1, 3}, {1, 2}};
    float y[2] = {1.0f, 2.0f};
    float dy[3] = {1.0f, 2.0f, 3.0f};
    float output[2] = {0.0f};
    vector<void*> datas = {static_cast<void*>(y), static_cast<void*>(dy), static_cast<void*>(output)};
    auto node_def = CreateSigmoidGradNodeDef(shapes, data_types, datas, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOIDGRAD_AICPU_UT, NULL_INPUT_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1}, {1}, {1}};
    float output[1] = {0.0f};
    vector<void*> datas = {static_cast<void*>(nullptr), static_cast<void*>(nullptr), static_cast<void*>(output)};
    auto node_def = CreateSigmoidGradNodeDef(shapes, data_types, datas, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
