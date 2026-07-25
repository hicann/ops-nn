/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

class TEST_SOFTMAXV2_AICPU_UT : public testing::Test {};

auto CreateSoftmaxV2NodeDef(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types,
                            const vector<void*>& datas,
                            const vector<int64_t>& axes) -> decltype(CpuKernelUtils::CreateNodeDef())
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SoftmaxV2", "SoftmaxV2")
        .Input({"x", data_types[0], shapes[0], datas[0]})
        .Output({"y", data_types[1], shapes[1], datas[1]})
        .Attr("axes", axes);
    return node_def;
}

template <typename T>
T SoftmaxExp(T v)
{
    return Eigen::numext::exp(v);
}

template <typename T>
vector<T> SoftmaxRefSingleAxis(const vector<T>& input, const vector<int64_t>& dims, int64_t axis)
{
    int64_t dim_size = static_cast<int64_t>(dims.size());
    int64_t pivot = (axis >= 0) ? axis : dim_size + axis;
    int64_t inner_size = 1;
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim_size; i++) {
        if (i > pivot) {
            inner_size *= dims[i];
        }
        if (i < pivot) {
            outer_size *= dims[i];
        }
    }
    int64_t pivot_len = dims[pivot];
    vector<T> output(input.size());
    for (int64_t o = 0; o < outer_size; o++) {
        for (int64_t i = 0; i < inner_size; i++) {
            T maxv = input[o * pivot_len * inner_size + 0 * inner_size + i];
            for (int64_t p = 1; p < pivot_len; p++) {
                T v = input[o * pivot_len * inner_size + p * inner_size + i];
                if (v > maxv) {
                    maxv = v;
                }
            }
            T sum = T(0);
            for (int64_t p = 0; p < pivot_len; p++) {
                sum += SoftmaxExp(input[o * pivot_len * inner_size + p * inner_size + i] - maxv);
            }
            sum = max(sum, static_cast<T>(1e-10));
            for (int64_t p = 0; p < pivot_len; p++) {
                int64_t idx = o * pivot_len * inner_size + p * inner_size + i;
                output[idx] = SoftmaxExp(input[idx] - maxv) / sum;
            }
        }
    }
    return output;
}

template <typename T>
vector<T> SoftmaxRefMultiAxes(const vector<T>& input, int64_t outer_size, int64_t inner_size)
{
    vector<T> output(input.size());
    for (int64_t o = 0; o < outer_size; o++) {
        T maxv = input[o * inner_size];
        for (int64_t i = 1; i < inner_size; i++) {
            if (input[o * inner_size + i] > maxv) {
                maxv = input[o * inner_size + i];
            }
        }
        T sum = T(0);
        for (int64_t i = 0; i < inner_size; i++) {
            sum += SoftmaxExp(input[o * inner_size + i] - maxv);
        }
        sum = max(sum, static_cast<T>(1e-10));
        for (int64_t i = 0; i < inner_size; i++) {
            output[o * inner_size + i] = SoftmaxExp(input[o * inner_size + i] - maxv) / sum;
        }
    }
    return output;
}

template <typename T>
void RunSoftmaxV2Kernel(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types,
                        const vector<T>& input, const vector<T>& expect_output, const vector<int64_t>& axes,
                        uint32_t expect_status = KERNEL_STATUS_OK)
{
    auto calc_size = [](const vector<int64_t>& shape) -> uint64_t {
        return shape.empty() ? 1 : accumulate(shape.begin(), shape.end(), 1LL, multiplies<int64_t>());
    };

    const uint64_t x_size = calc_size(shapes[0]);
    const uint64_t y_size = calc_size(shapes[1]);

    auto x_data = make_unique<T[]>(x_size);
    auto output_data = make_unique<T[]>(y_size);

    for (uint64_t i = 0; i < x_size; ++i) {
        x_data[i] = input[i];
    }
    for (uint64_t i = 0; i < y_size; ++i) {
        output_data[i] = T();
    }

    vector<void*> datas = {static_cast<void*>(x_data.get()), static_cast<void*>(output_data.get())};
    auto node_def = CreateSoftmaxV2NodeDef(shapes, data_types, datas, axes);
    RUN_KERNEL(node_def, HOST, expect_status);

    if (expect_status == KERNEL_STATUS_OK) {
        auto expect = make_unique<T[]>(y_size);
        for (uint64_t i = 0; i < y_size; ++i) {
            expect[i] = expect_output[i];
        }
        EXPECT_TRUE(CompareResult(output_data.get(), expect.get(), y_size));
    }
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    vector<float> x = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    vector<int64_t> axes = {-1};
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT16_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    vector<Eigen::half> x = {Eigen::half(1.0), Eigen::half(2.0), Eigen::half(3.0),
                             Eigen::half(1.0), Eigen::half(1.0), Eigen::half(1.0)};
    vector<int64_t> axes = {-1};
    vector<Eigen::half> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, DOUBLE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    vector<double> x = {0.1, 0.2, 0.3, 1.0, 2.0, 3.0};
    vector<int64_t> axes = {-1};
    vector<double> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_AXIS0_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    vector<int64_t> axes = {0};
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, MULTI_AXES_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 4}};
    vector<int64_t> axes = {1, 2};
    int64_t total = 2 * 3 * 4;
    vector<float> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<float>(i % 7) * 0.1f;
    }
    int64_t inner_size = 3 * 4;
    int64_t outer_size = total / inner_size;
    vector<float> expect = SoftmaxRefMultiAxes(x, outer_size, inner_size);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {}};
    vector<float> x = {3.0f};
    vector<int64_t> axes = {-1};
    vector<float> expect = {1.0f};
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, EMPTY_TENSOR_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{0, 1}, {0, 1}};
    vector<void*> datas = {nullptr, nullptr};
    auto node_def = CreateSoftmaxV2NodeDef(shapes, data_types, datas, {-1});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_LARGE_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{64, 128}, {64, 128}};
    vector<int64_t> axes = {-1};
    int64_t total = 64 * 128;
    vector<float> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<float>(i % 11) * 0.1f;
    }
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT16_LARGE_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{32, 128}, {32, 128}};
    vector<int64_t> axes = {-1};
    int64_t total = 32 * 128;
    vector<Eigen::half> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = Eigen::half(static_cast<float>(i % 13) * 0.1f);
    }
    vector<Eigen::half> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, DOUBLE_LARGE_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{64, 128}, {64, 128}};
    vector<int64_t> axes = {-1};
    int64_t total = 64 * 128;
    vector<double> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<double>(i % 11) * 0.1;
    }
    vector<double> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_3D_MIDDLE_AXIS_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 4}};
    vector<int64_t> axes = {1};
    int64_t total = 2 * 3 * 4;
    vector<float> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<float>(i % 7) * 0.1f;
    }
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_3D_AXIS0_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 4}};
    vector<int64_t> axes = {0};
    int64_t total = 2 * 3 * 4;
    vector<float> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<float>(i % 7) * 0.1f;
    }
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_1D_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{5}, {5}};
    vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    vector<int64_t> axes = {-1};
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_AXIS1_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{3, 4}, {3, 4}};
    vector<int64_t> axes = {1};
    vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_NEGATIVE_AXIS_3D_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 4}};
    vector<int64_t> axes = {-2};
    int64_t total = 2 * 3 * 4;
    vector<float> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<float>(i % 7) * 0.1f;
    }
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, MULTI_AXES_FLOAT16_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 4}};
    vector<int64_t> axes = {1, 2};
    int64_t total = 2 * 3 * 4;
    vector<Eigen::half> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = Eigen::half(static_cast<float>(i % 7) * 0.1f);
    }
    int64_t inner_size = 3 * 4;
    int64_t outer_size = total / inner_size;
    vector<Eigen::half> expect = SoftmaxRefMultiAxes(x, outer_size, inner_size);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, MULTI_AXES_DOUBLE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 4}};
    vector<int64_t> axes = {1, 2};
    int64_t total = 2 * 3 * 4;
    vector<double> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<double>(i % 7) * 0.1;
    }
    int64_t inner_size = 3 * 4;
    int64_t outer_size = total / inner_size;
    vector<double> expect = SoftmaxRefMultiAxes(x, outer_size, inner_size);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, MULTI_AXES_NEGATIVE_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 4}};
    vector<int64_t> axes = {-2, -1};
    int64_t total = 2 * 3 * 4;
    vector<float> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<float>(i % 7) * 0.1f;
    }
    int64_t inner_size = 3 * 4;
    int64_t outer_size = total / inner_size;
    vector<float> expect = SoftmaxRefMultiAxes(x, outer_size, inner_size);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, MULTI_AXES_THREE_AXES_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 4}};
    vector<int64_t> axes = {0, 1, 2};
    int64_t total = 2 * 3 * 4;
    vector<float> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<float>(i % 7) * 0.1f;
    }
    int64_t inner_size = total;
    int64_t outer_size = 1;
    vector<float> expect = SoftmaxRefMultiAxes(x, outer_size, inner_size);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, MULTI_AXES_LARGE_PARALLEL_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{8, 32, 16}, {8, 32, 16}};
    vector<int64_t> axes = {1, 2};
    int64_t total = 8 * 32 * 16;
    vector<float> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = static_cast<float>(i % 11) * 0.1f;
    }
    int64_t inner_size = 32 * 16;
    int64_t outer_size = total / inner_size;
    vector<float> expect = SoftmaxRefMultiAxes(x, outer_size, inner_size);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_ALL_NEGATIVE_VALUES_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 4}, {2, 4}};
    vector<float> x = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};
    vector<int64_t> axes = {-1};
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT_LARGE_VALUES_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 4}, {2, 4}};
    vector<float> x = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
    vector<int64_t> axes = {-1};
    vector<float> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, FLOAT16_3D_AXIS1_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 4}};
    vector<int64_t> axes = {1};
    int64_t total = 2 * 3 * 4;
    vector<Eigen::half> x(total);
    for (int64_t i = 0; i < total; ++i) {
        x[i] = Eigen::half(static_cast<float>(i % 7) * 0.1f);
    }
    vector<Eigen::half> expect = SoftmaxRefSingleAxis(x, shapes[0], axes[0]);
    RunSoftmaxV2Kernel(shapes, data_types, x, expect, axes);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, INPUT_DTYPE_DISMATCH)
{
    vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    float x[6] = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    double output[6] = {0.0};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(output)};
    auto node_def = CreateSoftmaxV2NodeDef(shapes, data_types, datas, {-1});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, INPUT_SHAPE_DISMATCH)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 2}};
    float x[6] = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    float output[4] = {0.0f};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(output)};
    auto node_def = CreateSoftmaxV2NodeDef(shapes, data_types, datas, {-1});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, DTYPE_UNSUPPORT)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    bool x[6] = {false, true, false, true, false, true};
    bool output[6] = {false};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(output)};
    auto node_def = CreateSoftmaxV2NodeDef(shapes, data_types, datas, {-1});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, AXES_EMPTY_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    float x[6] = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    float output[6] = {0.0f};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(output)};
    auto node_def = CreateSoftmaxV2NodeDef(shapes, data_types, datas, {});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, AXES_OUT_RANGE)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    float x[6] = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    float output[6] = {0.0f};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(output)};
    auto node_def = CreateSoftmaxV2NodeDef(shapes, data_types, datas, {8});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SOFTMAXV2_AICPU_UT, NULL_INPUT_EXCEPTION)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
    float output[6] = {0.0f};
    vector<void*> datas = {static_cast<void*>(nullptr), static_cast<void*>(output)};
    auto node_def = CreateSoftmaxV2NodeDef(shapes, data_types, datas, {-1});
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
