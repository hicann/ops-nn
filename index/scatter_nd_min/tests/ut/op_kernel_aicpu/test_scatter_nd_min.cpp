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

class TEST_SCATTER_ND_MIN_UT : public testing::Test {};

auto CreateScatterNdMinNodeDef(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types,
                               const vector<void*>& datas) -> decltype(CpuKernelUtils::CreateNodeDef())
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "ScatterNdMin", "ScatterNdMin")
        .Input({"var", data_types[0], shapes[0], datas[0]})
        .Input({"indices", data_types[1], shapes[1], datas[1]})
        .Input({"updates", data_types[2], shapes[2], datas[2]})
        .Output({"var", data_types[3], shapes[3], datas[3]});
    return node_def;
}

template <typename T, typename Index>
void RunScatterNdMinKernel(const vector<vector<int64_t>>& shapes, const vector<DataType>& data_types, const T* input_x,
                           const Index* input_indices, const T* input_updates, const T* expect_output)
{
    const uint64_t input_x_size = shapes[0].empty() ? 0 :
                                                      std::accumulate(shapes[0].begin(), shapes[0].end(), 1LL,
                                                                      std::multiplies<int64_t>());
    const uint64_t input_indices_size = shapes[1].empty() ? 0 :
                                                            std::accumulate(shapes[1].begin(), shapes[1].end(), 1LL,
                                                                            std::multiplies<int64_t>());
    const uint64_t input_updates_size = shapes[2].empty() ? 0 :
                                                            std::accumulate(shapes[2].begin(), shapes[2].end(), 1LL,
                                                                            std::multiplies<int64_t>());
    const uint64_t output_size = shapes[3].empty() ? 0 :
                                                     std::accumulate(shapes[3].begin(), shapes[3].end(), 1LL,
                                                                     std::multiplies<int64_t>());

    auto x_data = std::make_unique<T[]>(input_x_size);
    auto indices_data = std::make_unique<Index[]>(input_indices_size);
    auto updates_data = std::make_unique<T[]>(input_updates_size);
    auto output_data = std::make_unique<T[]>(output_size);
    auto expect = std::make_unique<T[]>(output_size);

    for (uint64_t i = 0; i < input_x_size; ++i) {
        x_data[i] = input_x[i];
    }
    for (uint64_t i = 0; i < input_indices_size; ++i) {
        indices_data[i] = input_indices[i];
    }
    for (uint64_t i = 0; i < input_updates_size; ++i) {
        updates_data[i] = input_updates[i];
    }
    for (uint64_t i = 0; i < output_size; ++i) {
        output_data[i] = T();
        expect[i] = expect_output[i];
    }

    vector<void*> datas = {static_cast<void*>(x_data.get()), static_cast<void*>(indices_data.get()),
                           static_cast<void*>(updates_data.get()), static_cast<void*>(output_data.get())};

    auto node_def = CreateScatterNdMinNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_TRUE(CompareResult(output_data.get(), expect.get(), output_size));
}

TEST_F(TEST_SCATTER_ND_MIN_UT, DATA_TYPE_DT_FLOAT_INT32_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{8}, {4, 1}, {4}, {8}};
    float input_x[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    int32_t input_indices[4] = {4, 3, 1, 7};
    float input_updates[4] = {0, 1, 2, 3};
    float expect_output[8] = {8, 2, 6, 1, 0, 3, 2, 1};
    RunScatterNdMinKernel(shapes, data_types, input_x, input_indices, input_updates, expect_output);
}

TEST_F(TEST_SCATTER_ND_MIN_UT, DATA_TYPE_DT_INT32_INT64_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 1}, {2, 3}, {2, 3}};
    int32_t input_x[6] = {10, 7, 11, 4, 5, 9};
    int64_t input_indices[2] = {0, 1};
    int32_t input_updates[6] = {1, 8, 3, 5, 2, 6};
    int32_t expect_output[6] = {1, 7, 3, 4, 2, 6};
    RunScatterNdMinKernel(shapes, data_types, input_x, input_indices, input_updates, expect_output);
}

TEST_F(TEST_SCATTER_ND_MIN_UT, FAILED_INDICES_TYPE)
{
    vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{8}, {4, 1}, {4}, {8}};
    float input_x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    double input_indices[4] = {4.2, 3.5, 1.7, 7.4};
    float input_updates[4] = {9, 10, 11, 12};
    float output[8] = {0};
    vector<void*> datas = {static_cast<void*>(input_x), static_cast<void*>(input_indices),
                           static_cast<void*>(input_updates), static_cast<void*>(output)};
    auto node_def = CreateScatterNdMinNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SCATTER_ND_MIN_UT, FAILED_OUT_OF_BOUNDS_INDEX)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{8}, {4, 1}, {4}, {8}};
    float input_x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int64_t input_indices[4] = {4, -1, -2, 7};
    float input_updates[4] = {9, 10, 11, 12};
    float output[8] = {0};
    vector<void*> datas = {static_cast<void*>(input_x), static_cast<void*>(input_indices),
                           static_cast<void*>(input_updates), static_cast<void*>(output)};
    auto node_def = CreateScatterNdMinNodeDef(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
