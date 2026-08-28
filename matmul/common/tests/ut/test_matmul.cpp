/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "opdev/op_executor.h"

#include "../../op_host/op_api/matmul.cpp"

// Stub InferShape to return success (overrides library version via --allow-multiple-definition)
aclnnStatus InferShape(uint32_t, op::OpArgList&, op::OpArgList&, op::OpArgList&) { return ACLNN_SUCCESS; }

using namespace l0op;
using namespace op;

namespace {

aclTensor* MakeInputTensor(aclOpExecutor* executor, const std::initializer_list<int64_t>& dims, op::DataType dtype,
                           op::Format format)
{
    return executor->AllocTensor(op::Shape(dims), dtype, format);
}

} // namespace

class MatMulL0OpTest : public testing::Test {};

TEST_F(MatMulL0OpTest, MatMulNdFp162Fp32)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* x2 = MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* out = MatMulNdFp162Fp32(x1, x2, nullptr, nullptr, false, false, false, 0, executor);
    EXPECT_NE(out, nullptr);
}

TEST_F(MatMulL0OpTest, MatMulNdFp162Fp32Adj)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* x2 = MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* out = MatMulNdFp162Fp32(x1, x2, nullptr, nullptr, true, true, false, 0, executor);
    EXPECT_NE(out, nullptr);
}

TEST_F(MatMulL0OpTest, MatMulNzNzNd)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* out = MatMulNzNzNd(x1, x2, nullptr, nullptr, false, false, false, 0, executor);
    EXPECT_NE(out, nullptr);
}

TEST_F(MatMulL0OpTest, MatMulNzNzNdAdj)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* out = MatMulNzNzNd(x1, x2, nullptr, nullptr, true, true, false, 0, executor);
    EXPECT_NE(out, nullptr);
}

TEST_F(MatMulL0OpTest, MatMulV3NzNzNd)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* out = MatMulV3NzNzNd(x1, x2, nullptr, false, false, false, 0, executor);
    EXPECT_NE(out, nullptr);
}

TEST_F(MatMulL0OpTest, MatMulV3NzNzNdAdj)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* out = MatMulV3NzNzNd(x1, x2, nullptr, true, true, false, 0, executor);
    EXPECT_NE(out, nullptr);
}

TEST_F(MatMulL0OpTest, MatMulV3NzNzNdFp162Fp32)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* out = MatMulV3NzNzNdFp162Fp32(x1, x2, nullptr, false, false, false, 0, executor);
    EXPECT_NE(out, nullptr);
}

TEST_F(MatMulL0OpTest, MatMulV3NzNzNdFp162Fp32Adj)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* out = MatMulV3NzNzNdFp162Fp32(x1, x2, nullptr, true, true, false, 0, executor);
    EXPECT_NE(out, nullptr);
}
