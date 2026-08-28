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

#include "../../op_host/op_api/batch_matmul.cpp"
#include "../../op_host/op_api/batch_matmul_util.cpp"

// Stub for l0op functions required by batch_matmul_util.cpp but defined in external lib
namespace l0op {
bool MmCheckHitV3Shape(const aclTensor*, const aclTensor*, const aclTensor*, bool, bool, op::Format, bool)
{
    return false;
}
bool BmmCheckHitV3Shape(const aclTensor*, const aclTensor*, const aclTensor*, bool, bool, op::Format, op::Format, bool)
{
    return false;
}
} // namespace l0op

using namespace l0op;
using namespace op;

namespace {

aclTensor* MakeInputTensor(aclOpExecutor* executor, const std::initializer_list<int64_t>& dims, op::DataType dtype,
                           op::Format format)
{
    return executor->AllocTensor(op::Shape(dims), dtype, format);
}

} // namespace

// ===================== l0op 函数测试 =====================

class BatchMatmulL0OpTest : public testing::Test {
protected:
    void TestNzFp162Fp16(bool adj = false)
    {
        auto uniqueExecutor = CREATE_EXECUTOR();
        ASSERT_NE(uniqueExecutor.get(), nullptr);
        auto* executor = uniqueExecutor.get();
        auto* x1 = adj ?
                       MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16,
                                       op::Format::FORMAT_FRACTAL_NZ) :
                       MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
        auto* x2 = adj ?
                       MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16,
                                       op::Format::FORMAT_FRACTAL_NZ) :
                       MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
        auto* out = BatchMatMulNzFp162Fp16(x1, x2, nullptr, nullptr, adj, adj, false, 0, executor);
        EXPECT_NE(out, nullptr);
    }

    void TestNdFp162Fp32(bool adj = false)
    {
        auto uniqueExecutor = CREATE_EXECUTOR();
        ASSERT_NE(uniqueExecutor.get(), nullptr);
        auto* executor = uniqueExecutor.get();
        auto* x1 = adj ? MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND) :
                         MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
        auto* x2 = adj ? MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND) :
                         MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
        auto* out = BatchMatMulNdFp162Fp32(x1, x2, nullptr, nullptr, adj, adj, false, 0, executor);
        EXPECT_NE(out, nullptr);
    }

    void TestNzFp162Fp32(bool adj = false)
    {
        auto uniqueExecutor = CREATE_EXECUTOR();
        ASSERT_NE(uniqueExecutor.get(), nullptr);
        auto* executor = uniqueExecutor.get();
        auto* x1 = adj ?
                       MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16,
                                       op::Format::FORMAT_FRACTAL_NZ) :
                       MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
        auto* x2 = adj ?
                       MakeInputTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16,
                                       op::Format::FORMAT_FRACTAL_NZ) :
                       MakeInputTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
        auto* out = BatchMatMulNzFp162Fp32(x1, x2, nullptr, nullptr, adj, adj, false, 0, executor);
        EXPECT_NE(out, nullptr);
    }
};

TEST_F(BatchMatmulL0OpTest, BatchMatMulNzFp162Fp16) { TestNzFp162Fp16(); }
TEST_F(BatchMatmulL0OpTest, BatchMatMulNzFp162Fp16Adj) { TestNzFp162Fp16(true); }
TEST_F(BatchMatmulL0OpTest, BatchMatMulNdFp162Fp32) { TestNdFp162Fp32(); }
TEST_F(BatchMatmulL0OpTest, BatchMatMulNdFp162Fp32Adj) { TestNdFp162Fp32(true); }
TEST_F(BatchMatmulL0OpTest, BatchMatMulNzFp162Fp32) { TestNzFp162Fp32(); }
TEST_F(BatchMatmulL0OpTest, BatchMatMulNzFp162Fp32Adj) { TestNzFp162Fp32(true); }

// ===================== SetTensorToNDFormat 测试 =====================

class SetTensorToNDFormatTest : public testing::Test {};

TEST_F(SetTensorToNDFormatTest, AlreadyND)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* tensor = MakeInputTensor(uniqueExecutor.get(), {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* result = SetTensorToNDFormat(tensor);
    EXPECT_EQ(result, tensor);
    EXPECT_EQ(result->GetStorageFormat(), op::Format::FORMAT_ND);
}

TEST_F(SetTensorToNDFormatTest, AlreadyNz)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* tensor = MakeInputTensor(uniqueExecutor.get(), {2, 64, 128}, op::DataType::DT_FLOAT16,
                                   op::Format::FORMAT_FRACTAL_NZ);
    auto* result = SetTensorToNDFormat(tensor);
    EXPECT_EQ(result, tensor);
    EXPECT_EQ(result->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
}

TEST_F(SetTensorToNDFormatTest, OtherFormatToND)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* tensor = MakeInputTensor(uniqueExecutor.get(), {2, 64, 128}, op::DataType::DT_FLOAT16,
                                   op::Format::FORMAT_NCHW);
    auto* result = SetTensorToNDFormat(tensor);
    EXPECT_EQ(result, tensor);
    EXPECT_EQ(result->GetStorageFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(result->GetViewFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(result->GetOriginalFormat(), op::Format::FORMAT_ND);
}

// ===================== CheckShapeEqualToMul 测试 =====================

class CheckShapeEqualToMulTest : public testing::Test {};

TEST_F(CheckShapeEqualToMulTest, BatchTooSmall)
{
    // batchNum < 128 -> false
    EXPECT_FALSE(CheckShapeEqualToMul(64, 64, 64, 2, 16));
}

TEST_F(CheckShapeEqualToMulTest, NInBlockRange)
{
    // nDim > 32/dataSize && nDim <= 256/dataSize, dataSize=2 -> nDim > 16 && nDim <= 128 -> false
    EXPECT_FALSE(CheckShapeEqualToMul(64, 64, 128, 2, 16));
}

TEST_F(CheckShapeEqualToMulTest, NDimIsOne) { EXPECT_FALSE(CheckShapeEqualToMul(64, 1, 128, 2, 16)); }

TEST_F(CheckShapeEqualToMulTest, NDimAlignedTo256B)
{
    // nDim=256 (>128 so passes range check), nDim % (256/2) == 0 -> false
    EXPECT_FALSE(CheckShapeEqualToMul(2, 256, 128, 2, 16));
}

TEST_F(CheckShapeEqualToMulTest, AllConditionsPass)
{
    // batchNum=128 >= 128, nDim=8 not in (16,128], nDim!=1, UB check: (16+16+16*16)*2=576 <= 253952
    // nDim % (256/2) = 8 % 128 = 8 != 0 -> true
    EXPECT_TRUE(CheckShapeEqualToMul(2, 8, 128, 2, 16));
}
