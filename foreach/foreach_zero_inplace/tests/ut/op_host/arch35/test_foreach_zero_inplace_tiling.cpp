/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/foreach_zero_inplace_tiling_data.h"

namespace {
using TensorDescription = gert::TilingContextPara::TensorDescription;

struct ForeachZeroInplaceCompileInfo {};

void RunTilingCase(const std::vector<TensorDescription>& inputs, const std::vector<TensorDescription>& outputs,
                   const std::vector<uint32_t>& inputInstances, const std::vector<uint32_t>& outputInstances,
                   ge::graphStatus expectedStatus, uint64_t expectedKey = 0)
{
    ForeachZeroInplaceCompileInfo compileInfo;
    gert::TilingContextPara context("ForeachZeroInplace", inputs, outputs, {}, inputInstances, outputInstances,
                                    &compileInfo, 64, 262144, 4096);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        ExecuteTestCase(context, expectedStatus, expectedKey, EMPTY_EXPECT_TILING_DATA, {0});
        return;
    }
    ExecuteTestCase(context, expectedStatus);
}
} // namespace

TEST(ForeachZeroInplaceTilingTest, accepts_matching_dynamic_ref_outputs)
{
    std::vector<TensorDescription> inputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = inputs;
    RunTilingCase(inputs, outputs, {2}, {2}, ge::GRAPH_SUCCESS, 1);
}

TEST(ForeachZeroInplaceTilingTest, accepts_ge_scalar_storage_shape)
{
    std::vector<TensorDescription> inputs = {
        {{{}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = inputs;
    RunTilingCase(inputs, outputs, {1}, {1}, ge::GRAPH_SUCCESS, 1);
}

TEST(ForeachZeroInplaceTilingTest, counts_ge_scalar_as_one_element)
{
    ForeachZeroInplaceCompileInfo compileInfo;
    std::vector<TensorDescription> inputs = {
        {{{}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = inputs;
    gert::TilingContextPara context("ForeachZeroInplace", inputs, outputs, {}, std::vector<uint32_t>{1},
                                    std::vector<uint32_t>{1}, &compileInfo, 64, 262144, 4096);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(ForeachZeroInplaceTilingData));
    const auto* tilingData = reinterpret_cast<const ForeachZeroInplaceTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->tensorNum, 1);
    EXPECT_EQ(tilingData->tensorElements[0], 1);
}

TEST(ForeachZeroInplaceTilingTest, rejects_noncanonical_scalar_storage_shape)
{
    std::vector<TensorDescription> inputs = {
        {{{}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = inputs;
    RunTilingCase(inputs, outputs, {1}, {1}, ge::GRAPH_FAILED);
}

TEST(ForeachZeroInplaceTilingTest, rejects_non_contiguous_input_metadata)
{
    std::vector<TensorDescription> inputs = {
        {{{2, 3}, {20}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = inputs;
    RunTilingCase(inputs, outputs, {1}, {1}, ge::GRAPH_FAILED);
}

TEST(ForeachZeroInplaceTilingTest, rejects_non_scalar_shape_mismatch_with_same_element_count)
{
    std::vector<TensorDescription> inputs = {
        {{{2, 3}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = inputs;
    RunTilingCase(inputs, outputs, {1}, {1}, ge::GRAPH_FAILED);
}

TEST(ForeachZeroInplaceTilingTest, rejects_ref_output_count_mismatch)
{
    std::vector<TensorDescription> inputs = {
        {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = {
        {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    RunTilingCase(inputs, outputs, {2}, {1}, ge::GRAPH_FAILED);
}

TEST(ForeachZeroInplaceTilingTest, rejects_ref_output_dtype_mismatch)
{
    std::vector<TensorDescription> inputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    };
    RunTilingCase(inputs, outputs, {1}, {1}, ge::GRAPH_FAILED);
}

TEST(ForeachZeroInplaceTilingTest, rejects_ref_output_shape_mismatch)
{
    std::vector<TensorDescription> inputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = {
        {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    RunTilingCase(inputs, outputs, {1}, {1}, ge::GRAPH_FAILED);
}

TEST(ForeachZeroInplaceTilingTest, rejects_scalar_ref_output_origin_shape_mismatch)
{
    std::vector<TensorDescription> inputs = {
        {{{}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<TensorDescription> outputs = {
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    RunTilingCase(inputs, outputs, {1}, {1}, ge::GRAPH_FAILED);
}

TEST(ForeachZeroInplaceTilingTest, ignores_input_ref_output_format_metadata)
{
    std::vector<TensorDescription> inputs = {
        {{{1, 1, 2, 3}, {1, 1, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr, ge::FORMAT_NCHW},
    };
    std::vector<TensorDescription> outputs = {
        {{{1, 1, 2, 3}, {1, 1, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC, false, nullptr, ge::FORMAT_NCHW},
    };
    RunTilingCase(inputs, outputs, {1}, {1}, ge::GRAPH_SUCCESS, 1);
}
