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
#include <vector>

#include "infershape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
using TensorDescription = gert::InfershapeContextPara::TensorDescription;

std::vector<TensorDescription> GetInputs()
{
    return {
        {{{3, 2, 5}, {3, 2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{5, 21}, {5, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{7, 21}, {7, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{3, 2}, {3, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{21}, {21}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{21}, {21}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1, 2, 7}, {1, 2, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<TensorDescription> GetOutputs()
{
    return std::vector<TensorDescription>(7, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND});
}

void CheckInputShape(size_t index, const gert::Shape& shape, ge::graphStatus expected)
{
    auto inputs = GetInputs();
    inputs[index].shape_.MutableOriginShape() = shape;
    inputs[index].shape_.MutableStorageShape() = shape;
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, expected);
}
} // namespace

TEST(DynamicAUGRUInfershape, derives_seven_non_aligned_outputs)
{
    gert::InfershapeContextPara context("DynamicAUGRU", GetInputs(), GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {3, 2, 7}));
}

TEST(DynamicAUGRUInfershape, accepts_missing_optional_inputs)
{
    auto inputs = GetInputs();
    inputs.erase(inputs.begin() + 4, inputs.end());
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs(),
                                        std::vector<uint32_t>{1, 1, 1, 1, 0, 0, 0, 0}, std::vector<uint32_t>(7, 1));
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {3, 2, 7}));
}

TEST(DynamicAUGRUInfershape, accepts_legacy_mask)
{
    auto inputs = GetInputs();
    inputs[6] = {{{3, 2, 7}, {3, 2, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {3, 2, 7}));
}

TEST(DynamicAUGRUInfershape, preserves_unknown_time_dimension)
{
    auto inputs = GetInputs();
    inputs[0].shape_ = {{-1, 2, 5}, {-1, 2, 5}};
    inputs[3].shape_ = {{-1, 2}, {-1, 2}};
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {-1, 2, 7}));
}

TEST(DynamicAUGRUInfershape, preserves_unknown_time_dimension_with_legacy_mask)
{
    auto inputs = GetInputs();
    inputs[0].shape_ = {{-1, 2, 5}, {-1, 2, 5}};
    inputs[3].shape_ = {{-1, 2}, {-1, 2}};
    inputs[6] = {{{-1, 2, 7}, {-1, 2, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {-1, 2, 7}));
}

TEST(DynamicAUGRUInfershape, handles_unknown_input_rank)
{
    auto inputs = GetInputs();
    inputs[0].shape_ = {{-2}, {-2}};
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {3, 2, 7}));
}

TEST(DynamicAUGRUInfershape, handles_all_required_unknown_ranks)
{
    auto inputs = GetInputs();
    for (size_t i = 0; i < 4; ++i) {
        inputs[i].shape_ = {{-2}, {-2}};
    }
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {-1, 2, 7}));
}

TEST(DynamicAUGRUInfershape, handles_unknown_optional_input_ranks)
{
    auto inputs = GetInputs();
    for (size_t i = 4; i < inputs.size(); ++i) {
        inputs[i].shape_ = {{-2}, {-2}};
    }
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {3, 2, 7}));
}

TEST(DynamicAUGRUInfershape, handles_all_input_unknown_ranks)
{
    auto inputs = GetInputs();
    for (auto& input : inputs) {
        input.shape_ = {{-2}, {-2}};
    }
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {-1, -1, -1}));
}

TEST(DynamicAUGRUInfershape, derives_hidden_size_around_unknown_rank)
{
    auto inputs = GetInputs();
    inputs[2].shape_ = {{-2}, {-2}};
    for (size_t i = 4; i < inputs.size(); ++i) {
        inputs[i].shape_ = {{-2}, {-2}};
    }
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {3, 2, 7}));
}

TEST(DynamicAUGRUInfershape, preserves_multiple_unknown_dimensions)
{
    auto inputs = GetInputs();
    inputs[0].shape_ = {{3, -1, -1}, {3, -1, -1}};
    inputs[1].shape_ = {{-1, 21}, {-1, 21}};
    inputs[3].shape_ = {{3, -1}, {3, -1}};
    inputs[6].shape_ = {{-1}, {-1}};
    inputs[7].shape_ = {{1, -1, 7}, {1, -1, 7}};
    gert::InfershapeContextPara context("DynamicAUGRU", inputs, GetOutputs());
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>(7, {3, -1, 7}));
}

TEST(DynamicAUGRUInfershape, rejects_unknown_rank_marker_inside_fixed_rank)
{
    CheckInputShape(0, {-2, 2, 5}, ge::GRAPH_FAILED);
    CheckInputShape(6, {3, -2, 7}, ge::GRAPH_FAILED);
}

TEST(DynamicAUGRUInfershape, rejects_invalid_input_rank) { CheckInputShape(0, {3, 5}, ge::GRAPH_FAILED); }

TEST(DynamicAUGRUInfershape, rejects_mismatched_input_projection) { CheckInputShape(1, {6, 21}, ge::GRAPH_FAILED); }

TEST(DynamicAUGRUInfershape, rejects_invalid_hidden_gate_width) { CheckInputShape(2, {7, 20}, ge::GRAPH_FAILED); }

TEST(DynamicAUGRUInfershape, rejects_mismatched_attention_batch) { CheckInputShape(3, {3, 4}, ge::GRAPH_FAILED); }

TEST(DynamicAUGRUInfershape, rejects_invalid_bias_width) { CheckInputShape(4, {20}, ge::GRAPH_FAILED); }

TEST(DynamicAUGRUInfershape, rejects_mismatched_sequence_length_batch) { CheckInputShape(6, {4}, ge::GRAPH_FAILED); }

TEST(DynamicAUGRUInfershape, rejects_invalid_sequence_rank) { CheckInputShape(6, {3, 2}, ge::GRAPH_FAILED); }

TEST(DynamicAUGRUInfershape, rejects_mismatched_mask_hidden_size) { CheckInputShape(6, {3, 2, 8}, ge::GRAPH_FAILED); }

TEST(DynamicAUGRUInfershape, rejects_invalid_initial_state) { CheckInputShape(7, {2, 2, 7}, ge::GRAPH_FAILED); }
