/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_deformable_offsets_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class DeformableOffsetsInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DeformableOffsetsInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DeformableOffsetsInfershape TearDown" << std::endl;
    }
};

TEST_F(DeformableOffsetsInfershape, deformable_offsets_infer_shape_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DeformableOffsets",
        {
            {{{4, 16, 64, 64}, {4, 16, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
            {{{4, 216, 64, 64}, {4, 216, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{4, 32, 192, 192}, {4, 32, 192, 192}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3})},
            {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
            {"deformable_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
            {"modulated", Ops::NN::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 16, 192, 192},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
