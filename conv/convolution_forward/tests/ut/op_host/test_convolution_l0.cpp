/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <map>
#include "opdev/platform.h"
#include "base/err_mgr.h"
#include "opdev/make_op_executor.h"
#include "../../../op_host/op_api/convolution.h"

class convolution_test : public testing::Test {};

TEST_F(convolution_test, owner_l0_list_failure_returns_null)
{
    op::SocVersionManager platform(op::SocVersion::ASCEND910B);
    auto executor = CREATE_EXECUTOR();
    ASSERT_NE(executor.get(), nullptr);
    auto input = executor->AllocTensor(op::Shape({1, 1, 3, 3}), op::DataType::DT_FLOAT16, op::Format::FORMAT_NCHW);
    auto weight = executor->AllocTensor(op::Shape({1, 1, 1, 1}), op::DataType::DT_FLOAT16, op::Format::FORMAT_NCHW);
    ASSERT_NE(input, nullptr);
    ASSERT_NE(weight, nullptr);
    const int64_t one[] = {1};
    const int64_t two[] = {1, 1};
    const int64_t pad[] = {0, 0};
    auto invalidStride = executor->AllocIntArray(one, 1);
    auto validDilation = executor->AllocIntArray(two, 2);
    auto validPad = executor->AllocIntArray(pad, 2);
    ASSERT_NE(invalidStride, nullptr);
    ASSERT_NE(validDilation, nullptr);
    ASSERT_NE(validPad, nullptr);
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrRawErrorMessages();
    EXPECT_EQ(l0op::Conv2d5HdFp16(input, weight, nullptr, invalidStride, validPad, validDilation, 1, executor.get()),
              nullptr);
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_NE(errors[0].error_id, nullptr);
    EXPECT_STREQ(errors[0].error_id.get(), "EZ0038");
    ASSERT_EQ(errors[0].args_key.size(), 4U);
    ASSERT_EQ(errors[0].args_value.size(), 4U);
    std::map<std::string, std::string> fields;
    for (size_t i = 0; i < errors[0].args_key.size(); ++i) {
        ASSERT_NE(errors[0].args_key[i], nullptr);
        ASSERT_NE(errors[0].args_value[i], nullptr);
        fields.emplace(errors[0].args_key[i].get(), errors[0].args_value[i].get());
    }
    const std::map<std::string, std::string> expected = {
        {"param_name", "strides"},
        {"op_name", "Convolution"},
        {"incorrect_size", "1"},
        {"reason", "At least 2 elements are required for 2D convolution"}};
    EXPECT_EQ(fields, expected);
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
