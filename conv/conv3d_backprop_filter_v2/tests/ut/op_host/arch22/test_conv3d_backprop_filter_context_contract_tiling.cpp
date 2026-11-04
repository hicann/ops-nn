/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "base/err_mgr.h"
// Exercise the private helpers in an isolated translation unit: the helpers and PAD_UPPER have internal linkage.
// Rename the implementation's only external entry point to keep it distinct from the production object.
#define SetConv3dBpFilterV2RunInfo TestContractSetConv3dBpFilterV2RunInfo
#include "../../../../op_host/op_tiling/common/conv_backprop_filter_context_utils.cpp"
#undef SetConv3dBpFilterV2RunInfo

TEST(FrameworkAttrsContract, FilterSetStridesAttr)
{
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrRawErrorMessages();
    // A real framework context with no ComputeNodeInfo models the broken framework contract.
    gert::TilingContext contextStorage{};
    auto context = &contextStorage;
    ASSERT_EQ(context->GetAttrs(), nullptr);
    Ops::NN::Conv::Conv3dBpFilterV2RunInfo info{};

    EXPECT_FALSE(Ops::NN::Conv::SetStridesAttr(context, info));
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_NE(errors[0].error_id, nullptr);
    EXPECT_STREQ(errors[0].error_id.get(), "EZ9999");
    ASSERT_NE(errors[0].error_message, nullptr);
    const std::string message(errors[0].error_message.get());
    EXPECT_NE(message.find("SetStridesAttr"), std::string::npos) << message;
    EXPECT_NE(message.find("ComputeNodeInfo"), std::string::npos) << message;
    EXPECT_NE(message.find("RuntimeAttrs"), std::string::npos) << message;
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(FrameworkAttrsContract, FilterValidateDilationsBasicInfo)
{
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrRawErrorMessages();
    // A real framework context with no ComputeNodeInfo models the broken framework contract.
    gert::TilingContext contextStorage{};
    auto context = &contextStorage;
    ASSERT_EQ(context->GetAttrs(), nullptr);

    const gert::ContinuousVector* values = nullptr;
    EXPECT_FALSE(Ops::NN::Conv::ValidateDilationsBasicInfo("Conv3DBackpropFilterV2", context, values));
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_NE(errors[0].error_id, nullptr);
    EXPECT_STREQ(errors[0].error_id.get(), "EZ9999");
    ASSERT_NE(errors[0].error_message, nullptr);
    const std::string message(errors[0].error_message.get());
    EXPECT_NE(message.find("ValidateDilationsBasicInfo"), std::string::npos) << message;
    EXPECT_NE(message.find("ComputeNodeInfo"), std::string::npos) << message;
    EXPECT_NE(message.find("RuntimeAttrs"), std::string::npos) << message;
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(FrameworkAttrsContract, FilterValidateGroupsAttr)
{
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrRawErrorMessages();
    // A real framework context with no ComputeNodeInfo models the broken framework contract.
    gert::TilingContext contextStorage{};
    auto context = &contextStorage;
    ASSERT_EQ(context->GetAttrs(), nullptr);

    int32_t groups = 0;
    EXPECT_FALSE(Ops::NN::Conv::ValidateGroupsAttr("Conv3DBackpropFilterV2", context->GetAttrs(), groups));
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_NE(errors[0].error_id, nullptr);
    EXPECT_STREQ(errors[0].error_id.get(), "EZ9999");
    ASSERT_NE(errors[0].error_message, nullptr);
    const std::string message(errors[0].error_message.get());
    EXPECT_NE(message.find("ValidateGroupsAttr"), std::string::npos) << message;
    EXPECT_NE(message.find("ComputeNodeInfo"), std::string::npos) << message;
    EXPECT_NE(message.find("RuntimeAttrs"), std::string::npos) << message;
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(FrameworkAttrsContract, FilterValidatePadsAttr)
{
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrRawErrorMessages();
    // A real framework context with no ComputeNodeInfo models the broken framework contract.
    gert::TilingContext contextStorage{};
    auto context = &contextStorage;
    ASSERT_EQ(context->GetAttrs(), nullptr);

    const gert::ContinuousVector* values = nullptr;
    EXPECT_FALSE(Ops::NN::Conv::ValidatePadsAttr("Conv3DBackpropFilterV2", context->GetAttrs(), values));
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_NE(errors[0].error_id, nullptr);
    EXPECT_STREQ(errors[0].error_id.get(), "EZ9999");
    ASSERT_NE(errors[0].error_message, nullptr);
    const std::string message(errors[0].error_message.get());
    EXPECT_NE(message.find("ValidatePadsAttr"), std::string::npos) << message;
    EXPECT_NE(message.find("ComputeNodeInfo"), std::string::npos) << message;
    EXPECT_NE(message.find("RuntimeAttrs"), std::string::npos) << message;
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}
