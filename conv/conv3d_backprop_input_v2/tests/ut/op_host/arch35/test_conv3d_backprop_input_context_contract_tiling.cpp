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
#include "base/err_mgr.h"
#include "../../../../op_host/op_tiling/common/conv_backprop_input_context_utils_internal.h"

namespace {
class FusionContractErrorScope {
public:
    FusionContractErrorScope()
    {
        (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
        (void)error_message::GetErrMgrRawErrorMessages();
    }
    ~FusionContractErrorScope() { (void)error_message::GetErrMgrRawErrorMessages(); }
};

void CheckMissingFrameworkMetadata(optiling::OpTypeV2 type, const char* opName)
{
    FusionContractErrorScope errors;
    // Function-level contract injection: real framework context, no mocked GetAttrs.
    // Public SetRunInfoToV2 checks/ parses context before calling this helper.
    gert::TilingContext context{};
    ASSERT_EQ(context.GetComputeNodeInfo(), nullptr);
    ASSERT_EQ(context.GetAttrs(), nullptr);
    Ops::NN::Conv::Conv3dBpInputV2RunInfo info{};
    info.enRelu0 = 1;
    info.enRelu1 = 1;
    EXPECT_FALSE(Ops::NN::Conv::GetFusionMode(info, opName, &context, type));
    EXPECT_EQ(info.enRelu0, 1);
    EXPECT_EQ(info.enRelu1, 1);
    const auto raw = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(raw.size(), 1U);
    ASSERT_NE(raw[0].error_id, nullptr);
    EXPECT_STREQ(raw[0].error_id.get(), "EZ9999");
    ASSERT_NE(raw[0].error_message, nullptr);
    const std::string message(raw[0].error_message.get());
    EXPECT_NE(message.find("GetFusionMode"), std::string::npos) << message;
    EXPECT_NE(message.find("ComputeNodeInfo"), std::string::npos) << message;
    EXPECT_NE(message.find("RuntimeAttrs"), std::string::npos) << message;
    EXPECT_NE(message.find(opName), std::string::npos) << message;
    EXPECT_NE(message.find("Initialize framework tiling metadata"), std::string::npos) << message;
    EXPECT_EQ(message.find("EZ0004"), std::string::npos) << message;
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(GetFusionModeA5Contract, MissingMetadataForExtend)
{
    CheckMissingFrameworkMetadata(optiling::OpTypeV2::kExtendConvTranspose, "ExtendConvTranspose");
}
TEST(GetFusionModeA5Contract, MissingMetadataForExtendV2)
{
    CheckMissingFrameworkMetadata(optiling::OpTypeV2::kExtendConvTransposeV2, "ExtendConvTransposeV2");
}
TEST(GetFusionModeA5Contract, NonExtendBypassesAttributeAccess)
{
    FusionContractErrorScope errors;
    gert::TilingContext context{};
    Ops::NN::Conv::Conv3dBpInputV2RunInfo info{};
    EXPECT_TRUE(Ops::NN::Conv::GetFusionMode(info, "Conv3DBackpropInputV2", &context,
                                             optiling::OpTypeV2::kConv3DBackpropInputV2));
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}
} // namespace
