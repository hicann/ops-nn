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

#include "../../../framework/normalize_bbox_tf_plugin.cpp"

TEST(NormalizeBBoxTfPluginTest, NoAttributeFusionParserReturnsSuccess)
{
    const std::vector<const google::protobuf::Message*> insideNodes;
    const ge::Operator op("normalize_bbox", "NormalizeBBox");

    EXPECT_EQ(domi::NormalizeBBoxParserParams(insideNodes, op), domi::SUCCESS);
}
