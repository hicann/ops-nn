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
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "op_impl_registry.h"

class AdamInferShapeTest : public testing::Test {};

TEST_F(AdamInferShapeTest, null_context_reports_stage_error_once)
{
    auto implementation = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOneAssign");
    ASSERT_NE(implementation, nullptr);
    ASSERT_NE(implementation->infer_shape, nullptr);
    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    errors.GetRawErrorMessages();
    EXPECT_EQ(implementation->infer_shape(nullptr), ge::GRAPH_FAILED);
    auto reports = errors.GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(reports[0].error_id, "EZ1014");
}
