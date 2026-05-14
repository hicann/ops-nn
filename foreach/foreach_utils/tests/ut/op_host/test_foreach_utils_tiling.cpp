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
#include "op_impl_registry.h"

class ForeachUtilsTilingTest : public testing::Test {};

TEST_F(ForeachUtilsTilingTest, register_common_tiling_functions)
{
    const char* ops[] = {"ForeachAddList", "ForeachMulScalar", "ForeachRoundOffNumber", "ForeachSubScalar"};
    for (const char* op : ops) {
        const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op);
        ASSERT_NE(impl, nullptr) << op;
        EXPECT_NE(impl->tiling, nullptr) << op;
        EXPECT_NE(impl->tiling_parse, nullptr) << op;
    }
}
