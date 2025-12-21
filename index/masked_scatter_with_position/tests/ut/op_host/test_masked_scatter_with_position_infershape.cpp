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
#include <iostream>
#include "../../../op_graph/masked_scatter_with_position_graph_infer.cpp"
#include "infershape_test_util.h"
#include "ut_op_common.h"

using namespace ge;
class MaskedScatterWithPositionProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaskedScatterWithPositionProtoTest SetUpTestCase" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaskedScatterWithPositionProtoTest TearDown" << std::endl;
    }
};
