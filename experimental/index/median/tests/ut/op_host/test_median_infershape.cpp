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

#include "infershape_case_executor.h"

class MedianInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MedianInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MedianInfershape TearDown" << std::endl; }
};

// Median: 1 输入(input) / 2 输出(values, indices)；输出 shape = 输入去掉末轴，indices 为 int32。
TEST_F(MedianInfershape, median_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Median",
        {
            {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // input
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // values
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // indices
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, -1, -1}, // values
        {1, -1, -1}, // indices
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
