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

class MishInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MishInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MishInfershape TearDown" << std::endl; }
};

TEST_F(MishInfershape, mish_infershape_test_float16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mish",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {32, 4, 4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MishInfershape, mish_infershape_test_float)
{
    gert::InfershapeContextPara infershapeContextPara("Mish",
                                                      {
                                                          {{{8, 256}, {8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{8, 256}, {8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 256},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MishInfershape, mish_infershape_test_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mish",
        {
            {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, -1, -1, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
