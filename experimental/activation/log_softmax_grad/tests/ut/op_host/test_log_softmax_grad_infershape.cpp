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
#include <iostream>
#include "infershape_case_executor.h"

class LogSoftmaxGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LogSoftmaxGradInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LogSoftmaxGradInfershapeTest TearDown" << std::endl; }
};

// infershape: output_z shape == input_dy shape, independent of axis
TEST_F(LogSoftmaxGradInfershapeTest, infershape_case_3d_float_axis_last)
{
    gert::InfershapeContextPara infershapeContextPara("LogSoftmaxGrad",
                                                      {
                                                          {{{4, 4, 4}, {4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4, 4, 4}, {4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogSoftmaxGradInfershapeTest, infershape_case_4d_float16_axis_mid)
{
    gert::InfershapeContextPara infershapeContextPara("LogSoftmaxGrad",
                                                      {
                                                          {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogSoftmaxGradInfershapeTest, infershape_case_2d_bf16_axis_first)
{
    gert::InfershapeContextPara infershapeContextPara("LogSoftmaxGrad",
                                                      {
                                                          {{{10, 20}, {10, 20}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{10, 20}, {10, 20}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 20}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogSoftmaxGradInfershapeTest, infershape_case_3d_float_axis_last_neg)
{
    gert::InfershapeContextPara infershapeContextPara("LogSoftmaxGrad",
                                                      {
                                                          {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogSoftmaxGradInfershapeTest, infershape_case_2d_float16_default_axis)
{
    gert::InfershapeContextPara infershapeContextPara("LogSoftmaxGrad",
                                                      {
                                                          {{{3, 3}, {3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{3, 3}, {3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
