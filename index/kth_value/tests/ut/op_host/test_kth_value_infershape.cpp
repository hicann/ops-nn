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
#include "infer_shape_context_faker.h"
#include "infershape_case_executor.h"

class KthValueInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "KthValueInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "KthValueInfershape TearDown" << std::endl; }
};

TEST_F(KthValueInfershape, kthvalue_infershape_2d_last_axis)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{3, 10}, {3, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(5)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1}, {3, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_3d_first_axis)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{4, 5, 6}, {4, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 5, 6}, {1, 5, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_3d_middle_axis)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{2, 8, 3}, {2, 8, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 3}, {2, 1, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_1d)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_negative_dim)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 4}, {2, 1, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_invalid_dim_out_of_range)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{3, 10}, {3, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(5)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1}, {3, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_invalid_negative_dim_out_of_range)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{3, 10}, {3, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(5)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-3)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1}, {3, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_scalar_input_fails)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_int8_dtype)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{5, 20}, {5, 20}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(10)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 1}, {5, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KthValueInfershape, kthvalue_infershape_uint64_dtype)
{
    gert::InfershapeContextPara infershapeContextPara("KthValue",
                                                      {
                                                          {{{4, 8}, {4, 8}}, ge::DT_UINT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"k", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 1}, {4, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
