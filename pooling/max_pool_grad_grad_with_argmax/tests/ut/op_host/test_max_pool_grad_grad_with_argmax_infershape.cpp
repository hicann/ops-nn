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
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"

using namespace ge;

class MaxPoolGradGradWithArgmaxInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolGradGradWithArgmaxInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaxPoolGradGradWithArgmaxInfershape TearDown" << std::endl; }
};

TEST_F(MaxPoolGradGradWithArgmaxInfershape, max_pool_grad_grad_with_argmax_unknown_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradGradWithArgmax")->infer_shape;
    gert::StorageShape xShape = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
    gert::StorageShape gradShape = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
    gert::StorageShape argmaxShape = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")}})
                      .InputShapes({&xShape, &gradShape, &argmaxShape})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {-1, -1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(MaxPoolGradGradWithArgmaxInfershape, max_pool_grad_grad_with_argmax_partial_unknown_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradGradWithArgmax")->infer_shape;
    gert::StorageShape xShape = {{1, -1, 8, 8}, {1, -1, 8, 8}};
    gert::StorageShape gradShape = {{1, -1, 4, 4}, {1, -1, 4, 4}};
    gert::StorageShape argmaxShape = {{1, -1, 4, 4}, {1, -1, 4, 4}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")}})
                      .InputShapes({&xShape, &gradShape, &argmaxShape})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {1, -1, 4, 4};
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(MaxPoolGradGradWithArgmaxInfershape, max_pool_grad_grad_with_argmax_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradGradWithArgmax")->infer_shape;
    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape gradShape = {{-2}, {}};
    gert::StorageShape argmaxShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")}})
                      .InputShapes({&xShape, &gradShape, &argmaxShape})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedOutputShape));
}
