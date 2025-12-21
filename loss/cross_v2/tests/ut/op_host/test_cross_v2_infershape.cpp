/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "../../../op_graph/cross_v2_proto.h"

class CrossV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "CrossV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CrossV2 Proto Test TearDown" << std::endl;
    }
};

std::vector<int64_t> CrossV2ToVector(const gert::Shape& shape) {
    size_t shape_size = shape.GetDimNum();
    std::vector<int64_t> shape_vec(shape_size, 0);
    for (size_t i = 0; i < shape_size; i++) {
        shape_vec[i] = shape.GetDim(i);
    }
    return shape_vec;
}

TEST_F(CrossV2, cross_v2_infershape_bf16_mean_case)
{
    gert::StorageShape Shape = {{4096, 3}, {4096, 3}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("CrossV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&Shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("CrossV2")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedYShape = {4096, 3};
    auto yShape = context->GetOutputShape(0);
    EXPECT_EQ(CrossV2ToVector(*yShape), expectedYShape);
}