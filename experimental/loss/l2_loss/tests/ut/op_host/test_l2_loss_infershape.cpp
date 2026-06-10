/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "infer_shape_context_faker.h"
#include "op_impl_registry.h"

using namespace std;
using namespace ge;

class L2LossInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "L2LossInfershape SetUp" << endl; }
    static void TearDownTestCase() { cout << "L2LossInfershape TearDown" << endl; }
};

static ge::graphStatus RunL2LossInfershape(const gert::StorageShape& xShape, ge::DataType dtype,
                                           int64_t& outDim0)
{
    auto infershape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("L2Loss")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("L2Loss")
                      .NodeIoNum(1, 1)
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({const_cast<gert::StorageShape*>(&xShape)})
                      .Build();
    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    ge::graphStatus ret = infershape_func(context);
    if (ret == ge::GRAPH_SUCCESS) {
        const gert::Shape* outShape = context->GetOutputShape(0);
        if (outShape != nullptr && outShape->GetDimNum() >= 1) {
            outDim0 = outShape->GetDim(0);
        }
    }
    return ret;
}

TEST_F(L2LossInfershape, l2_loss_infershape_fp32_success)
{
    gert::StorageShape x_shape({3, 4}, {3, 4});
    int64_t outDim0 = -1;
    EXPECT_EQ(RunL2LossInfershape(x_shape, ge::DT_FLOAT, outDim0), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outDim0, 1);
}

TEST_F(L2LossInfershape, l2_loss_infershape_fp16_success)
{
    gert::StorageShape x_shape({5, 10}, {5, 10});
    int64_t outDim0 = -1;
    EXPECT_EQ(RunL2LossInfershape(x_shape, ge::DT_FLOAT16, outDim0), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outDim0, 1);
}

TEST_F(L2LossInfershape, l2_loss_infershape_dynamic_shape)
{
    gert::StorageShape x_shape({5, -1}, {5, -1});
    int64_t outDim0 = -1;
    EXPECT_EQ(RunL2LossInfershape(x_shape, ge::DT_FLOAT, outDim0), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outDim0, 1);
}
