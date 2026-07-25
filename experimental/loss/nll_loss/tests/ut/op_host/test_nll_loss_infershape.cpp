/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_nll_loss_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "infer_shape_context_faker.h"
#include "op_impl_registry.h"

using namespace std;
using namespace ge;

class NllLossInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "NllLossInfershape SetUp" << endl; }
    static void TearDownTestCase() { cout << "NllLossInfershape TearDown" << endl; }
};

static ge::graphStatus RunNllLossInfershape(const gert::StorageShape& xShape, const gert::StorageShape& targetShape,
                                            int64_t& yDim0, int64_t& twDim0)
{
    auto infershape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("NllLoss")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("NllLoss")
                      .NodeIoNum(2, 2)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes(
                          {const_cast<gert::StorageShape*>(&xShape), const_cast<gert::StorageShape*>(&targetShape)})
                      .Build();
    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = infershape_func(context);
    if (ret == ge::GRAPH_SUCCESS) {
        const gert::Shape* yShape = context->GetOutputShape(0);
        const gert::Shape* twShape = context->GetOutputShape(1);
        if (yShape != nullptr && yShape->GetDimNum() >= 1) {
            yDim0 = yShape->GetDim(0);
        }
        if (twShape != nullptr && twShape->GetDimNum() >= 1) {
            twDim0 = twShape->GetDim(0);
        }
    }
    return ret;
}

TEST_F(NllLossInfershape, nll_loss_infershape_default_reduction_success)
{
    gert::StorageShape xShape({4, 8}, {4, 8});
    gert::StorageShape targetShape({4}, {4});
    int64_t yDim0 = -1;
    int64_t twDim0 = -1;
    EXPECT_EQ(RunNllLossInfershape(xShape, targetShape, yDim0, twDim0), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yDim0, 1);
    EXPECT_EQ(twDim0, 1);
}
