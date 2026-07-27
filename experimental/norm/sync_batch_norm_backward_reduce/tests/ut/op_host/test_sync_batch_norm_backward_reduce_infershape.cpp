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
 * \file test_sync_batch_norm_backward_reduce_infershape.cpp
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

class SyncBatchNormBackwardReduceInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SyncBatchNormBackwardReduceInfershape SetUp" << endl; }
    static void TearDownTestCase() { cout << "SyncBatchNormBackwardReduceInfershape TearDown" << endl; }
};

TEST_F(SyncBatchNormBackwardReduceInfershape, infershape_success)
{
    auto infershape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormBackwardReduce")->infer_shape;
    ASSERT_NE(infershape_func, nullptr);

    gert::StorageShape sumDyShape({16, 32}, {16, 32});
    gert::StorageShape sumDyDxPadShape({16, 32}, {16, 32});
    gert::StorageShape meanShape({16, 32}, {16, 32});
    gert::StorageShape invStdShape({16, 32}, {16, 32});
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SyncBatchNormBackwardReduce")
                      .NodeIoNum(4, 2)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&sumDyShape, &sumDyDxPadShape, &meanShape, &invStdShape})
                      .Build();
    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(infershape_func(context), ge::GRAPH_SUCCESS);
    const gert::Shape* out0 = context->GetOutputShape(0);
    const gert::Shape* out1 = context->GetOutputShape(1);
    ASSERT_NE(out0, nullptr);
    ASSERT_NE(out1, nullptr);
    ASSERT_EQ(out0->GetDimNum(), 2u);
    EXPECT_EQ(out0->GetDim(0), 16);
    EXPECT_EQ(out0->GetDim(1), 32);
    EXPECT_EQ(out1->GetDim(0), 16);
    EXPECT_EQ(out1->GetDim(1), 32);
}
