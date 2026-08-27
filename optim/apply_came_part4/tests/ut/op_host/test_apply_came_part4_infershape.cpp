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
 * \file test_apply_came_part4_infershape.cpp
 * \brief ApplyCamePart4 infershape UT.
 *
 * 覆盖:全输入 shape 推导(param_out=param_in, r_out=r_in, c_out=c_in)、
 * 可选输入(sum_r/global_shape)缺省与传入、param 3D 非法输入拒绝。
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "op_common/op_host/util/shape_util.h"

class ApplyCamePart4Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyCamePart4Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyCamePart4Test TearDown" << std::endl; }
};

TEST_F(ApplyCamePart4Test, apply_came_part4_infershape_all_inputs)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart4"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart4")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape paramShape = {{96, 256}, {96, 256}};
    gert::StorageShape mShape = {{96, 256}, {96, 256}};
    gert::StorageShape rShape = {{96}, {96}};
    gert::StorageShape cShape = {{256}, {256}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape globalShape = {{2}, {2}};
    gert::StorageShape paramOutShape = {};
    gert::StorageShape rOutShape = {};
    gert::StorageShape cOutShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&paramShape, &mShape, &rShape, &cShape, &scalarShape, &scalarShape, &scalarShape,
                                    &scalarShape, &scalarShape, &scalarShape, &scalarShape, &globalShape})
                      .OutputShapes({&paramOutShape, &rOutShape, &cOutShape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(context->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 96);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 256);
    ASSERT_EQ(context->GetOutputShape(1)->GetDimNum(), 1);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), 96);
    ASSERT_EQ(context->GetOutputShape(2)->GetDimNum(), 1);
    EXPECT_EQ(context->GetOutputShape(2)->GetDim(0), 256);
}

TEST_F(ApplyCamePart4Test, apply_came_part4_infershape_optional_inputs_absent)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart4"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart4")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape paramShape = {{64, 32}, {64, 32}};
    gert::StorageShape mShape = {{64, 32}, {64, 32}};
    gert::StorageShape rShape = {{64}, {64}};
    gert::StorageShape cShape = {{32}, {32}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape paramOutShape = {};
    gert::StorageShape rOutShape = {};
    gert::StorageShape cOutShape = {};

    // 可选输入 sum_r(7) 与 global_shape(11) 缺省
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&paramShape, &mShape, &rShape, &cShape, &scalarShape, &scalarShape, &scalarShape,
                                    nullptr, &scalarShape, &scalarShape, &scalarShape, nullptr})
                      .OutputShapes({&paramOutShape, &rOutShape, &cOutShape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(context->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 64);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 32);
    ASSERT_EQ(context->GetOutputShape(1)->GetDimNum(), 1);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), 64);
    ASSERT_EQ(context->GetOutputShape(2)->GetDimNum(), 1);
    EXPECT_EQ(context->GetOutputShape(2)->GetDim(0), 32);
}

TEST_F(ApplyCamePart4Test, apply_came_part4_infershape_param_3d_rejected)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart4"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart4")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape paramShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape mShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape rShape = {{2}, {2}};
    gert::StorageShape cShape = {{4}, {4}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape paramOutShape = {};
    gert::StorageShape rOutShape = {};
    gert::StorageShape cOutShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&paramShape, &mShape, &rShape, &cShape, &scalarShape, &scalarShape, &scalarShape,
                                    nullptr, &scalarShape, &scalarShape, &scalarShape, nullptr})
                      .OutputShapes({&paramOutShape, &rOutShape, &cOutShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(ApplyCamePart4Test, apply_came_part4_infershape_unknown_rank)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart4"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyCamePart4")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // UNKNOWN_RANK(-2) 输入:三个输出都必须保持 unknown-rank 契约
    gert::StorageShape paramShape = {{-2}, {-2}};
    gert::StorageShape mShape = {{-2}, {-2}};
    gert::StorageShape rShape = {{-2}, {-2}};
    gert::StorageShape cShape = {{-2}, {-2}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape paramOutShape = {};
    gert::StorageShape rOutShape = {};
    gert::StorageShape cOutShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&paramShape, &mShape, &rShape, &cShape, &scalarShape, &scalarShape, &scalarShape,
                                    nullptr, &scalarShape, &scalarShape, &scalarShape, nullptr})
                      .OutputShapes({&paramOutShape, &rOutShape, &cOutShape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(Ops::Base::IsUnknownRank(*context->GetOutputShape(0)));
    EXPECT_TRUE(Ops::Base::IsUnknownRank(*context->GetOutputShape(1)));
    EXPECT_TRUE(Ops::Base::IsUnknownRank(*context->GetOutputShape(2)));
}
