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
 * \file test_normalize_bbox_infershape.cpp
 * \brief NormalizeBBox shape validation and propagation tests.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "base/context_builder/op_infer_shape_context_builder.h"

class NormalizeBBoxProto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NormalizeBBox Proto Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "NormalizeBBox Proto Test TearDown" << std::endl; }
};

// InferShape: normal layout (batch, num, 4) -> y == boxes
TEST_F(NormalizeBBoxProto, normalize_bbox_infershape_normal_layout)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NormalizeBBox")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape boxesShape = {{2, 8, 4}, {2, 8, 4}};
    gert::StorageShape shapeHwShape = {{2, 3}, {2, 3}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&boxesShape, &shapeHwShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[2, 8, 4]");
}

// InferShape: reversed layout (batch, 4, num) -> y == boxes (memory layout only, shape copied verbatim)
TEST_F(NormalizeBBoxProto, normalize_bbox_infershape_reversed_layout)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NormalizeBBox")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape boxesShape = {{2, 4, 8}, {2, 4, 8}};
    gert::StorageShape shapeHwShape = {{2, 3}, {2, 3}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeAttrs({{"reversed_box", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .InputShapes({&boxesShape, &shapeHwShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[2, 4, 8]");
}

// InferShape: large multi-batch shape preserved
TEST_F(NormalizeBBoxProto, normalize_bbox_infershape_large_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NormalizeBBox")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape boxesShape = {{64, 1024, 4}, {64, 1024, 4}};
    gert::StorageShape shapeHwShape = {{64, 3}, {64, 3}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&boxesShape, &shapeHwShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[64, 1024, 4]");
}

// InferShape: dynamic dim (-1) in boxes -> y preserves -1
TEST_F(NormalizeBBoxProto, normalize_bbox_infershape_dynamic_dim)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NormalizeBBox")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape boxesShape = {{-1, -1, 4}, {-1, -1, 4}};
    gert::StorageShape shapeHwShape = {{2, 3}, {2, 3}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&boxesShape, &shapeHwShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[-1, -1, 4]");
}

// InferShape: unknown rank (-2) in boxes -> y is unknown rank
TEST_F(NormalizeBBoxProto, normalize_bbox_infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NormalizeBBox")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape boxesShape = {{-2}, {-2}};
    gert::StorageShape shapeHwShape = {{2, 3}, {2, 3}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&boxesShape, &shapeHwShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[-2]");
}

// InferShape: reversed large layout (batch, 4, num) preserved verbatim
TEST_F(NormalizeBBoxProto, normalize_bbox_infershape_reversed_large_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NormalizeBBox")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape boxesShape = {{8, 4, 4096}, {8, 4, 4096}};
    gert::StorageShape shapeHwShape = {{8, 3}, {8, 3}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeAttrs({{"reversed_box", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .InputShapes({&boxesShape, &shapeHwShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[8, 4, 4096]");
}

TEST_F(NormalizeBBoxProto, rejects_known_invalid_shapes_and_preserves_dynamic_shapes)
{
    struct Case {
        std::vector<int64_t> boxes;
        std::vector<int64_t> shapeHw;
        bool reversed;
        bool valid;
    };
    const std::vector<Case> cases = {
        {{4}, {4, 3}, false, false},
        {{1, 1, 1, 1, 1, 1, 1, 1, 4}, {1, 3}, false, false},
        {{2, 3, 4}, {6}, false, false},
        {{2, 3, 4}, {2, 1, 3}, false, false},
        {{2, 3, 4}, {2, 2}, false, false},
        {{2, 3, 4}, {3, 3}, false, false},
        {{2, 3, 3}, {2, 3}, false, false},
        {{2, 3, 4}, {2, 3}, true, false},
        {{-1, -1, 4}, {-1, 3}, false, true},
        {{2, 3, -1}, {2, 3}, false, true},
        {{-2}, {-2}, false, true},
        {{2, 0, 4}, {2, 3}, false, true},
        {{1, 1, 1, 1, 1, 1, 1, 4}, {1, 3}, false, true},
    };
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("NormalizeBBox");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    for (size_t index = 0; index < cases.size(); ++index) {
        SCOPED_TRACE(index);
        const auto& item = cases[index];
        gert::StorageShape boxesShape, hwShape;
        for (size_t i = 0; i < item.boxes.size(); ++i) {
            boxesShape.MutableOriginShape().AppendDim(item.boxes[i]);
            boxesShape.MutableStorageShape().AppendDim(item.boxes[i]);
        }
        for (size_t i = 0; i < item.shapeHw.size(); ++i) {
            hwShape.MutableOriginShape().AppendDim(item.shapeHw[i]);
            hwShape.MutableStorageShape().AppendDim(item.shapeHw[i]);
        }
        gert::Tensor boxes(boxesShape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, ge::DT_FLOAT);
        gert::Tensor hw(hwShape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, ge::DT_INT32);
        gert::OpInferShapeContextBuilder builder;
        builder.OpType("NormalizeBBox").OpName("normalize_bbox_shape_check").IONum(2, 1);
        builder.InputTensors({&boxes, &hw}).AppendAttr(item.reversed);
        builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        auto holder = builder.Build();
        ASSERT_NE(holder.GetContext(), nullptr);
        EXPECT_EQ(impl->infer_shape(holder.GetContext()), item.valid ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED);
        if (item.valid) {
            EXPECT_EQ(*holder.GetContext()->GetOutputShape(0), boxesShape.GetOriginShape());
        }
    }
}
