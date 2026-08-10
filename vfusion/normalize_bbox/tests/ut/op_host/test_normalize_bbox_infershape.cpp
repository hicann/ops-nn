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
 * \brief NormalizeBBox InferShape UT (iteration-1 core path)
 *
 * Coverage:
 *   - InferShape: y.shape == boxes.shape for normal (batch,num,4) and reversed (batch,4,num) layouts
 *
 *   InferDataType (y.dtype == boxes.dtype) now lives in
 *   op_graph/normalize_bbox_graph_infer.cpp: it is a graph-only deliverable. The op_graph
 *   UT module links graph_plugin_obj only and has no access to the tests/ut/common
 *   infershape fakers, so it is not covered here -- same as every other op in this repo
 *   that keeps InferDataType in op_graph (bn_infer_grad / lp_norm_update / in_infer_v2).
 */

#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

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

// ============================================================================================
// Iteration-3: round out InferShape (layout) coverage.
//   InferShape copies boxes->y verbatim (layout/dtype agnostic). dtype *rejection* lives
//   in the host tiling layer, NOT in the proto layer.
// ============================================================================================

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
                      .InputShapes({&boxesShape, &shapeHwShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[8, 4, 4096]");
}
