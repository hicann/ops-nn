/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_map_index_infershape.cpp
 * \brief
 */
#include <cstdint>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"

TEST(MapIndexInfershapeTest, map_index_infershape_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24};
    gert::Shape levelIndexShape = {3};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, &levelIndexShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 0);
}

TEST(MapIndexInfershapeTest, map_index_infershape_invalid_x_dim_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8, 4};
    gert::Shape dataSeqShape = {24};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_not_multiple_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {25};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_level_index_len_mismatch_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24};
    gert::Shape levelIndexShape = {4}; // 24 / 8 = 3，长度不符

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, &levelIndexShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_level_index_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24};
    gert::Shape levelIndexShape = {-1}; // 动态维：1.0 口径跳过 level_index 长度校验

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, &levelIndexShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 0);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_data_seq_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {-1}; // 动态维：不支持动态场景，显式拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_x_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-1}; // x 为动态维：不支持动态场景，显式拒绝
    gert::Shape dataSeqShape = {24};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_rank_x_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-2}; // x 为未知秩：不支持动态场景，显式拒绝
    gert::Shape dataSeqShape = {24};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_rank_data_seq_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {-2}; // data_seq 为未知秩：不支持动态场景，显式拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_unknown_rank_level_index_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {24};
    gert::Shape levelIndexShape = {-2}; // level_index 为未知秩：跳过长度校验

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, &levelIndexShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 0);
}

TEST(MapIndexInfershapeTest, map_index_infershape_x_length_zero_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {0}; // x 长度为 0：拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, nullptr, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_exceed_max_gear_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::Shape dataSeqShape = {2056}; // 2056 / 8 = 257 档 > 256 上限：拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(MapIndexInfershapeTest, map_index_infershape_boundary_max_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // 边界恰好达标：x = 24000（长度上限），档位 = 6144000 / 24000 = 256（档位上限）
    gert::Shape xShape = {24000};
    gert::Shape dataSeqShape = {6144000};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &dataSeqShape, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 0);
}

TEST(MapIndexInfershapeTest, map_index_infershape_exceed_max_len_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MapIndex")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {24001}; // x 长度 24001 > 24000 上限：拒绝

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, nullptr, nullptr})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
