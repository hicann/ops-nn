/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_extend_conv2d_dyn_infershape.cpp
 * \brief
 */

#include <vector>
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "log/log.h"

const char* const DEFAULT_ATTR_VAL = "";

class Conv2DV2RuntimeDynInferShape : public testing::Test {};

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dDynFmapC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 1, 3, 3}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dDynWeightC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 16, 16}, {}};
    gert::StorageShape wShape = {{1, -1, 3, 3}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dDynGroups)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 16, 16}, {}};
    gert::StorageShape wShape = {{1, -1, 3, 3}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(-3)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFmapUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{1, 2, 3, 3}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFmapUnknownRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape wShape = {{1, 2, 3, 3}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFilterUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 16, 16}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFilterUnknownRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 16, 16}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dAllUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dAllUnknownRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFmapUnknownRankWeightUnknownShapeC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 3, 3}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFmapUnknownRankWeightUnknownShapeN)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, 4, 3, 3}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFmapUnknownRankWeightUnknownShapeHW)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 4, -1, -1}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFmapUnknownRankWeightUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFilterUnknownRankFmapUnknownShapeN)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, 2, 16, 16}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFilterUnknownRankFmapUnknownShapeC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 16, 16}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFilterUnknownRankFmapUnknownShapeHW)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, -1, -1}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};


    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dFilterUnknownRankFmapUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dPadNegative)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dPadModeSame)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dPadModeSameUpper)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;


    gert::StorageShape xShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_UPPER")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dPadModeSameLower)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_LOWER")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dPadModeSameValid)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    gert::StorageShape y1Shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&y0Shape, &y1Shape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRange0ToN)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{0, 0, 0, 0};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{0, 0, 0, 0};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{0, 0, 0, 0};
    gert::Shape y0ShapeMax{10, 10, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRange1ToN)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 1, 3, 3};
    gert::Shape y0ShapeMax{10, 10, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRange1To1)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1};
    gert::Shape xShapeMax{1, 1, 1, 1};
    gert::Shape wShapeMin{1, 1, 1, 1};
    gert::Shape wShapeMax{1, 1, 1, 1};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 1, 3, 3};
    gert::Shape y0ShapeMax{1, 1, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMinNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeNToN)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 20};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{10, 10, 1, 1};
    gert::Shape y0ShapeMax{10, 10, 1, 1};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRange1ToNWeightStc)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 20}; // static weight
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 10, 1, 1};
    gert::Shape y0ShapeMax{10, 10, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRange1ToNFmapStc)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 20}; // static fmap
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{10, 1, 12, 22};
    gert::Shape y0ShapeMax{10, 10, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangePadSame)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 1, 1, 1};
    gert::Shape y0ShapeMax{10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangePadSameLower)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 1, 1, 1};
    gert::Shape y0ShapeMax{10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-21, -12, -14, -16})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_LOWER")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangePadSameUpper)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 1, 1, 1};
    gert::Shape y0ShapeMax{10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-21, -12, -14, -16})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_UPPER")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeValid)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 1, 1, 1};
    gert::Shape y0ShapeMax{10, 10, 1, 1};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-21, -12, -14, -16})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeInvalidStride)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 1, 1, 1};
    gert::Shape y0ShapeMax{10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, -1, -1, -1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeInvalidDilation)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 1, 1, 1};
    gert::Shape y0ShapeMax{10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, -1, -1, -1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeFmapNDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 20};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 10, 2, 2};
    gert::Shape y0ShapeMax{10, 10, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeFmapNCDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 20};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 10, 2, 2};
    gert::Shape y0ShapeMax{10, 10, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeFmapNCHDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 20};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 20};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{1, 10, 1, 2};
    gert::Shape y0ShapeMax{10, 10, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeWeightWDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{10, 10, 2, 11};
    gert::Shape y0ShapeMax{10, 10, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeWeightHWDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{10, 10, 6, 11};
    gert::Shape y0ShapeMax{10, 10, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferShapeRangeWeightCHWDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 20};
    gert::Shape wShapeMin{10, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 20};
    gert::Shape y0ShapeMinNull{};
    gert::Shape y0ShapeMaxNull{};
    gert::Shape y0ShapeMin{10, 10, 6, 11};
    gert::Shape y0ShapeMax{10, 10, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> y0ShapeRange(&y0ShapeMinNull, &y0ShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&y0ShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(y0ShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(y0ShapeMax));
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferDtypeUNDEFINED)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 1, 3, 3}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    ge::DataType xDtype = ge::DT_INT8;
    ge::DataType y0Dtype = ge::DT_UNDEFINED;
    ge::DataType y1Dtype = ge::DT_UNDEFINED;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .InputDataTypes({&xDtype})
                      .OutputDataTypes({&y0Dtype, &y1Dtype})
                      .Build();

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT8);
}

TEST_F(Conv2DV2RuntimeDynInferShape, SupportedExtendConv2dInferDtypeDEFINED)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 1, 3, 3}, {}};
    gert::StorageShape y0Shape = {{}, {}};
    ge::DataType xDtype = ge::DT_INT8;
    ge::DataType y0Dtype = ge::DT_UNDEFINED;
    ge::DataType y1Dtype = ge::DT_UNDEFINED;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                          {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputDataTypes({&xDtype})
                      .OutputDataTypes({&y0Dtype, &y1Dtype})
                      .Build();

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConv2D")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
}