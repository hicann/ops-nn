/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_conv3d_dyn_infershape.cpp
 * \brief
 */

#include <vector>
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "log/log.h"

const char* const DEFAULT_ATTR_VAL = "";
class Conv3DV2RuntimeDynInferShape : public testing::Test {};

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dDynFmapC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 1, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dDynWeightC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, -1, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dDynGroups)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, -1, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(-3)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFmapUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{1, 2, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFmapUnknownRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape wShape = {{1, 2, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFilterUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFilterUnknownRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dAllUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dAllUnknownRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFmapUnknownRankWeightUnknownShapeC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 1, 3, 3}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFmapUnknownRankWeightUnknownShapeN)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, 4, 1, 3, 3}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFmapUnknownRankWeightUnknownShapeD)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 4, -1, 3, 3}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFmapUnknownRankWeightUnknownShapeHW)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 4, 1, -1, -1}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFmapUnknownRankWeightUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFilterUnknownRankFmapUnknownShapeN)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, 2, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFilterUnknownRankFmapUnknownShapeC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFilterUnknownRankFmapUnknownShapeD)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, -1, 16, 16}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFilterUnknownRankFmapUnknownShapeHW)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 1, -1, -1}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dFilterUnknownRankFmapUnknownShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dPadNegative)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dPadModeSame)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dPadModeSameUpper)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_UPPER")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dPadModeSameLower)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_LOWER")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dPadModeSameValid)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRange0ToN)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{0, 0, 0, 0, 0};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{0, 0, 0, 0, 0};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{0, 0, 0, 0, 0};
    gert::Shape yShapeMax{10, 10, 3, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRange1ToN)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 1, 3, 3, 3};
    gert::Shape yShapeMax{10, 10, 3, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRange1To1)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 1};
    gert::Shape xShapeMax{1, 1, 1, 1, 1};
    gert::Shape wShapeMin{1, 1, 1, 1, 1};
    gert::Shape wShapeMax{1, 1, 1, 1, 1};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 1, 3, 3, 3};
    gert::Shape yShapeMax{1, 1, 3, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMinNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeNToN)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 10, 20};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{10, 10, 3, 1, 1};
    gert::Shape yShapeMax{10, 10, 3, 1, 1};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 0, 0, 0, 0})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRange1ToNWeightStc)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 10, 20}; // static weight
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 10, 1, 1, 1};
    gert::Shape yShapeMax{10, 10, 3, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRange1ToNFmapStc)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 10, 20}; // static fmap
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{10, 1, 12, 12, 22};
    gert::Shape yShapeMax{10, 10, 3, 3, 3};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangePadSame)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 1, 1, 1, 1};
    gert::Shape yShapeMax{10, 10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1, -1, -1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangePadSameLower)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 1, 1, 1, 1};
    gert::Shape yShapeMax{10, 10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -2, -21, -12, -14, -16})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_LOWER")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangePadSameUpper)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 1, 1, 1, 1};
    gert::Shape yShapeMax{10, 10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -2, -21, -12, -14, -16})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_UPPER")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeValid)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 1, 1, 1, 1};
    gert::Shape yShapeMax{10, 10, 1, 1, 1};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -2, -21, -12, -14, -16})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeInvalidStride)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 1, 1, 1, 1};
    gert::Shape yShapeMax{10, 10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, -1, -1, -1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeInvalidDilation)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 1};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{1, 1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 1, 1, 1, 1};
    gert::Shape yShapeMax{10, 10, 10, 10, 20};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, -1, -1, -1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeFmapNDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 10, 20};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 10, 2, 2, 2};
    gert::Shape yShapeMax{10, 10, 2, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeFmapNCDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 10, 20};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 10, 2, 2, 2};
    gert::Shape yShapeMax{10, 10, 2, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeFmapNCDDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 10, 20};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 10, 1, 2, 2};
    gert::Shape yShapeMax{10, 10, 2, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeFmapNCDHDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{1, 1, 1, 1, 20};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 10, 20};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{1, 10, 1, 1, 2};
    gert::Shape yShapeMax{10, 10, 2, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeWeightWDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 10, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{10, 10, 2, 2, 11};
    gert::Shape yShapeMax{10, 10, 2, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeWeightHWDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 10, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{10, 10, 2, 6, 11};
    gert::Shape yShapeMax{10, 10, 2, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeWeightDHWDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 10, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{10, 10, 6, 6, 11};
    gert::Shape yShapeMax{10, 10, 2, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}

TEST_F(Conv3DV2RuntimeDynInferShape, SupportedQuantConv3dInferShapeRangeWeightCDHWDyn)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape_range;

    gert::Shape xShapeMin{10, 10, 10, 10, 20};
    gert::Shape xShapeMax{10, 10, 10, 10, 20};
    gert::Shape wShapeMin{10, 1, 1, 1, 1};
    gert::Shape wShapeMax{10, 10, 10, 10, 20};
    gert::Shape yShapeMinNull{};
    gert::Shape yShapeMaxNull{};
    gert::Shape yShapeMin{10, 10, 6, 6, 11};
    gert::Shape yShapeMax{10, 10, 2, 2, 2};

    gert::Range<gert::Shape> xShapeRange(&xShapeMin, &xShapeMax);
    gert::Range<gert::Shape> wShapeRange(&wShapeMin, &wShapeMax);
    gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
    auto holder = gert::InferShapeRangeContextFaker()
                            .NodeIoNum(2, 1)
                            .IrInstanceNum({1, 1})
                            .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                            .NodeAttrs({
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                            })
                            .InputShapeRanges({&xShapeRange, &wShapeRange})
                            .OutputShapeRanges({&yShapeRange})
                            .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(yShapeMin));
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(yShapeMax));
}
