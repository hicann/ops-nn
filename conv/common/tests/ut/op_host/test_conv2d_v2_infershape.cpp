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
 * \file test_conv2d_v2_infershape.cpp
 * \brief
 */

#include <vector>
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "log/log.h"

const char* const DEFAULT_ATTR_VAL = "";

class Conv2DV2RuntimeInferShape : public testing::Test {};

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2Fp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2Fp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2Bf16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Conv2DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2Fp16WithBias)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{2, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &wShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2Fp16PadModeSame)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{2, 32, 3, 3}, {}};
    gert::StorageShape biasShape = {{2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &wShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2Fp16PadModeSameUpper)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{2, 32, 4, 4}, {}};
    gert::StorageShape biasShape = {{2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &wShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_UPPER")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2Fp16PadModeSameLower)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{2, 32, 4, 4}, {}};
    gert::StorageShape biasShape = {{2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &wShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_LOWER")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2Fp16PadModeValid)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{2, 32, 4, 4}, {}};
    gert::StorageShape biasShape = {{2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &wShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedQuantConv2dv2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv2D")->infer_shape;
    gert::StorageShape xShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape wShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedQuantConv2dv2InvalidRoundMode)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv2D")->infer_shape;
    gert::StorageShape xShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape wShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("round")}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedQuantConv2dv2ValidRoundModeHif8)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv2D")->infer_shape;
    gert::StorageShape xShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape wShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_HIFLOAT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_HIFLOAT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("hybrid")}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedQuantConv2dv2InvalidRoundModeHif8)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv2D")->infer_shape;
    gert::StorageShape xShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape wShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_HIFLOAT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_HIFLOAT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, UnSupportedQuantConv2dv2Offset)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv2D")->infer_shape;
    gert::StorageShape xShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape wShape = {{1, 1, 1, 1}, {}};
    gert::StorageShape ndShape = {{1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT64, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(3, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")}
                      })
                      .InputShapes({&xShape, &wShape, &ndShape, &ndShape, &ndShape})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, UnsupportedConv2dv2FmapFomart)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, UnsupportedConv2dv2WeightFomart)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, UnsupportedConv2dv2OutFomart)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, UnsupportedConv2dv2CNotEqual)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 16, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, UnsupportedConv2dv2KernelGTFmap)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{3, 1, 30, 20}, {}};
    gert::StorageShape wShape = {{1, 1, 23, 23}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 10, 17})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Fmap_N_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{0, 32, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[0, 16, 16, 16]");
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Fmap_Cin_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 0, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}


TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Fmap_Hin_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 0, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 1, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 0, 16]");
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Fmap_Win_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 18, 0}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 16, 0]");
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Weight_Cout_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 18, 18}, {}};
    gert::StorageShape wShape = {{0, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 0, 16, 16]");
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Weight_Cin_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 0, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Weight_Kh_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 0, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Weight_Kw_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 0}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Input_Zero_Output_Not_Zero_H)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 0, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({5, 5, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Input_Zero_Output_Not_Zero_W)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 18, 0}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 5, 5})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Fmap_Hin_Zero_negative)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 0, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Fmap_Win_Zero_negative)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 18, 0}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv2DV2RuntimeInferShape, Conv2dv2_Format_NCHW_Fmap_N_Zero_Hif8)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{0, 32, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_HIFLOAT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_HIFLOAT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[0, 16, 16, 16]");
}


TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2NHWCFp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{3, 3, 32, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2NHWCFp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{3, 3, 32, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2NHWCBf16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{3, 3, 32, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Conv2DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2NHWCFp16WithBias)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{3, 3, 32, 2}, {}};
    gert::StorageShape biasShape = {{2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &wShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DV2RuntimeInferShape, SupportedConv2dv2NHWCFp16PadModeSame)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{3, 3, 32, 2}, {}};
    gert::StorageShape biasShape = {{2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &wShape, &biasShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}