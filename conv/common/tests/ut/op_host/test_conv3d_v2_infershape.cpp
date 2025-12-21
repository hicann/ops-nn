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
 * \file test_conv3d_v2_infershape.cpp
 * \brief
 */

#include <vector>
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "log/log.h"

const char* const DEFAULT_ATTR_VAL = "";

class Conv3DV2RuntimeInferShape : public testing::Test {};

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2Fp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2Fp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2Bf16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2OutNDHWC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2FmapNDHWC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2WeightNDHWC)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 3, 3, 32, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, UnsupportedConv3dv2CNotEqual)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 16, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, UnsupportedConv3dv2KernelGTFmap)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{3, 1, 32, 30, 20}, {}};
    gert::StorageShape wShape = {{1, 1, 20, 23, 23}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 5, 10, 17})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Fmap_N_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{0, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[0, 16, 2, 16, 16]");
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Fmap_Cin_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 0, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Fmap_Din_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 0, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 0, 16, 16]");
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Fmap_Hin_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 0, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 1, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 2, 0, 16]");
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Fmap_Win_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 0}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 2, 16, 0]");
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Weight_Cout_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{0, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 0, 2, 16, 16]");
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Weight_Cin_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 0, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Weight_Kd_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 0, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Weight_Kh_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 0, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Weight_Kw_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 0}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Input_Zero_Output_Not_Zero_D)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 0, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({5, 5, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Input_Zero_Output_Not_Zero_H)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 0, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 5, 5, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Input_Zero_Output_Not_Zero_W)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 0}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 5, 5})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Fmap_Din_Zero_negative)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 0, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Fmap_Hin_Zero_negative)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 0, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NCDHW_Fmap_Win_Zero_negative)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 0}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2PadModeSame)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2PadModeValid)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2PadModeSameUpper)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 2, 4}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_UPPER")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2PadModeSameLower)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 2, 4}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SAME_LOWER")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2PadModeInvalid)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 2, 4}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("INVALID")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedQuantConv3d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantConv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 1, 16, 16}, {}};
    gert::StorageShape wShape = {{1, 32, 1, 3, 3}, {}};
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
}

TEST_F(Conv3DV2RuntimeInferShape, supportedConv3dv2NDHWCfp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{1, 3, 3, 32, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, supportedConv3dv2NDHWCfp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{1, 3, 3, 32, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, supportedConv3dv2NDHWCbf16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{1, 1, 16, 16, 32}, {}};
    gert::StorageShape wShape = {{1, 3, 3, 32, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NDHWC_Fmap_N_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{0, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[0, 2, 16, 16, 16]");
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NDHWC_Fmap_Cin_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 0}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(Conv3DV2RuntimeInferShape, Conv3dv2_Format_NDHWC_Fmap_Din_Zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    gert::StorageShape xShape = {{2, 0, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{1, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 0, 16, 16, 16]");
}

TEST_F(Conv3DV2RuntimeInferShape, SupportedConv3dv2GroupsTwo)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DV2")->infer_shape;

    // NCDHW: N=2, C=32, D=3, H=18, W=18
    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    // Weight (NCDHW layout for filter): Cout=8, Cin/groups=16 (groups=2), Kd=1, Kh=3, Kw=3
    gert::StorageShape wShape = {{8, 16, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({
                          // Stride 1 everywhere
                          {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          // Pad H/W by 1 each side to keep H/W; no D pad
                          {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 1, 1, 1, 1})},
                          {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                          {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                          {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                          {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                          {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    // Expect: N=2, C=Cout=8, D=3 (kd=1 no pad), H=18 (kh=3, pad=1), W=18 (kw=3, pad=1)
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 8, 3, 18, 18]");
}
