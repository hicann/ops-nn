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
 * \file test_conv3d_infershape.cpp
 * \brief Conv3D 算子 RT2.0 InferShape / InferShapeRange UT（迁移自 canndev RT1.0/RT2.0 UT 并补充用例）
 */

#include <vector>

#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "log/log.h"

namespace {
// 属性布局：strides(0) pads(1) dilations(2) groups(3) data_format(4) offset_x(5) padding(6)
std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeConv3DAttrs(const std::vector<int64_t>& strides,
                                                                       const std::vector<int64_t>& pads,
                                                                       const std::vector<int64_t>& dilations,
                                                                       int64_t groups, const std::string& dataFormat,
                                                                       const std::string& padding)
{
    return {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
            {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
            {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
            {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
            {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)},
            {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)}};
}
} // namespace

class Conv3DRuntimeInferShape : public testing::Test {};

// 基础静态 shape（迁移自 RT1.0 conv3d_Format_NCDHW_Padding_VALID）
TEST_F(Conv3DRuntimeInferShape, Conv3dNcdhwBasic)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 2, 16, 16]");
}

// NDHWC 输入（迁移自 RT1.0 conv3d_Format_NDHWC_Padding_SAME）
TEST_F(Conv3DRuntimeInferShape, Conv3dNdhwcBasic)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{16, 2, 3, 3, 32}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 2, 16, 16, 16]");
}

// DHWCN filter（迁移自 RT1.0 conv3d_Filter_Format_DHWCN）
TEST_F(Conv3DRuntimeInferShape, Conv3dFilterDhwcn)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 2, 16, 16, 16]");
}

// stride / dilation 组合（迁移自 RT1.0 conv3d_Dilation_h_Not_EQ_1_SUCCESS）
TEST_F(Conv3DRuntimeInferShape, Conv3dStrideDilation)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 2, 2, 2, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    // od = (3 - 2 * (2 - 1) - 1) / 1 + 1 = 1, oh = ow = (18 - 2 * (3 - 1) - 1) / 1 + 1 = 14
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 1, 14, 14, 16]");
}

// padding=SAME（迁移自 canndev RT2.0 conv3d_NDHWC_DHWCN_NDHWC_SAME_1）
TEST_F(Conv3DRuntimeInferShape, Conv3dPaddingSame)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{40, 48, 14, 14, 3}, {}};
    gert::StorageShape wShape = {{5, 3, 3, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 2, 3, 1}, {-1, -1, -1, -1, -1, -1}, {1, 1, 1, 1, 1}, 1, "NDHWC",
                                                 "SAME"))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    // SAME：od = ceil(48 / 1) = 48, oh = ceil(14 / 2) = 7, ow = ceil(14 / 3) = 5
    ASSERT_EQ(Ops::Base::ToString(*output), "[40, 48, 7, 5, 3]");
}

// padding=VALID（补齐 RT1.0 功能：pads 置 0）
TEST_F(Conv3DRuntimeInferShape, Conv3dPaddingValid)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{8, 28, 8, 60, 88}, {}};
    gert::StorageShape wShape = {{32, 28, 1, 2, 2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 2, 2, 2}, {5, 5, 5, 5, 5, 5}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID"))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    // VALID：pads 置 0，od = 8 / 2 = 4, oh = 30, ow = 44
    ASSERT_EQ(Ops::Base::ToString(*output), "[8, 32, 4, 30, 44]");
}

// 动态 shape + padding=SAME：动态维 pads 记 -1、输出维 -1（对齐 RT1.0 SetConv3dDynamicPads）
TEST_F(Conv3DRuntimeInferShape, Conv3dPaddingSameDynamic)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 14, 14, 3}, {}};
    gert::StorageShape wShape = {{5, 3, 3, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 2, 3, 1}, {-1, -1, -1, -1, -1, -1}, {1, 1, 1, 1, 1}, 1, "NDHWC",
                                                 "SAME"))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    // D 维动态：od = -1；H/W 静态：oh = ceil(14 / 2) = 7, ow = ceil(14 / 3) = 5；kn 为 filter DHWCN 末位 3
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, -1, 7, 5, 3]");
}

// 动态 shape + pads 含 -1 占位：放行（RT1.0 动态语义；老 RT2.0 一律拦截负 pads）
TEST_F(Conv3DRuntimeInferShape, Conv3dDynamicNegativePadsPass)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, 14, 14, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 1, 1, 1}, {-1, -1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    // D 维动态（pads -1 占位放行）：od = -1；H/W 静态按 pads=1 计算
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, -1, 14, 14, 64]");
}

// 动态 shape：D/H/W 为 -1（迁移自 RT1.0 conv3d_dynamic_dhw_high_unlimited）
TEST_F(Conv3DRuntimeInferShape, Conv3dDynamicDhw)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{1, -1, -1, -1, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, -1, -1, -1, 64]");
}

// 动态 shape：N/C 为 -1（迁移自 RT1.0 conv3d_dynamic_ncw_normal）
TEST_F(Conv3DRuntimeInferShape, Conv3dDynamicNc)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, 7, 14, -1}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[-1, 16, 7, 14, -1]");
}

// 全 -1 动态 shape（迁移自 RT1.0 conv3d_dynamic_ncdhw_no_range）
TEST_F(Conv3DRuntimeInferShape, Conv3dDynamicAll)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{-1, -1, -1, -1, -1}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[-1, 16, -1, -1, -1]");
}

// unknown rank（-2，迁移自 RT1.0 conv3d_dyanmic_rank）
TEST_F(Conv3DRuntimeInferShape, Conv3dUnknownRank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[-1, 16, -1, -1, -1]");
}

// 零 Tensor：x 的 N 维为 0
TEST_F(Conv3DRuntimeInferShape, Conv3dFmapNZero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{0, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[0, 16, 2, 16, 16]");
}

// 零 Tensor：x 的 C/D 维为 0，输出同为 0 维
TEST_F(Conv3DRuntimeInferShape, Conv3dFmapDZero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 0, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 1, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 0, 16, 16]");
}

// 零 Tensor：x 的 Cin 为 0 但输出非零 Tensor，应拦截
TEST_F(Conv3DRuntimeInferShape, Conv3dFmapCinZeroFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 0, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 零 Tensor：filter 的 Cout 为 0
TEST_F(Conv3DRuntimeInferShape, Conv3dWeightCoutZero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{0, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 0, 2, 16, 16]");
}

// 零 Tensor：filter 的 Kd 为 0，应拦截
TEST_F(Conv3DRuntimeInferShape, Conv3dWeightKdZeroFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 0, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// x format 不支持（迁移自 canndev RT2.0 conv3d_ND_NCDHW_NCDHW_VALID_1）
TEST_F(Conv3DRuntimeInferShape, Conv3dXFormatInvalidFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{8, 28, 8, 60, 88}, {}};
    gert::StorageShape wShape = {{32, 28, 1, 2, 2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID"))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// filter format 不支持
TEST_F(Conv3DRuntimeInferShape, Conv3dFilterFormatInvalidFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{8, 28, 8, 60, 88}, {}};
    gert::StorageShape wShape = {{32, 28, 1, 2, 2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID"))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// x 与 filter dtype 不一致应拦截（对齐 RT1.0 Conv3DVerify，补运行期校验）
TEST_F(Conv3DRuntimeInferShape, Conv3dDtypeMismatchFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// y format 不支持（迁移自 RT1.0 conv3d_Y_Format_Error_Failed）
TEST_F(Conv3DRuntimeInferShape, Conv3dYFormatInvalidFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// x 非 5 维（迁移自 RT1.0 conv3d_Format_NDHWC_Padding_SAME_X_shape_Failed）
TEST_F(Conv3DRuntimeInferShape, Conv3dXDimInvalidFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 2, 3, 3, 32}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", "SAME"))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// pads 长度非法（迁移自 RT1.0 conv3d_Padding_Size_Failed）
TEST_F(Conv3DRuntimeInferShape, Conv3dPadsSizeFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// strides 长度非法（迁移自 RT1.0 conv3d_Strides_Length_Failed）
TEST_F(Conv3DRuntimeInferShape, Conv3dStridesSizeFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// dilations 长度非法（迁移自 RT1.0 conv3d_Dilation_Length_Failed）
TEST_F(Conv3DRuntimeInferShape, Conv3dDilationsSizeFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 负 strides（迁移自 RT1.0 conv3d_Negative_Strides_Failed，canndev RT2.0 未拦截，已补齐）
TEST_F(Conv3DRuntimeInferShape, Conv3dNegativeStridesFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, -1, -1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 负 pads（迁移自 RT1.0 conv3d_Negative_Padding_Failed）
TEST_F(Conv3DRuntimeInferShape, Conv3dNegativePadsFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 18, 18, 32}, {}};
    gert::StorageShape wShape = {{2, 3, 3, 32, 16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, -1, -1, -1, -1, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// dilations 越界（H 维 256 超出 [1, 255] 值域，对齐 conv3dv2 tiling ParseDilationLegal）
TEST_F(Conv3DRuntimeInferShape, Conv3dDilationRangeFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 256, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 输出维公式 int64 溢出拦截（乘加链溢出保护）
TEST_F(Conv3DRuntimeInferShape, Conv3dOutputDimOverflowFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 9223372036854775807LL, 8, 8}, {}};
    gert::StorageShape wShape = {{16, 32, 3, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// data_format 非法（迁移自 RT1.0 conv3d_Data_Format_Failed）
TEST_F(Conv3DRuntimeInferShape, Conv3dDataFormatInvalidFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCHWDD", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 通道不整除（ic % kc != 0，应拦截）
TEST_F(Conv3DRuntimeInferShape, Conv3dChannelNotDivisibleFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 15, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// groups 与通道不一致（ic / kc != groups，应拦截）
TEST_F(Conv3DRuntimeInferShape, Conv3dGroupsMismatchFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 16, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 3, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 隐式 groups（ic / kc == groups，groups=1 时按 ic/kc 生效，对齐 RT1.0 SetGroupsConv）
TEST_F(Conv3DRuntimeInferShape, Conv3dImplicitGroups)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 16, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 2, 16, 16]");
}

// padding 后输入小于卷积核（迁移自 Conv3DV2 UnsupportedConv3dv2KernelGTFmap）
TEST_F(Conv3DRuntimeInferShape, Conv3dKernelGtFmapFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{3, 1, 32, 30, 20}, {}};
    gert::StorageShape wShape = {{1, 1, 20, 23, 23}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 5, 10, 17}, {0, 0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// InferShapeRange：SPECIFIC pad，D/H/W 动态
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeSpecific)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D"), nullptr);
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;
    ASSERT_NE(inferShapeRangeFunc, nullptr);

    gert::Shape xMin = {1, 32, 4, 8, 16};
    gert::Shape xMax = {1, 32, 16, 64, 128};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape targetMin = {1, 16, 2, 4, 8};
    gert::Shape targetMax = {1, 16, 8, 32, 64};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// InferShapeRange：SAME padding
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeSame)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 4, 8, 16};
    gert::Shape xMax = {1, 32, 16, 64, 128};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "SAME"))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    // SAME：od = ceil(d / strd)，oh = ceil(h / strh)，ow = ceil(w / strw)
    gert::Shape targetMin = {1, 16, 2, 4, 8};
    gert::Shape targetMax = {1, 16, 8, 32, 64};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// InferShapeRange：输出维公式 int64 溢出拦截（非 SAME 分支 pads 乘加链溢出保护）
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangePadsOverflowFail)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 4, 8, 16};
    gert::Shape xMax = {1, 32, 16, 64, 128};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 2, 2, 2},
                                                 {4611686018427387904LL, 4611686018427387904LL, 0, 0, 0, 0},
                                                 {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}

// InferShapeRange：SAME 分支 ceil 加法 int64 溢出拦截
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeSameOverflowFail)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 9223372036854775807LL, 8, 16};
    gert::Shape xMax = {1, 32, 9223372036854775807LL, 64, 128};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "SAME"))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}

// InferShapeRange：padding=VALID（pads 置 0 计算，忽略显式 pads，对齐 RT1.0/Conv3DV2）
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangePaddingValid)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 4, 8, 16};
    gert::Shape xMax = {1, 32, 16, 64, 128};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(
                          MakeConv3DAttrs({1, 1, 2, 2, 2}, {3, 3, 3, 3, 3, 3}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID"))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    // VALID：按 pads=0 计算（显式 pads=3 被忽略）：od 低/高 = (4-3)/2+1=1 / (16-3)/2+1=7，
    // oh = (8-3)/2+1=3 / (64-3)/2+1=31，ow = (16-3)/2+1=7 / (128-3)/2+1=63
    gert::Shape targetMin = {1, 16, 1, 3, 7};
    gert::Shape targetMax = {1, 16, 7, 31, 63};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// InferShapeRange：NDHWC + DHWCN filter
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeNdhwc)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 4, 8, 16, 32};
    gert::Shape xMax = {1, 16, 64, 128, 32};
    gert::Shape wMin = {3, 3, 3, 32, 16};
    gert::Shape wMax = {3, 3, 3, 32, 16};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 2, 2, 2, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape targetMin = {1, 1, 3, 7, 16};
    gert::Shape targetMax = {1, 7, 31, 63, 16};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// InferShapeRange：上界 -1（不限制）与 4096 封顶
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeHighUnlimited)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, -1, -1, -1, 32};
    gert::Shape xMax = {1, -1, -1, -1, 32};
    gert::Shape wMin = {2, 3, 3, 32, 64};
    gert::Shape wMax = {2, 3, 3, 32, 64};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NDHWC", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    // D/H/W 上界 -1（不限制），下界对齐 RT1.0 为 1
    gert::Shape targetMin = {1, 1, 1, 1, 64};
    gert::Shape targetMax = {1, -1, -1, -1, 64};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// InferShapeRange：动态卷积核（filter 维度带 range）
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeDynamicKernel)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 8, 8, 8};
    gert::Shape xMax = {1, 32, 64, 64, 64};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 5, 5, 5};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    // 下界用小卷积核：od >= (8 - 3 + 1) = 6，上界用大输入/小核：od <= 64 - 3 + 1 = 62
    gert::Shape targetMin = {1, 16, 6, 6, 6};
    gert::Shape targetMax = {1, 16, 60, 60, 60};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// InferShapeRange：N/C 维不参与 4096 封顶（对齐 RT1.0：clamp 仅对 D/H/W 生效）
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeNcNotCapped)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 4, 8, 16};
    gert::Shape xMax = {8192, 32, 16, 64, 128};
    gert::Shape wMin = {8192, 32, 3, 3, 3};
    gert::Shape wMax = {8192, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    // N 维原样继承 x 的 N 维 range（1~8192），C 维 = (kn, kn) = (8192, 8192)，均不封顶
    gert::Shape targetMin = {1, 8192, 2, 4, 8};
    gert::Shape targetMax = {8192, 8192, 8, 32, 64};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// InferShapeRange：D/H/W 动态维上界封顶 4096（DYNAMIC_RANGE_UPPER_BOUND，对齐 RT1.0）
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeDhwCapped)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 4, 4, 4};
    gert::Shape xMax = {1, 32, 16384, 16384, 16384};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    // od = oh = ow：下界 (4 - 2 - 1) + 1 = 2，上界 16382 封顶 4096
    gert::Shape targetMin = {1, 16, 2, 2, 2};
    gert::Shape targetMax = {1, 16, 4096, 4096, 4096};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// InferShapeRange：kn 无 range 信息（-1）时 C 维退化为 (1, -1)（对齐 RT1.0）
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeKnUnknownFallback)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 4, 8, 16};
    gert::Shape xMax = {1, 32, 16, 64, 128};
    gert::Shape wMin = {-1, 32, 3, 3, 3};
    gert::Shape wMax = {-1, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    // filter N 维 range 无信息（-1）时，C 维对齐 RT1.0 退化为 (1, -1)
    gert::Shape targetMin = {1, 1, 2, 4, 8};
    gert::Shape targetMax = {1, -1, 8, 32, 64};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// pads 长度 1 广播（兼容老 RT2.0 cube_util GetConv3DPads 语义）
TEST_F(Conv3DRuntimeInferShape, Conv3dPadsBroadcastLen1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {1}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    // pads={1} 等价 {1,1,1,1,1,1}：od=(3+2-1-1)/1+1=4, oh=ow=(18+2-2-1)/1+1=18
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 4, 18, 18]");
}

// pads 长度 3 广播（兼容老 RT2.0 cube_util GetConv3DPads 语义）
TEST_F(Conv3DRuntimeInferShape, Conv3dPadsBroadcastLen3)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {1, 2, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    // pads={1,2,1} 等价 {1,1,2,2,1,1}：od=(3+2-1-1)/1+1=4, oh=(18+4-2-1)/1+1=20, ow=18
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 16, 4, 20, 18]");
}

// padding 属性非法值（对齐 RT1.0 SetPadListConv3dForPadding L7892 拦截，补齐）
TEST_F(Conv3DRuntimeInferShape, Conv3dPaddingInvalidFail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape;

    gert::StorageShape xShape = {{2, 32, 3, 18, 18}, {}};
    gert::StorageShape wShape = {{16, 32, 2, 3, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW",
                                                 "SAME_UPPER"))
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// InferShapeRange：输入 range 下界 0 时输出下界收敛为 1
// （对齐 RT1.0 SetConv3dOutShapeDimRange L8078：下界恒 max(1)，无零 Tensor 下界 0）
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeZeroLowerBoundClampToOne)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 0, 0, 0};
    gert::Shape xMax = {1, 32, 16, 64, 128};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    // od/oh/ow 下界公式值 (0-3+1)/1+1 = -2，收敛为 1；上界 (16-3+1)=14、(64-3+1)=62、(128-3+1)=126
    gert::Shape targetMin = {1, 16, 1, 1, 1};
    gert::Shape targetMax = {1, 16, 14, 62, 126};
    gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
    ASSERT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange);
}

// InferShapeRange：维数非法应拦截
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeDimInvalidFail)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 4};
    gert::Shape xMax = {1, 32, 16};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}

// InferShapeRange：x format 非法应拦截（对齐 InferShape 的 GetConv3DXShape）
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeXFormatInvalidFail)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 4, 8, 16};
    gert::Shape xMax = {1, 32, 16, 64, 128};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}

// InferShapeRange：filter format 非法应拦截（对齐 InferShape 的 GetConv3DFilterShape）
TEST_F(Conv3DRuntimeInferShape, Conv3dShapeRangeFilterFormatInvalidFail)
{
    auto inferShapeRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3D")->infer_shape_range;

    gert::Shape xMin = {1, 32, 4, 8, 16};
    gert::Shape xMax = {1, 32, 16, 64, 128};
    gert::Shape wMin = {16, 32, 3, 3, 3};
    gert::Shape wMax = {16, 32, 3, 3, 3};
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> xRange(&xMin, &xMax);
    gert::Range<gert::Shape> wRange(&wMin, &wMax);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);

    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInputNum(2)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .NodeAttrs(MakeConv3DAttrs({1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", ""))
                      .Build();

    ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}
