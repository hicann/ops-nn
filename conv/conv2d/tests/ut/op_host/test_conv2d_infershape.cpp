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
 * \file test_conv2d_infershape.cpp
 * \brief
 */

#include <string>
#include <utility>
#include <vector>
#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "error_util.h"
#include "log/log.h"

const char* const DEFAULT_ATTR_VAL = "";

class Conv2DRuntimeInferShape : public testing::Test {};

struct Conv2DInferOpt {
    std::vector<int64_t> strides{1, 1, 1, 1};
    std::vector<int64_t> pads{1, 1, 1, 1};
    std::vector<int64_t> dilations{1, 1, 1, 1};
    int64_t groups = 1;
    const char* dataFormat = "NCHW";
    const char* padding = DEFAULT_ATTR_VAL;
    const char* autoPad = DEFAULT_ATTR_VAL;
    ge::Format xFmt = ge::FORMAT_NCHW;
    ge::Format wFmt = ge::FORMAT_NCHW;
    ge::Format yFmt = ge::FORMAT_NCHW;
    ge::Format biasFmt = ge::FORMAT_ND;
    ge::Format offsetWFmt = ge::FORMAT_ND;
};

static std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeConv2DAttrs(const Conv2DInferOpt& opt)
{
    return {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.strides)},
            {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.pads)},
            {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.dilations)},
            {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(opt.groups)},
            {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(opt.dataFormat)},
            {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(opt.padding)},
            {"auto_pad", Ops::NN::AnyValue::CreateFrom<std::string>(opt.autoPad)}};
}

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t d : dims) {
        shape.MutableOriginShape().AppendDim(d);
    }
    return shape;
}

static ge::graphStatus RunConv2DInfer(const std::vector<int64_t>& xDims, const std::vector<int64_t>& wDims,
                                      std::string* yStr, const Conv2DInferOpt& opt = {},
                                      const std::vector<int64_t>* biasDims = nullptr,
                                      const std::vector<int64_t>* offsetWDims = nullptr)
{
    gert::StorageShape xShape = MakeStorageShape(xDims);
    gert::StorageShape wShape = MakeStorageShape(wDims);
    gert::StorageShape yShape;
    gert::StorageShape biasShape;
    gert::StorageShape offsetWShape;
    std::vector<gert::StorageShape*> inputs{&xShape, &wShape};
    if (biasDims != nullptr) {
        biasShape = MakeStorageShape(*biasDims);
        inputs.push_back(&biasShape);
    }
    if (offsetWDims != nullptr) {
        offsetWShape = MakeStorageShape(*offsetWDims);
        inputs.push_back(&offsetWShape);
    }

    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(inputs.size(), 1).IrInstanceNum(std::vector<uint32_t>(inputs.size(), 1));
    faker.NodeInputTd(0, ge::DT_FLOAT16, opt.xFmt, ge::FORMAT_RESERVED);
    faker.NodeInputTd(1, ge::DT_FLOAT16, opt.wFmt, ge::FORMAT_RESERVED);
    if (biasDims != nullptr) {
        faker.NodeInputTd(2, ge::DT_FLOAT16, opt.biasFmt, ge::FORMAT_RESERVED);
    }
    if (offsetWDims != nullptr) {
        faker.NodeInputTd(3, ge::DT_INT8, opt.offsetWFmt, ge::FORMAT_RESERVED);
    }
    faker.NodeOutputTd(0, ge::DT_FLOAT16, opt.yFmt, ge::FORMAT_RESERVED);
    faker.NodeAttrs(MakeConv2DAttrs(opt)).InputShapes(inputs).OutputShapes({&yShape});
    auto holder = faker.Build();
    auto* inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;
    const ge::graphStatus ret = inferShapeFunc(holder.GetContext<gert::InferShapeContext>());
    if (yStr != nullptr && ret == ge::GRAPH_SUCCESS) {
        *yStr = Shape2String(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0));
    }
    return ret;
}

static ge::graphStatus RunConv2DRange(const gert::Shape& xMin, const gert::Shape& xMax, const gert::Shape& wMin,
                                      const gert::Shape& wMax, std::string* yMinStr, std::string* yMaxStr,
                                      const Conv2DInferOpt& opt)
{
    gert::Shape xMinShape = xMin;
    gert::Shape xMaxShape = xMax;
    gert::Shape wMinShape = wMin;
    gert::Shape wMaxShape = wMax;
    gert::Shape yMin{};
    gert::Shape yMax{};
    gert::Range<gert::Shape> xRange(&xMinShape, &xMaxShape);
    gert::Range<gert::Shape> wRange(&wMinShape, &wMaxShape);
    gert::Range<gert::Shape> yRange(&yMin, &yMax);
    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, opt.xFmt, ge::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, opt.wFmt, ge::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, opt.yFmt, ge::FORMAT_RESERVED)
                      .NodeAttrs(MakeConv2DAttrs(opt))
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .Build();
    auto* inferRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape_range;
    const ge::graphStatus ret = inferRangeFunc(holder.GetContext<gert::InferShapeRangeContext>());
    if (ret == ge::GRAPH_SUCCESS) {
        *yMinStr = Shape2String(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin());
        *yMaxStr = Shape2String(*holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax());
    }
    return ret;
}

TEST_F(Conv2DRuntimeInferShape, basic1)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, basic2)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 2, 4};
    opt.pads = {0, 0, 0, 0};
    opt.dilations = {1, 1, 2, 2};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({3, 90, 100, 78}, {66, 30, 5, 5}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[3, 66, 46, 18]");
}

TEST_F(Conv2DRuntimeInferShape, xShapeNHWC)
{
    Conv2DInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.yFmt = ge::FORMAT_NHWC;
    opt.dataFormat = "NHWC";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 16, 16, 32}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 16, 16, 64]");
}

TEST_F(Conv2DRuntimeInferShape, wShapeNHWC)
{
    Conv2DInferOpt opt;
    opt.wFmt = ge::FORMAT_NHWC;
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 3, 3, 32}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, wShapeHWCN)
{
    Conv2DInferOpt opt;
    opt.wFmt = ge::FORMAT_HWCN;
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {3, 3, 32, 64}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, bias1D)
{
    std::vector<int64_t> bias{64};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, {}, &bias), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, bias4DNCHW)
{
    Conv2DInferOpt opt;
    opt.biasFmt = ge::FORMAT_NCHW;
    std::vector<int64_t> bias{1, 64, 1, 1};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt, &bias), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, bias4DNHWC)
{
    Conv2DInferOpt opt;
    opt.biasFmt = ge::FORMAT_NHWC;
    std::vector<int64_t> bias{1, 1, 1, 64};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt, &bias), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, paddingSAME)
{
    Conv2DInferOpt opt;
    opt.pads = {-1, -1, -1, -1};
    opt.padding = "SAME";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, paddingVALID)
{
    Conv2DInferOpt opt;
    opt.pads = {-1, -1, -1, -1};
    opt.padding = "VALID";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, autoPadSAME_UPPER)
{
    Conv2DInferOpt opt;
    opt.pads = {-1, -1, -1, -1};
    opt.autoPad = "SAME_UPPER";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, autoPadSAME_LOWER)
{
    Conv2DInferOpt opt;
    opt.pads = {-1, -1, -1, -1};
    opt.autoPad = "SAME_LOWER";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, autoPadNOTSET)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    opt.autoPad = "NOTSET";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, autoPadVALID)
{
    Conv2DInferOpt opt;
    opt.pads = {-1, -1, -1, -1};
    opt.autoPad = "VALID";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, xShapeCHWN)
{
    Conv2DInferOpt opt;
    opt.xFmt = ge::FORMAT_CHWN;
    opt.yFmt = ge::FORMAT_CHWN;
    opt.dataFormat = "CHWN";
    ASSERT_EQ(RunConv2DInfer({32, 16, 16, 1}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, wShapeCHWN)
{
    Conv2DInferOpt opt;
    opt.wFmt = ge::FORMAT_CHWN;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {32, 3, 3, 64}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, bias4DCHWN)
{
    Conv2DInferOpt opt;
    opt.biasFmt = ge::FORMAT_CHWN;
    std::vector<int64_t> bias{64, 1, 1, 1};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt, &bias), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, yShapeCHWN)
{
    Conv2DInferOpt opt;
    opt.yFmt = ge::FORMAT_CHWN;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, bias2D)
{
    std::vector<int64_t> bias{64, 2};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, {}, &bias), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidBiasChannel)
{
    Conv2DInferOpt opt;
    opt.biasFmt = ge::FORMAT_NCHW;
    std::vector<int64_t> bias{1, 1, 1, 1};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt, &bias), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidBiasOtherDims)
{
    Conv2DInferOpt opt;
    opt.biasFmt = ge::FORMAT_NCHW;
    std::vector<int64_t> bias{1, 64, 1, 3};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt, &bias), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, negativeStrides)
{
    Conv2DInferOpt opt;
    opt.strides = {-1, -1, -1, -1};
    std::vector<int64_t> bias{64};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt, &bias), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, negativeDilations)
{
    Conv2DInferOpt opt;
    opt.dilations = {-1, -1, -1, -1};
    std::vector<int64_t> bias{64};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt, &bias), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidGroups1)
{
    Conv2DInferOpt opt;
    opt.groups = 2;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidGroups2)
{
    Conv2DInferOpt opt;
    opt.groups = 3;
    ASSERT_EQ(RunConv2DInfer({1, 9, 16, 16}, {32, 3, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, invalidGroups3)
{
    ASSERT_EQ(RunConv2DInfer({1, 9, 16, 16}, {32, 4, 3, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedStridesDimNum)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 1, 1, 1};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedDilationsDimNum)
{
    Conv2DInferOpt opt;
    opt.dilations = {1, 1, 1, 1, 1};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, negativePads)
{
    Conv2DInferOpt opt;
    opt.pads = {1, 1, 1, -11};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedPadsDimNum)
{
    Conv2DInferOpt opt;
    opt.pads = {1, 1, 1, 1, 1};
    opt.padding = "EXPLICIT";
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedPadding)
{
    Conv2DInferOpt opt;
    opt.pads = {1, 1, 1, -11};
    opt.padding = "INVALID_PADDING";
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unsupportedAutoPad)
{
    Conv2DInferOpt opt;
    opt.pads = {1, 1, 1, -11};
    opt.autoPad = "INVALID_AUTO_PAD";
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, negativeInputWithPadsDilations)
{
    Conv2DInferOpt opt;
    opt.dilations = {1, 1, 3, 3};
    ASSERT_EQ(RunConv2DInfer({1, 32, 3, 3}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, ZeroTensorInputZeroTensorOutput)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({0, 32, 16, 16}, {64, 32, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[0, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, ZeroTensorInputZeroTensorOutput1)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {0, 32, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 0, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, ZeroTensorInputZeroTensorOutput2)
{
    ASSERT_EQ(RunConv2DInfer({1, 32, 0, 16}, {64, 32, 3, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, ZeroTensorInputZeroTensorOutput3)
{
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 0}, {64, 32, 3, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, ZeroTensorInputZeroTensorOutput4)
{
    ASSERT_EQ(RunConv2DInfer({1, 0, 0, 0}, {64, 0, 3, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, ZeroTensorInputZeroTensorOutput5)
{
    ASSERT_EQ(RunConv2DInfer({0, 0, 0, 0}, {64, 0, 3, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, ZeroTensorInputZeroTensorOutput6)
{
    ASSERT_EQ(RunConv2DInfer({0, 0, 0, 0}, {0, 0, 3, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, ZeroTensorInputZeroTensorOutput7)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 0, 16, 16}, {64, 0, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 0, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, ZeroTensorInputZeroTensorOutput8)
{
    Conv2DInferOpt opt;
    opt.groups = 2;
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 0, 16, 16}, {64, 0, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 0, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, UnsupportedHWZeroTensorInput)
{
    ASSERT_EQ(RunConv2DInfer({1, 32, 0, 16}, {64, 32, 2, 2}, nullptr), ge::GRAPH_FAILED);
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 0}, {64, 32, 2, 2}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, UnsupportedKhKwZeroTensorInput)
{
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 0, 3}, nullptr), ge::GRAPH_FAILED);
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 0}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, UnsupportedCinZeroTensorInput1)
{
    ASSERT_EQ(RunConv2DInfer({1, 0, 16, 16}, {64, 32, 3, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, UnsupportedCinGroupZeroTensorInput1)
{
    Conv2DInferOpt opt;
    opt.groups = 2;
    ASSERT_EQ(RunConv2DInfer({1, 0, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, UnsupportedKcZeroTensorInput)
{
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 0, 3, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, UnsupportedKcGroupZeroTensorInput)
{
    Conv2DInferOpt opt;
    opt.groups = 2;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 0, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, UnsupportedNormalTensorInputZeroTensorOutput)
{
    ASSERT_EQ(RunConv2DInfer({1, 32, 1, 16}, {64, 32, 4, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, dynamicHWKeepUnknown)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, -1, -1}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, -1, -1]");
}

TEST_F(Conv2DRuntimeInferShape, paddingSameDynamicHW)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    opt.padding = "SAME";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, -1, -1}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, -1, -1]");
}

TEST_F(Conv2DRuntimeInferShape, dynamicIcWithPaddingSame)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    opt.padding = "SAME";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, -1, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, xAllUnknownDim)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    opt.padding = "SAME";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({-1, -1, -1, -1}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[-1, 64, -1, -1]");
}

TEST_F(Conv2DRuntimeInferShape, filterAllUnknownDim)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    opt.padding = "VALID";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {-1, -1, -1, -1}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, -1, -1, -1]");
}

TEST_F(Conv2DRuntimeInferShape, khUnknownKeepOhUnknown)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, -1, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, -1, 16]");
}

TEST_F(Conv2DRuntimeInferShape, kwUnknownKeepOwUnknown)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, -1}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, -1]");
}

TEST_F(Conv2DRuntimeInferShape, unknownRankOutputAllUnknown)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({-2}, {64, 32, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[-1, -1, -1, -1]");
}

TEST_F(Conv2DRuntimeInferShape, dynamicBiasCSuccess)
{
    std::vector<int64_t> bias{-1};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, {}, &bias), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, unknownRankBiasSuccess)
{
    std::vector<int64_t> bias{-2};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, {}, &bias), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, groupsZeroFailed)
{
    Conv2DInferOpt opt;
    opt.groups = 0;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, groupsNegativeStaticFailed)
{
    Conv2DInferOpt opt;
    opt.groups = -1;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
    opt.groups = -2;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, groupsNegativeUnknownShapeSkip)
{
    Conv2DInferOpt opt;
    opt.groups = -1;
    opt.pads = {0, 0, 0, 0};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, -1, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, -1, 14]");
}

TEST_F(Conv2DRuntimeInferShape, offsetWNotEmptyFailed)
{
    std::vector<int64_t> bias{64};
    std::vector<int64_t> offsetW{1, 16, 1, 1};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, {}, &bias, &offsetW), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, xNhwcYNchwFailed)
{
    Conv2DInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.yFmt = ge::FORMAT_NCHW;
    opt.dataFormat = "NHWC";
    ASSERT_EQ(RunConv2DInfer({1, 16, 16, 32}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, cutPadsDynamicHStaticW)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 2, 2, 1};
    opt.pads = {1, 2, 1, 2};
    opt.xFmt = ge::FORMAT_NHWC;
    opt.wFmt = ge::FORMAT_HWCN;
    opt.yFmt = ge::FORMAT_NHWC;
    opt.dataFormat = "NHWC";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({4, -1, 1, 16}, {3, 3, 16, 1}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[4, -1, 1, 1]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeExplicit)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.pads = {0, 0, 0, 0};
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 1, 1]");
    ASSERT_EQ(yMax, "[4, 64, 4, 9]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeSame)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.pads = {0, 0, 0, 0};
    opt.padding = "SAME";
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 1, 1]");
    ASSERT_EQ(yMax, "[4, 64, 5, 10]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeCFromFilterN)
{
    Conv2DInferOpt opt;
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 8, 8}, {4, 32, 8, 8}, {8, 32, 3, 3}, {128, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 8, 8, 8]");
    ASSERT_EQ(yMax, "[4, 128, 8, 8]");
}

TEST_F(Conv2DRuntimeInferShape, groupsImplicitRewriteSuccess)
{
    Conv2DInferOpt opt;
    opt.groups = 1;
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 8, 16, 16}, {8, 4, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 8, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, dynamicIcGroupsFill)
{
    Conv2DInferOpt opt;
    opt.groups = 2;
    opt.pads = {0, 0, 0, 0};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, -1, 16, 16}, {8, 4, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 8, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, kcUnknownSkipGroupsCheck)
{
    Conv2DInferOpt opt;
    opt.groups = 2;
    opt.pads = {0, 0, 0, 0};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {8, -1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 8, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, knUnknownSkipGroupsDivisible)
{
    Conv2DInferOpt opt;
    opt.groups = 2;
    opt.pads = {0, 0, 0, 0};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {-1, 16, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, -1, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, paddingExplicitSuccess)
{
    Conv2DInferOpt opt;
    opt.padding = "EXPLICIT";
    opt.pads = {0, 0, 0, 0};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, autoPadNotSetPadsNot4D)
{
    Conv2DInferOpt opt;
    opt.autoPad = "NOTSET";
    opt.pads = {1, 1};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, padsNot4DWithoutOverride)
{
    Conv2DInferOpt opt;
    opt.pads = {1, 1};
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, paddingSameOverriddenByAutoPadValid)
{
    Conv2DInferOpt opt;
    opt.pads = {-1, -1, -1, -1};
    opt.padding = "SAME";
    opt.autoPad = "VALID";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 14, 14]");
}

TEST_F(Conv2DRuntimeInferShape, paddingValidOverriddenByAutoPadSameUpper)
{
    Conv2DInferOpt opt;
    opt.pads = {-1, -1, -1, -1};
    opt.padding = "VALID";
    opt.autoPad = "SAME_UPPER";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, filterUnknownRankOutputAllUnknown)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {-2}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[-1, -1, -1, -1]");
}

TEST_F(Conv2DRuntimeInferShape, bothUnknownRankOutputAllUnknown)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({-2}, {-2}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[-1, -1, -1, -1]");
}

TEST_F(Conv2DRuntimeInferShape, xDimNumNot4)
{
    ASSERT_EQ(RunConv2DInfer({1, 32, 16}, {64, 32, 3, 3}, nullptr), ge::GRAPH_FAILED);
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16, 1}, {64, 32, 3, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, wDimNumNot4)
{
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, groupsZeroDynamicFailed)
{
    Conv2DInferOpt opt;
    opt.groups = 0;
    ASSERT_EQ(RunConv2DInfer({1, 32, -1, 16}, {64, 32, 3, 3}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, unknownRankSkipsIllegalGroups)
{
    Conv2DInferOpt opt;
    opt.groups = -1;
    std::string y;
    ASSERT_EQ(RunConv2DInfer({-2}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[-1, -1, -1, -1]");
}

TEST_F(Conv2DRuntimeInferShape, dynamicNegativePadSkipped)
{
    Conv2DInferOpt opt;
    opt.pads = {1, 1, 1, -1};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, -1, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, -1, 14]");
}

TEST_F(Conv2DRuntimeInferShape, dynamicBatch)
{
    std::string y;
    ASSERT_EQ(RunConv2DInfer({-1, 32, 16, 16}, {64, 32, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[-1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, dynamicSkipsZeroTensorCheck)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({0, 32, -1, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[0, 64, -1, 14]");
}

TEST_F(Conv2DRuntimeInferShape, biasSkippedWhenOutChannelUnknown)
{
    std::vector<int64_t> bias{7};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {-1, 32, 3, 3}, &y, {}, &bias), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, -1, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, offsetWEmptySuccess)
{
    std::vector<int64_t> bias{64};
    std::vector<int64_t> offsetW;
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 16, 16}, {64, 32, 3, 3}, &y, {}, &bias, &offsetW), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 16, 16]");
}

TEST_F(Conv2DRuntimeInferShape, nhwcXHwcnW)
{
    Conv2DInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.wFmt = ge::FORMAT_HWCN;
    opt.yFmt = ge::FORMAT_NHWC;
    opt.dataFormat = "NHWC";
    opt.strides = {1, 1, 1, 1};
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 16, 16, 32}, {3, 3, 32, 64}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 16, 16, 64]");
}

TEST_F(Conv2DRuntimeInferShape, sameStride2OddH)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 2, 1};
    opt.pads = {0, 0, 0, 0};
    opt.padding = "SAME";
    std::string y;
    ASSERT_EQ(RunConv2DInfer({1, 32, 5, 16}, {64, 32, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 64, 3, 16]");
}

TEST_F(Conv2DRuntimeInferShape, nullInferShapeContext)
{
    auto* inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    ASSERT_EQ(inferShapeFunc(nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, nullInferShapeRangeContext)
{
    auto* inferRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->infer_shape_range;
    ASSERT_NE(inferRangeFunc, nullptr);
    ASSERT_EQ(inferRangeFunc(nullptr), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeSameUpper)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.pads = {0, 0, 0, 0};
    opt.autoPad = "SAME_UPPER";
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 1, 1]");
    ASSERT_EQ(yMax, "[4, 64, 5, 10]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeSameLower)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.pads = {0, 0, 0, 0};
    opt.autoPad = "SAME_LOWER";
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 1, 1]");
    ASSERT_EQ(yMax, "[4, 64, 5, 10]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeValidIgnoresPads)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.pads = {10, 10, 10, 10};
    opt.autoPad = "VALID";
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 1, 1]");
    ASSERT_EQ(yMax, "[4, 64, 4, 9]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangePaddingValidIgnoresPads)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.pads = {10, 10, 10, 10};
    opt.padding = "VALID";
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 1, 1]");
    ASSERT_EQ(yMax, "[4, 64, 4, 9]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeKnAllUnknown)
{
    Conv2DInferOpt opt;
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 8, 8}, {4, 32, 8, 8}, {-1, 32, 3, 3}, {-1, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 1, 8, 8]");
    ASSERT_EQ(yMax, "[4, -1, 8, 8]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeHighUnknown)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 1, 1}, {2, 32, -1, 8}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 1, 1]");
    ASSERT_EQ(yMax, "[2, 64, -1, 6]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeLowZero)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 0, 0}, {1, 32, 8, 8}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 0, 0]");
    ASSERT_EQ(yMax, "[1, 64, 6, 6]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeNhwc)
{
    Conv2DInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.yFmt = ge::FORMAT_NHWC;
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 8, 8, 32}, {4, 8, 8, 32}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 8, 8, 64]");
    ASSERT_EQ(yMax, "[4, 8, 8, 64]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeHwcnFilter)
{
    Conv2DInferOpt opt;
    opt.wFmt = ge::FORMAT_HWCN;
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 8, 8}, {1, 32, 8, 8}, {3, 3, 32, 8}, {3, 3, 32, 128}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 8, 8, 8]");
    ASSERT_EQ(yMax, "[1, 128, 8, 8]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeXFormatFailed)
{
    Conv2DInferOpt opt;
    opt.xFmt = ge::FORMAT_CHWN;
    opt.yFmt = ge::FORMAT_CHWN;
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 8, 8}, {1, 32, 8, 8}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeFilterFormatFailed)
{
    Conv2DInferOpt opt;
    opt.wFmt = ge::FORMAT_CHWN;
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 8, 8}, {1, 32, 8, 8}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeXDimNumFailed)
{
    Conv2DInferOpt opt;
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32}, {4, 32}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt), ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeStrideNot4D)
{
    Conv2DInferOpt opt;
    opt.strides = {1, 1, 1};
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 8, 8}, {1, 32, 8, 8}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_FAILED);
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeDilation)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    opt.dilations = {1, 1, 2, 2};
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 5, 5}, {1, 32, 10, 10}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 1, 1]");
    ASSERT_EQ(yMax, "[1, 64, 6, 6]");
}

TEST_F(Conv2DRuntimeInferShape, inferShapeRangeKernelLowZero)
{
    Conv2DInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunConv2DRange({1, 32, 0, 0}, {1, 32, 8, 8}, {64, 32, 0, 0}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 64, 0, 0]");
    ASSERT_EQ(yMax, "[1, 64, 6, 6]");
}
