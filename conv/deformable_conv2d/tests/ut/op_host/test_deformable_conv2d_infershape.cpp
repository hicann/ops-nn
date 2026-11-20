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
 * \file test_deformable_conv2d_infershape.cpp
 * \brief 覆盖DeformableConv2D InferShape校验规格：
 *        静态/动态/unknown_rank shape推导；x/filter格式约束（filter仅NCHW/HWCN）；
 *        strides/dilations/pads属性校验；groups/deformable_groups/modulated约束；
 *        offsets形状校验（通道=dfm_groups*kh*kw*3）；bias约束；零张量检查；
 *        InferShapeRange输出range推导。
 */

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

#include <string>
#include <utility>
#include <vector>

class DeformableConv2DRuntimeInferShape : public testing::Test {};

struct DfmInferOpt {
    std::vector<int64_t> strides{1, 1, 1, 1};
    std::vector<int64_t> pads{0, 0, 0, 0};
    std::vector<int64_t> dilations{1, 1, 1, 1};
    int64_t groups = 1;
    const char* dataFormat = "NCHW";
    int64_t deformableGroups = 1;
    bool modulated = true;
    ge::Format xFmt = ge::FORMAT_NCHW;
    ge::Format wFmt = ge::FORMAT_NCHW;
    ge::Format offsetsFmt = ge::FORMAT_NCHW;
    ge::Format yFmt = ge::FORMAT_NCHW;
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType wDtype = ge::DT_FLOAT16;
};

// 属性顺序与proto声明一致：strides(0) pads(1) dilations(2) groups(3) data_format(4)
// deformable_groups(5) modulated(6)
static std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeDfmAttrs(const DfmInferOpt& opt)
{
    return {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.strides)},
            {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.pads)},
            {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.dilations)},
            {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(opt.groups)},
            {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(opt.dataFormat)},
            {"deformable_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(opt.deformableGroups)},
            {"modulated", Ops::NN::AnyValue::CreateFrom<bool>(opt.modulated)}};
}

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t d : dims) {
        shape.MutableOriginShape().AppendDim(d);
    }
    return shape;
}

static ge::graphStatus RunDfmInfer(const std::vector<int64_t>& xDims, const std::vector<int64_t>& wDims,
                                   const std::vector<int64_t>& offsetsDims, std::string* yStr,
                                   const DfmInferOpt& opt = {}, const std::vector<int64_t>* biasDims = nullptr,
                                   const std::string& testName = "fakeOp")
{
    gert::StorageShape xShape = MakeStorageShape(xDims);
    gert::StorageShape wShape = MakeStorageShape(wDims);
    gert::StorageShape offsetsShape = MakeStorageShape(offsetsDims);
    gert::StorageShape yShape;
    gert::StorageShape biasShape;
    std::vector<gert::StorageShape*> inputs{&xShape, &wShape, &offsetsShape};
    if (biasDims != nullptr) {
        biasShape = MakeStorageShape(*biasDims);
        inputs.push_back(&biasShape);
    }

    gert::InferShapeContextFaker faker;
    faker.SetOpType(testName);
    faker.NodeIoNum(inputs.size(), 1).IrInstanceNum(std::vector<uint32_t>(inputs.size(), 1u));
    faker.NodeInputTd(0, opt.xDtype, opt.xFmt, ge::FORMAT_RESERVED);
    faker.NodeInputTd(1, opt.wDtype, opt.wFmt, ge::FORMAT_RESERVED);
    faker.NodeInputTd(2, opt.xDtype, opt.offsetsFmt, ge::FORMAT_RESERVED);
    if (biasDims != nullptr) {
        faker.NodeInputTd(3, opt.xDtype, ge::FORMAT_ND, ge::FORMAT_RESERVED);
    }
    faker.NodeOutputTd(0, opt.xDtype, opt.yFmt, ge::FORMAT_RESERVED);
    faker.NodeAttrs(MakeDfmAttrs(opt)).InputShapes(inputs).OutputShapes({&yShape});
    auto holder = faker.Build();
    auto* inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DeformableConv2D")->infer_shape;
    const ge::graphStatus ret = inferShapeFunc(holder.GetContext<gert::InferShapeContext>());
    if (yStr != nullptr && ret == ge::GRAPH_SUCCESS) {
        *yStr = Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0));
    }
    return ret;
}

static ge::graphStatus RunDfmRange(const gert::Shape& xMin, const gert::Shape& xMax, const gert::Shape& wMin,
                                   const gert::Shape& wMax, std::string* yMinStr, std::string* yMaxStr,
                                   const DfmInferOpt& opt = {})
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
                      .NodeAttrs(MakeDfmAttrs(opt))
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .Build();
    auto* inferRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DeformableConv2D")->infer_shape_range;
    const ge::graphStatus ret = inferRangeFunc(holder.GetContext<gert::InferShapeRangeContext>());
    if (ret == ge::GRAPH_SUCCESS) {
        *yMinStr = Ops::Base::ToString(
            *holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin());
        *yMaxStr = Ops::Base::ToString(
            *holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax());
    }
    return ret;
}

// pointer-based variant: a null pointer builds a half-set/empty range for the null-check UTs
static ge::graphStatus RunDfmRangeRaw(gert::Shape* xMin, gert::Shape* xMax, gert::Shape* wMin, gert::Shape* wMax,
                                      gert::Shape* yMin, gert::Shape* yMax, const DfmInferOpt& opt = {})
{
    gert::Range<gert::Shape> xRange(xMin, xMax);
    gert::Range<gert::Shape> wRange(wMin, wMax);
    gert::Range<gert::Shape> yRange(yMin, yMax);
    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, opt.xFmt, ge::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, opt.wFmt, ge::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, opt.yFmt, ge::FORMAT_RESERVED)
                      .NodeAttrs(MakeDfmAttrs(opt))
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .Build();
    auto* inferRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DeformableConv2D")->infer_shape_range;
    return inferRangeFunc(holder.GetContext<gert::InferShapeRangeContext>());
}

// ==================== 静态shape推导 ====================

TEST_F(DeformableConv2DRuntimeInferShape, staticNchwSuccess)
{
    std::string y;
    // dfg=1, kh=kw=3 → offsets channel = 1*3*3*3 = 27
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, &y), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, 6, 6]");
}

TEST_F(DeformableConv2DRuntimeInferShape, staticNhwcHwcnSuccess)
{
    std::string y;
    DfmInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.wFmt = ge::FORMAT_HWCN;
    opt.offsetsFmt = ge::FORMAT_NHWC;
    opt.yFmt = ge::FORMAT_NHWC;
    ASSERT_EQ(RunDfmInfer({1, 8, 8, 4}, {3, 3, 4, 4}, {1, 6, 6, 27}, &y, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 6, 6, 4]");
}

TEST_F(DeformableConv2DRuntimeInferShape, stride2PadsSuccess)
{
    std::string y;
    DfmInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.pads = {1, 1, 1, 1};
    // oh = (16 + 1 + 1 - 2 - 1) / 2 + 1 = 8
    ASSERT_EQ(RunDfmInfer({1, 4, 16, 16}, {4, 4, 3, 3}, {1, 27, 8, 8}, &y, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, 8, 8]");
}

TEST_F(DeformableConv2DRuntimeInferShape, dilation2Success)
{
    std::string y;
    DfmInferOpt opt;
    opt.dilations = {1, 1, 2, 2};
    // oh = (12 - 2*2 - 1) / 1 + 1 = 8
    ASSERT_EQ(RunDfmInfer({1, 4, 12, 12}, {4, 4, 3, 3}, {1, 27, 8, 8}, &y, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, 8, 8]");
}

TEST_F(DeformableConv2DRuntimeInferShape, withBiasSuccess)
{
    std::string y;
    const std::vector<int64_t> bias = {4};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, &y, {}, &bias), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, 6, 6]");
}

TEST_F(DeformableConv2DRuntimeInferShape, groupsSuccess)
{
    std::string y;
    DfmInferOpt opt;
    opt.groups = 2;
    // ic=4 == kc=2 * groups=2, kn=4 % groups=2 == 0
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 2, 3, 3}, {1, 27, 6, 6}, &y, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, 6, 6]");
}

TEST_F(DeformableConv2DRuntimeInferShape, deformableGroups2Success)
{
    std::string y;
    DfmInferOpt opt;
    opt.deformableGroups = 2;
    // ic=4 % dfg=2 == 0, offsets channel = 2*3*3*3 = 54
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 54, 6, 6}, &y, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, 6, 6]");
}

TEST_F(DeformableConv2DRuntimeInferShape, zeroTensorInputHSuccess)
{
    std::string y;
    // kh=kw=1 → oh=0, y keeps the zero dim
    ASSERT_EQ(RunDfmInfer({1, 4, 0, 5}, {4, 4, 1, 1}, {1, 3, 0, 5}, &y), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, 0, 5]");
}

TEST_F(DeformableConv2DRuntimeInferShape, zeroTensorFilterNSuccess)
{
    std::string y;
    // kn=0 → y C=0
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {0, 4, 3, 3}, {1, 27, 6, 6}, &y), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 0, 6, 6]");
}

// ==================== 动态/unknown rank ====================

TEST_F(DeformableConv2DRuntimeInferShape, unknownRankSuccess)
{
    std::string y;
    ASSERT_EQ(RunDfmInfer({-2}, {4, 4, 3, 3}, {1, 27, 6, 6}, &y), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[-1, 4, -1, -1]");
}

TEST_F(DeformableConv2DRuntimeInferShape, dynamicHwSuccess)
{
    std::string y;
    ASSERT_EQ(RunDfmInfer({1, 4, -1, -1}, {4, 4, 3, 3}, {1, 27, -1, -1}, &y), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, -1, -1]");
}

TEST_F(DeformableConv2DRuntimeInferShape, dynamicOffsetsChannelUnknownSuccess)
{
    std::string y;
    ASSERT_EQ(RunDfmInfer({1, 4, -1, -1}, {4, 4, 3, 3}, {1, -1, -1, -1}, &y), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, -1, -1]");
}

TEST_F(DeformableConv2DRuntimeInferShape, dynamicOffsetsChannelMatchedSuccess)
{
    std::string y;
    ASSERT_EQ(RunDfmInfer({1, 4, -1, -1}, {4, 4, 3, 3}, {1, 27, -1, -1}, &y), ge::GRAPH_SUCCESS);
    EXPECT_EQ(y, "[1, 4, -1, -1]");
}

// ==================== 输入约束 ====================

TEST_F(DeformableConv2DRuntimeInferShape, xNot4dFail)
{
    ASSERT_EQ(RunDfmInfer({1, 4, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, xFormatNdFail)
{
    DfmInferOpt opt;
    opt.xFmt = ge::FORMAT_ND;
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, filterNot4dFail)
{
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 3, 3}, {1, 27, 6, 6}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, filterFormatNhwcFail)
{
    // filter accepts only NCHW and HWCN
    DfmInferOpt opt;
    opt.wFmt = ge::FORMAT_NHWC;
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, filterKhZeroFail)
{
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 0, 3}, {1, 27, 6, 6}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, dtypeMismatchFail)
{
    DfmInferOpt opt;
    opt.wDtype = ge::DT_FLOAT;
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, biasNot1dFail)
{
    const std::vector<int64_t> bias = {1, 4};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, {}, &bias), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, biasSizeMismatchFail)
{
    const std::vector<int64_t> bias = {3};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, {}, &bias), ge::GRAPH_FAILED);
}

// ==================== 属性约束 ====================

TEST_F(DeformableConv2DRuntimeInferShape, stridesSize3Fail)
{
    DfmInferOpt opt;
    opt.strides = {1, 1, 1};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, strideHZeroFail)
{
    DfmInferOpt opt;
    opt.strides = {1, 1, 0, 1};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, strideNNot1Fail)
{
    DfmInferOpt opt;
    opt.strides = {2, 1, 1, 1};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, dilationHZeroFail)
{
    DfmInferOpt opt;
    opt.dilations = {1, 1, 0, 1};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, dilationCNot1Fail)
{
    DfmInferOpt opt;
    opt.dilations = {1, 2, 1, 1};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, dilationSize3Fail)
{
    DfmInferOpt opt;
    opt.dilations = {1, 1, 1};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, padsNegativeFail)
{
    DfmInferOpt opt;
    opt.pads = {-1, 0, 0, 0};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, padsSize3Fail)
{
    DfmInferOpt opt;
    opt.pads = {0, 0, 0};
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

// ==================== groups/deformable_groups/modulated 约束 ====================

TEST_F(DeformableConv2DRuntimeInferShape, groupsChannelMismatchFail)
{
    DfmInferOpt opt;
    opt.groups = 1;
    // ic=4 != kc=3 * 1
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 3, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, knNotDivisibleByGroupsFail)
{
    DfmInferOpt opt;
    opt.groups = 3;
    // kn=4 % 3 != 0
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, deformableGroupsZeroFail)
{
    DfmInferOpt opt;
    opt.deformableGroups = 0;
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, deformableGroupsNotDivideFail)
{
    DfmInferOpt opt;
    opt.deformableGroups = 3;
    // ic=4 % 3 != 0
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, modulatedFalseFail)
{
    DfmInferOpt opt;
    opt.modulated = false;
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

// ==================== offsets 约束 ====================

TEST_F(DeformableConv2DRuntimeInferShape, offsetsNot4dFail)
{
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, unknownRankOffsetsNot4dFail)
{
    // offsets 4D check applies even when x is unknown rank
    ASSERT_EQ(RunDfmInfer({-2}, {4, 4, 3, 3}, {1, 27, 6}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, offsetsFormatNdFail)
{
    DfmInferOpt opt;
    opt.offsetsFmt = ge::FORMAT_ND;
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, offsetsChannelMismatchFail)
{
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 26, 6, 6}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, offsetsHMismatchFail)
{
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 5, 6}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, dynamicOffsetsChannelMismatchFail)
{
    // dynamic: only the channel dim is checked, 26 != 27 and 26 != -1
    ASSERT_EQ(RunDfmInfer({1, 4, -1, -1}, {4, 4, 3, 3}, {1, 26, -1, -1}, nullptr), ge::GRAPH_FAILED);
}

// ==================== 零张量与padding覆盖约束 ====================

TEST_F(DeformableConv2DRuntimeInferShape, zeroTensorNegativeOutputFail)
{
    // x h=0 with kh=3 → oh < 0, zero tensor input infers a negative output
    ASSERT_EQ(RunDfmInfer({1, 4, 0, 8}, {4, 4, 3, 3}, {1, 27, 0, 6}, nullptr), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, inputWithPadFail)
{
    // ih=2 with kh=3 → ihPad < 0, image size after padding is smaller than the filter
    ASSERT_EQ(RunDfmInfer({1, 4, 2, 2}, {4, 4, 3, 3}, {1, 27, 0, 0}, nullptr), ge::GRAPH_FAILED);
}

// ==================== InferShapeRange ====================

TEST_F(DeformableConv2DRuntimeInferShape, rangeNchwSuccess)
{
    gert::Shape xMin = gert::Shape({1, 4, 1, 1});
    gert::Shape xMax = gert::Shape({1, 4, 100, 100});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yMin, "[1, 4, 1, 1]");
    EXPECT_EQ(yMax, "[1, 4, 98, 98]");
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeStride2WithPadsSuccess)
{
    DfmInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.pads = {1, 1, 1, 1};
    gert::Shape xMin = gert::Shape({1, 4, 4, 4});
    gert::Shape xMax = gert::Shape({1, 4, 200, 200});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    // low = (4 + 2 - 2 - 1) / 2 + 1 = 2, high = (200 + 2 - 2 - 1) / 2 + 1 = 100
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yMin, "[1, 4, 2, 2]");
    EXPECT_EQ(yMax, "[1, 4, 100, 100]");
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeZeroTensorLowerBoundSuccess)
{
    gert::Shape xMin = gert::Shape({1, 4, 0, 0});
    gert::Shape xMax = gert::Shape({1, 4, 10, 10});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    // inLow=0 keeps the lower bound at 0 instead of 1
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yMin, "[1, 4, 0, 0]");
    EXPECT_EQ(yMax, "[1, 4, 8, 8]");
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeUnknownHighKeptSuccess)
{
    gert::Shape xMin = gert::Shape({1, 4, 1, 1});
    gert::Shape xMax = gert::Shape({1, 4, -1, -1});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yMin, "[1, 4, 1, 1]");
    EXPECT_EQ(yMax, "[1, 4, -1, -1]");
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeNhwcHwcnSuccess)
{
    DfmInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.wFmt = ge::FORMAT_HWCN;
    opt.yFmt = ge::FORMAT_NHWC;
    gert::Shape xMin = gert::Shape({1, 1, 1, 4});
    gert::Shape xMax = gert::Shape({1, 100, 100, 4});
    gert::Shape wMin = gert::Shape({3, 3, 4, 4});
    gert::Shape wMax = gert::Shape({3, 3, 4, 4});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yMin, "[1, 1, 1, 4]");
    EXPECT_EQ(yMax, "[1, 98, 98, 4]");
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeStrideZeroFail)
{
    DfmInferOpt opt;
    opt.strides = {1, 1, 0, 1};
    gert::Shape xMin = gert::Shape({1, 4, 1, 1});
    gert::Shape xMax = gert::Shape({1, 4, 10, 10});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeStrideNegativeFail)
{
    DfmInferOpt opt;
    opt.strides = {1, 1, -2, 1};
    gert::Shape xMin = gert::Shape({1, 4, 1, 1});
    gert::Shape xMax = gert::Shape({1, 4, 10, 10});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeDilationNegativeFail)
{
    DfmInferOpt opt;
    opt.dilations = {1, 1, -1, 1};
    gert::Shape xMin = gert::Shape({1, 4, 1, 1});
    gert::Shape xMax = gert::Shape({1, 4, 10, 10});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax, opt), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeStrideNcNotOneFail)
{
    DfmInferOpt opt;
    opt.strides = {1, 2, 1, 1};
    gert::Shape xMin = gert::Shape({1, 4, 1, 1});
    gert::Shape xMax = gert::Shape({1, 4, 10, 10});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax, opt), ge::GRAPH_FAILED);
}

// Partial-dynamic coverage: H static, W dynamic; the H range uses the inferred output oh, not the input H.
TEST_F(DeformableConv2DRuntimeInferShape, rangeHStaticWDynamicSuccess)
{
    gert::Shape xMin = gert::Shape({1, 4, 8, 2});
    gert::Shape xMax = gert::Shape({1, 4, 8, 12});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax), ge::GRAPH_SUCCESS);
}

// Partial-dynamic coverage: H dynamic, W static. ow uses the static output dim.
TEST_F(DeformableConv2DRuntimeInferShape, rangeHDynamicWStaticSuccess)
{
    gert::Shape xMin = gert::Shape({1, 4, 2, 8});
    gert::Shape xMax = gert::Shape({1, 4, 12, 8});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax), ge::GRAPH_SUCCESS);
}

// non-4D x is rejected before the range derivation
TEST_F(DeformableConv2DRuntimeInferShape, rangeXDimNumFail)
{
    gert::Shape xMin = gert::Shape({1, 4, 8});
    gert::Shape xMax = gert::Shape({1, 4, 10});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax), ge::GRAPH_FAILED);
}

// non-4D filter is rejected
TEST_F(DeformableConv2DRuntimeInferShape, rangeFilterDimNumFail)
{
    gert::Shape xMin = gert::Shape({1, 4, 1, 1});
    gert::Shape xMax = gert::Shape({1, 4, 10, 10});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax), ge::GRAPH_FAILED);
}

// unknown rank ([-2]) skips the range derivation
TEST_F(DeformableConv2DRuntimeInferShape, rangeXUnknownRankSkipSuccess)
{
    gert::Shape xUnknownRank = gert::Shape({-2});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    gert::Shape yMin{};
    gert::Shape yMax{};
    ASSERT_EQ(RunDfmRangeRaw(&xUnknownRank, &xUnknownRank, &wMin, &wMax, &yMin, &yMax), ge::GRAPH_SUCCESS);
}

// a half-set x range is malformed
TEST_F(DeformableConv2DRuntimeInferShape, rangeXMinOnlyFail)
{
    gert::Shape xMin = gert::Shape({1, 4, 8, 2});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    gert::Shape yMin{};
    gert::Shape yMax{};
    ASSERT_EQ(RunDfmRangeRaw(&xMin, nullptr, &wMin, &wMax, &yMin, &yMax), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeXMaxOnlyFail)
{
    gert::Shape xMax = gert::Shape({1, 4, 8, 12});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    gert::Shape yMin{};
    gert::Shape yMax{};
    ASSERT_EQ(RunDfmRangeRaw(nullptr, &xMax, &wMin, &wMax, &yMin, &yMax), ge::GRAPH_FAILED);
}

TEST_F(DeformableConv2DRuntimeInferShape, rangeFilterFormatFail)
{
    DfmInferOpt opt;
    opt.wFmt = ge::FORMAT_ND;
    gert::Shape xMin = gert::Shape({1, 4, 1, 1});
    gert::Shape xMax = gert::Shape({1, 4, 10, 10});
    gert::Shape wMin = gert::Shape({4, 4, 3, 3});
    gert::Shape wMax = gert::Shape({4, 4, 3, 3});
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDfmRange(xMin, xMax, wMin, wMax, &yMin, &yMax, opt), ge::GRAPH_FAILED);
}

// groups=0 is rejected before any multiply/modulo to guard against division by zero
TEST_F(DeformableConv2DRuntimeInferShape, groupsZeroFail)
{
    DfmInferOpt opt;
    opt.groups = 0;
    ASSERT_EQ(RunDfmInfer({1, 4, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}

// groups=0 with dynamic x C=-1 must also be rejected (regression guard for the modulo crash path)
TEST_F(DeformableConv2DRuntimeInferShape, groupsZeroDynamicCFail)
{
    DfmInferOpt opt;
    opt.groups = 0;
    ASSERT_EQ(RunDfmInfer({1, -1, 8, 8}, {4, 4, 3, 3}, {1, 27, 6, 6}, nullptr, opt), ge::GRAPH_FAILED);
}
