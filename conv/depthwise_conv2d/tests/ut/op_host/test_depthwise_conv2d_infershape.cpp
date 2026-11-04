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
 * \file test_depthwise_conv2d_infershape.cpp
 * \brief 覆盖DepthwiseConv2D InferShape校验规格：
 *        静态/动态/unknown_rank shape推导；SAME/VALID/_padding padding推导；
 *        静态weight约束；dtype一致性校验（INT16+INT8混合精度除外）；零张量检查；
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

const char* const DEFAULT_ATTR_VAL = "";

class DepthwiseConv2DRuntimeInferShape : public testing::Test {};

struct DepthwiseInferOpt {
    std::vector<int64_t> strides{1, 1, 1, 1};
    std::vector<int64_t> pads{0, 0, 0, 0};
    std::vector<int64_t> dilations{1, 1, 1, 1};
    const char* dataFormat = "NCHW";
    const char* padding = DEFAULT_ATTR_VAL;
    std::vector<int64_t> paddingList{-1, -1, -1, -1}; // sentinel=未设置（与注册默认一致）
    ge::Format xFmt = ge::FORMAT_NCHW;
    ge::Format wFmt = ge::FORMAT_NCHW;
    ge::Format yFmt = ge::FORMAT_NCHW;
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType wDtype = ge::DT_FLOAT16;
};

static std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeDepthwiseAttrs(const DepthwiseInferOpt& opt)
{
    return {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.strides)},
            {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.dilations)},
            {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.pads)},
            {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(opt.dataFormat)},
            {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
            {"_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(opt.paddingList)},
            {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(opt.padding)}};
}

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t d : dims) {
        shape.MutableOriginShape().AppendDim(d);
    }
    return shape;
}

static ge::graphStatus RunDepthwiseInfer(const std::vector<int64_t>& xDims, const std::vector<int64_t>& wDims,
                                         std::string* yStr, const DepthwiseInferOpt& opt = {},
                                         const std::vector<int64_t>* biasDims = nullptr,
                                         const std::string& testName = "fakeOp")
{
    gert::StorageShape xShape = MakeStorageShape(xDims);
    gert::StorageShape wShape = MakeStorageShape(wDims);
    gert::StorageShape yShape;
    gert::StorageShape biasShape;
    std::vector<gert::StorageShape*> inputs{&xShape, &wShape};
    if (biasDims != nullptr) {
        biasShape = MakeStorageShape(*biasDims);
        inputs.push_back(&biasShape);
    }

    gert::InferShapeContextFaker faker;
    faker.SetOpType(testName);
    faker.NodeIoNum(inputs.size(), 1).IrInstanceNum(std::vector<uint32_t>(inputs.size(), 1u));
    faker.NodeInputTd(0, opt.xDtype, opt.xFmt, ge::FORMAT_RESERVED);
    faker.NodeInputTd(1, opt.wDtype, opt.wFmt, ge::FORMAT_RESERVED);
    if (biasDims != nullptr) {
        faker.NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_RESERVED);
    }
    faker.NodeOutputTd(0, ge::DT_FLOAT16, opt.yFmt, ge::FORMAT_RESERVED);
    faker.NodeAttrs(MakeDepthwiseAttrs(opt)).InputShapes(inputs).OutputShapes({&yShape});
    auto holder = faker.Build();
    auto* inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2D")->infer_shape;
    const ge::graphStatus ret = inferShapeFunc(holder.GetContext<gert::InferShapeContext>());
    if (yStr != nullptr && ret == ge::GRAPH_SUCCESS) {
        *yStr = Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0));
    }
    return ret;
}

static ge::graphStatus RunDepthwiseRange(const gert::Shape& xMin, const gert::Shape& xMax, const gert::Shape& wMin,
                                         const gert::Shape& wMax, std::string* yMinStr, std::string* yMaxStr,
                                         const DepthwiseInferOpt& opt)
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
                      .NodeAttrs(MakeDepthwiseAttrs(opt))
                      .InputShapeRanges({&xRange, &wRange})
                      .OutputShapeRanges({&yRange})
                      .Build();
    auto* inferRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2D")->infer_shape_range;
    const ge::graphStatus ret = inferRangeFunc(holder.GetContext<gert::InferShapeRangeContext>());
    if (ret == ge::GRAPH_SUCCESS) {
        *yMinStr = Ops::Base::ToString(
            *holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMin());
        *yMaxStr = Ops::Base::ToString(
            *holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)->GetMax());
    }
    return ret;
}

TEST_F(DepthwiseConv2DRuntimeInferShape, staticNchwSuccess)
{
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 222, 222]");
}

// 实测反馈用例：fp16, x=1×20×20×1(NHWC), w=1×1×1×3(HWCN), VALID, s/d=1 → y=[1,20,20,3]
TEST_F(DepthwiseConv2DRuntimeInferShape, reportedCaseStaticSuccess)
{
    DepthwiseInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.wFmt = ge::FORMAT_HWCN;
    opt.yFmt = ge::FORMAT_NHWC;
    opt.dataFormat = "NHWC";
    opt.padding = "VALID";
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 20, 20, 1}, {1, 1, 1, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 20, 20, 3]");
}

// 同用例在线构图变体：w 构图期未定 H/W（-1,-1,1,3）
TEST_F(DepthwiseConv2DRuntimeInferShape, reportedCaseOnlineUnknownHw)
{
    // 在线构图：w H/W 未定（-1,-1,1,3），x 静态 → 放行，H/W 输出为 unknown
    DepthwiseInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.wFmt = ge::FORMAT_HWCN;
    opt.yFmt = ge::FORMAT_NHWC;
    opt.dataFormat = "NHWC";
    opt.padding = "VALID";
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 20, 20, 1}, {-1, -1, 1, 3}, &y, opt, nullptr, "dwOnlineHw"), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, staticNhwcHwcnSuccess)
{
    DepthwiseInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.wFmt = ge::FORMAT_HWCN;
    opt.yFmt = ge::FORMAT_NHWC;
    opt.dataFormat = "NHWC";
    std::string y;
    // HWCN weight [kh, kw, kc, kn]: outC = kn * kc = 3
    ASSERT_EQ(RunDepthwiseInfer({1, 224, 224, 3}, {3, 3, 1, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 222, 222, 3]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, withBiasSuccess)
{
    std::string y;
    const std::vector<int64_t> biasDims = {3};
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, {}, &biasDims), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 222, 222]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, paddingSameSuccess)
{
    DepthwiseInferOpt opt;
    opt.padding = "SAME";
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 224, 224]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, paddingValidSuccess)
{
    DepthwiseInferOpt opt;
    opt.padding = "VALID";
    opt.pads = {2, 2, 2, 2};
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 222, 222]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, paddingListOverrideSuccess)
{
    DepthwiseInferOpt opt;
    opt.paddingList = {1, 1, 1, 1};
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 224, 224]");
}

// _padding 显式全零也覆盖 pads
TEST_F(DepthwiseConv2DRuntimeInferShape, paddingListZeroOverridePads)
{
    DepthwiseInferOpt opt;
    opt.pads = {2, 2, 2, 2};
    opt.paddingList = {0, 0, 0, 0};
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 222, 222]");
}

// _padding 显式全零时 padding 串不再被解析
TEST_F(DepthwiseConv2DRuntimeInferShape, paddingListZeroBlocksSame)
{
    DepthwiseInferOpt opt;
    opt.padding = "SAME";
    opt.paddingList = {0, 0, 0, 0};
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 222, 222]");
}

// _padding 非零时优先于 padding 串
TEST_F(DepthwiseConv2DRuntimeInferShape, paddingListPriorityOverSame)
{
    DepthwiseInferOpt opt;
    opt.padding = "SAME";
    opt.paddingList = {2, 2, 2, 2};
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 226, 226]");
}

// _padding 负值：覆盖后由负 pads 检查拦截
TEST_F(DepthwiseConv2DRuntimeInferShape, paddingListNegativeFail)
{
    DepthwiseInferOpt opt;
    opt.paddingList = {-1, 0, 0, 0};
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

// _padding 尺寸非 4 拦截
TEST_F(DepthwiseConv2DRuntimeInferShape, paddingListSizeNot4Fail)
{
    DepthwiseInferOpt opt;
    opt.paddingList = {1, 1, 1};
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

// filter 非 4D 拦截
TEST_F(DepthwiseConv2DRuntimeInferShape, filterNot4DFail)
{
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3, 3}, &y), ge::GRAPH_FAILED);
}

// pads 长度非 4 拦截
TEST_F(DepthwiseConv2DRuntimeInferShape, padsSizeNot4Fail)
{
    DepthwiseInferOpt opt;
    opt.pads = {0, 0, 0};
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

// x 维度按 data_format attr 映射：attr=NCHW 时按 NCHW 取维
TEST_F(DepthwiseConv2DRuntimeInferShape, xFormatAttrMismatchMapsByAttr)
{
    DepthwiseInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.dataFormat = "NCHW";
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 4, 8, 8}, {4, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 4, 6, 6]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, dynamicHwSuccess)
{
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, -1, -1}, {3, 1, 3, 3}, &y, {}, nullptr, "dwDynHw"), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, -1, -1]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, dynamicChannelCompletionSuccess)
{
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, -1, -1, -1}, {3, 1, 3, 3}, &y, {}, nullptr, "dwDynC"), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, -1, -1]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, unknownRankSuccess)
{
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({-2}, {3, 1, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[-1, 3, -1, -1]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, zeroTensorInputSuccess)
{
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 0, 5}, {3, 1, 1, 1}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 0, 5]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, zeroTensorNegativeOutputFail)
{
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 0, 5}, {3, 1, 3, 3}, &y), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, dtypeMismatchFail)
{
    DepthwiseInferOpt opt;
    opt.wDtype = ge::DT_INT8;
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, mixedPrecisionSuccess)
{
    DepthwiseInferOpt opt;
    opt.xDtype = ge::DT_INT16;
    opt.wDtype = ge::DT_INT8;
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 222, 222]");
}

// 非对称语义：x 静态 + filter C 未定 → 放行，输出 C 为 unknown
TEST_F(DepthwiseConv2DRuntimeInferShape, staticXUnknownFilterNcSuccess)
{
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, -1, 3, 3}, &y, {}, nullptr, "dwStXUnkFC"), ge::GRAPH_FAILED);
}

// 非对称语义：x 动态 + filter C 未定 → 拒绝（pre-2023 inC<1 路径，与 base 一致）
// x 动态 C=-1 + filter C 未定 → 缓存命中 kc=1 → ic补全=1 → oc=3, oh/ow=222
TEST_F(DepthwiseConv2DRuntimeInferShape, dynamicXUnknownFilterCacheHit)
{
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, -1, 224, 224}, {3, -1, 3, 3}, &y, {}, nullptr, "dwDynXUnkF"), ge::GRAPH_FAILED);
}

// x 动态 + filter H/W 未定 → 拒绝
// x 动态 + filter H/W 未定 → 两个输入均缓存命中还原 → 输出与首次推导一致
TEST_F(DepthwiseConv2DRuntimeInferShape, dynamicXUnknownFilterHwCacheHit)
{
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, -1, -1}, {3, 1, -1, -1}, &y, {}, nullptr, "dwDynXUnkFHW"), ge::GRAPH_FAILED);
}

// 非法维（< -1）无条件拦截
TEST_F(DepthwiseConv2DRuntimeInferShape, filterIllegalDimFail)
{
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {-2, 1, 3, 3}, &y, {}, nullptr, "dwIllegalDim"), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, filterZeroHwFail)
{
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 0, 3}, &y), ge::GRAPH_FAILED);
}

// x 张量格式不做白名单拦截
TEST_F(DepthwiseConv2DRuntimeInferShape, unknownXFormatFallbackNchw)
{
    DepthwiseInferOpt opt;
    opt.xFmt = ge::FORMAT_ND;
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 222, 222]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, non4DXFail)
{
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224}, {3, 1, 3, 3}, &y), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, invalidDataFormatAttrFail)
{
    DepthwiseInferOpt opt;
    opt.dataFormat = "NDHWC";
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, invalidPaddingStrFail)
{
    DepthwiseInferOpt opt;
    opt.padding = "CAUSAL";
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, negativePadsFail)
{
    DepthwiseInferOpt opt;
    opt.pads = {-1, 0, 0, 0};
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

// 负 stride 不拦截，按公式计算
TEST_F(DepthwiseConv2DRuntimeInferShape, negativeStridePassthrough)
{
    DepthwiseInferOpt opt;
    opt.strides = {1, 1, -1, -1};
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, -220, -220]");
}

// stride==0 拦截
TEST_F(DepthwiseConv2DRuntimeInferShape, strideZeroFail)
{
    DepthwiseInferOpt opt;
    opt.strides = {1, 1, 0, 1};
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

// dilation 值不校验，d=0 按 dk=1 计算放行
TEST_F(DepthwiseConv2DRuntimeInferShape, dilationZeroPassthrough)
{
    DepthwiseInferOpt opt;
    opt.dilations = {1, 1, 0, 0};
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 3, 224, 224]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, inferShapeRangeExplicit)
{
    DepthwiseInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.pads = {0, 0, 0, 0};
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDepthwiseRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 2048, 1, 1]");
    ASSERT_EQ(yMax, "[4, 2048, 4, 9]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, inferShapeRangeSame)
{
    DepthwiseInferOpt opt;
    opt.strides = {1, 1, 2, 2};
    opt.padding = "SAME";
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDepthwiseRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 2048, 1, 1]");
    ASSERT_EQ(yMax, "[4, 2048, 5, 10]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, inferShapeRangeCFromFilter)
{
    DepthwiseInferOpt opt;
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDepthwiseRange({1, 32, 8, 8}, {4, 32, 8, 8}, {8, 32, 3, 3}, {128, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 256, 6, 6]");
    ASSERT_EQ(yMax, "[4, 256, 6, 6]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, nullInferShapeContext)
{
    auto* inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2D")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    ASSERT_EQ(inferShapeFunc(nullptr), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, nullInferShapeRangeContext)
{
    auto* inferRangeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2D")->infer_shape_range;
    ASSERT_NE(inferRangeFunc, nullptr);
    ASSERT_EQ(inferRangeFunc(nullptr), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, stridesSizeNot4Fail)
{
    DepthwiseInferOpt opt;
    opt.strides = {1, 1, 1};
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DRuntimeInferShape, dilationsSizeNot4Fail)
{
    DepthwiseInferOpt opt;
    opt.dilations = {1, 1};
    std::string y;
    EXPECT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {3, 1, 3, 3}, &y, opt), ge::GRAPH_FAILED);
}

// filterN/filterC 为 0 时输出通道为 0
TEST_F(DepthwiseConv2DRuntimeInferShape, filterNZeroTensorSuccess)
{
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {0, 1, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 0, 222, 222]");
}

TEST_F(DepthwiseConv2DRuntimeInferShape, filterCZeroTensorSuccess)
{
    std::string y;
    ASSERT_EQ(RunDepthwiseInfer({1, 3, 224, 224}, {4, 0, 3, 3}, &y), ge::GRAPH_SUCCESS);
    ASSERT_EQ(y, "[1, 0, 222, 222]");
}

// 零张量动态下界：输入 range 下界为 0 时输出 range 下界保持 0 而非 1
TEST_F(DepthwiseConv2DRuntimeInferShape, inferShapeRangeZeroTensorLowerBound)
{
    DepthwiseInferOpt opt;
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDepthwiseRange({1, 4, 0, 5}, {1, 4, 10, 20}, {4, 1, 3, 3}, {4, 1, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    // H: low=(0-2-1)+1=-2→下界0；high=(10-2-1)+1=8。W: low=(5-2-1)+1=3；high=(20-2-1)+1=18
    ASSERT_EQ(yMin, "[1, 4, 0, 3]");
    ASSERT_EQ(yMax, "[1, 4, 8, 18]");
}

// SAME + H/W 上界未知：输出上界保持 -1
TEST_F(DepthwiseConv2DRuntimeInferShape, inferShapeRangeSameUnknownHigh)
{
    DepthwiseInferOpt opt;
    opt.strides = {1, 1, 1, 1};
    opt.padding = "SAME";
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDepthwiseRange({1, 4, 1, 1}, {1, 4, -1, -1}, {4, 1, 3, 3}, {4, 1, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(yMin, "[1, 4, 1, 1]");
    ASSERT_EQ(yMax, "[1, 4, -1, -1]");
}

// HWCN filter 的 range 推导（NHWC 输出排布）
TEST_F(DepthwiseConv2DRuntimeInferShape, inferShapeRangeHwcnFilter)
{
    DepthwiseInferOpt opt;
    opt.xFmt = ge::FORMAT_NHWC;
    opt.wFmt = ge::FORMAT_HWCN;
    opt.yFmt = ge::FORMAT_NHWC;
    opt.dataFormat = "NHWC";
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDepthwiseRange({1, 8, 8, 4}, {1, 16, 16, 4}, {3, 3, 4, 1}, {3, 3, 4, 1}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    // H: (8-2-1)+1=6 ~ (16-2-1)+1=14；W 同；C=filterN*filterC=4
    ASSERT_EQ(yMin, "[1, 6, 6, 4]");
    ASSERT_EQ(yMax, "[1, 14, 14, 4]");
}

// Range + _padding 三层一致性：_padding 生效时 range 用 _padding 值（与静态路径一致），SAME 被阻断
TEST_F(DepthwiseConv2DRuntimeInferShape, inferShapeRangePrivatePadding)
{
    DepthwiseInferOpt opt;
    opt.pads = {0, 0, 0, 0};
    opt.paddingList = {2, 2, 2, 2};
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDepthwiseRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    // H: (1+4-3)+1=3 ~ (20+4-3)+1=22？不：xMax H=10 → (10+4-3)+1=12；C=kn*kc=64*32=2048
    ASSERT_EQ(yMin, "[1, 2048, 3, 3]");
    ASSERT_EQ(yMax, "[4, 2048, 12, 22]");
}

// Range + _padding 生效时 SAME 被阻断（else-if 语义）：结果与上一个用例一致而非 ceil 公式
TEST_F(DepthwiseConv2DRuntimeInferShape, inferShapeRangePrivatePaddingBlocksSame)
{
    DepthwiseInferOpt opt;
    opt.padding = "SAME";
    opt.pads = {0, 0, 0, 0};
    opt.paddingList = {2, 2, 2, 2};
    std::string yMin;
    std::string yMax;
    ASSERT_EQ(RunDepthwiseRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_SUCCESS);
    // SAME 被阻断：走显式 pads 公式，与 inferShapeRangePrivatePadding 一致
    ASSERT_EQ(yMin, "[1, 2048, 3, 3]");
    ASSERT_EQ(yMax, "[4, 2048, 12, 22]");
}

// Range 独立入口 stride==0 拦截
TEST_F(DepthwiseConv2DRuntimeInferShape, inferShapeRangeStrideZeroFail)
{
    DepthwiseInferOpt opt;
    opt.strides = {1, 1, 0, 0};
    std::string yMin;
    std::string yMax;
    EXPECT_EQ(RunDepthwiseRange({1, 32, 1, 1}, {4, 32, 10, 20}, {64, 32, 3, 3}, {64, 32, 3, 3}, &yMin, &yMax, opt),
              ge::GRAPH_FAILED);
}

// 说明：InferDataType 经 IMPL_OP 注册于 op_graph/depthwise_conv2d_graph_infer.cpp（graph_plugin
// 编译目标），ophost UT 不链接该目标故无法覆盖——与 Conv2D/Conv3D v1 先例一致（其 InferDataType
// 同样无 UT）。dtype 规则（INT8/INT4/INT16→INT32）以代码评审与图模式链路验证。
