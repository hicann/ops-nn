/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"

#include "../../../op_graph/ascend_dequant_proto.h"

using namespace ge;

namespace {

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (const int64_t d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

const auto* GetImpl() { return gert::OpImplRegistry::GetInstance().GetOpImpl("AscendDequant"); }

class AscendDequantInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AscendDequantInfershapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AscendDequantInfershapeTest TearDown" << std::endl; }
};

struct InferCase {
    std::vector<int64_t> xShape;
    std::vector<int64_t> scaleShape;
    ge::DataType xDtype = ge::DT_INT32;
    ge::DataType scaleDtype = ge::DT_UINT64;
    ge::DataType yDtype = ge::DT_FLOAT; // declared output dtype; attr dtype=0 default
    int64_t dtypeAttr = 0;
    bool sqrtMode = false;
    bool reluFlag = false;
    ge::Format xFormat = ge::FORMAT_ND;
    ge::Format yFormat = ge::FORMAT_ND;
    ge::Format scaleFormat = ge::FORMAT_ND;
};

ge::graphStatus RunInferShape(const InferCase& c, std::vector<int64_t>& outShape)
{
    const auto* impl = GetImpl();
    if (impl == nullptr || impl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::StorageShape xShape = MakeStorageShape(c.xShape);
    gert::StorageShape sShape = MakeStorageShape(c.scaleShape);
    gert::StorageShape yShape = MakeStorageShape({});
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, c.xDtype, c.xFormat, c.xFormat)
                      .NodeInputTd(1, c.scaleDtype, c.scaleFormat, c.scaleFormat)
                      .NodeOutputTd(0, c.yDtype, c.yFormat, c.yFormat)
                      .NodeAttrs({{"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(c.sqrtMode)},
                                  {"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(c.reluFlag)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(c.dtypeAttr)}})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .Build();
    const ge::graphStatus ret = impl->infer_shape(holder.GetContext<gert::InferShapeContext>());
    outShape.clear();
    if (ret == ge::GRAPH_SUCCESS) {
        const gert::Shape* y = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        if (y != nullptr) {
            for (size_t i = 0; i < y->GetDimNum(); ++i) {
                outShape.push_back(y->GetDim(i));
            }
        }
    }
    return ret;
}

ge::graphStatus RunInferDataType(int64_t dtypeAttr, ge::DataType& outDtype)
{
    const auto* impl = GetImpl();
    if (impl == nullptr || impl->infer_datatype == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType xDtype = ge::DT_INT32;
    ge::DataType scaleDtype = ge::DT_UINT64;
    ge::DataType yDtype = ge::DT_FLOAT16;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(2)
                      .NodeIoNum(2, 1)
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&xDtype, &scaleDtype})
                      .OutputDataTypes({&yDtype})
                      .NodeAttrs({{"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtypeAttr)}})
                      .Build();
    auto* ctx = holder.GetContext<gert::InferDataTypeContext>();
    const ge::graphStatus ret = impl->infer_datatype(ctx);
    if (ret == ge::GRAPH_SUCCESS) {
        outDtype = ctx->GetOutputDataType(0);
    }
    return ret;
}

} // namespace

// ---------------------------------------------------------------------------
// Graph prototype registration: the attr default is DT_FLOAT (dtype = 0).
// ---------------------------------------------------------------------------
TEST(AscendDequantProtoTest, RegistersInputsOutputsAndDefaultAttrs)
{
    ge::op::AscendDequant op;

    EXPECT_EQ(op.GetInputsSize(), 2U);
    EXPECT_EQ(op.GetOutputsSize(), 1U);
    EXPECT_FALSE(op.get_attr_sqrt_mode());
    EXPECT_FALSE(op.get_attr_relu_flag());
    EXPECT_EQ(op.get_attr_dtype(), 0); // DT_FLOAT, the prototype default
}

TEST(AscendDequantProtoTest, UpdatesAttrValues)
{
    ge::op::AscendDequant op;

    op.set_attr_sqrt_mode(true);
    op.set_attr_relu_flag(true);
    op.set_attr_dtype(1);

    EXPECT_TRUE(op.get_attr_sqrt_mode());
    EXPECT_TRUE(op.get_attr_relu_flag());
    EXPECT_EQ(op.get_attr_dtype(), 1);
}

TEST(AscendDequantProtoTest, UpdatesInputAndOutputDesc)
{
    ge::op::AscendDequant op;

    ge::TensorDesc input_desc(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_INT32);
    EXPECT_EQ(op.update_input_desc_x(input_desc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_input_desc_x().GetDataType(), ge::DT_INT32);

    ge::TensorDesc scale_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_UINT64);
    EXPECT_EQ(op.update_input_desc_deq_scale(scale_desc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_input_desc_deq_scale().GetDataType(), ge::DT_UINT64);

    ge::TensorDesc output_desc(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_FLOAT16);
    EXPECT_EQ(op.update_output_desc_y(output_desc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_output_desc_y().GetDataType(), ge::DT_FLOAT16);
}

// ---------------------------------------------------------------------------
// Real InferShape / InferDataType.
// ---------------------------------------------------------------------------
TEST_F(AscendDequantInfershapeTest, InfershapeRank2PerChannel)
{
    InferCase c;
    c.xShape = {2, 3};
    c.scaleShape = {3};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 3}));

    ge::DataType dt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(1, dt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dt, ge::DT_FLOAT16);
}

TEST_F(AscendDequantInfershapeTest, InfershapeRank1PerTensor)
{
    InferCase c;
    c.xShape = {16};
    c.scaleShape = {1};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{16}));
}

TEST_F(AscendDequantInfershapeTest, InfershapeRank4AlignedScale)
{
    InferCase c;
    c.xShape = {2, 32, 4, 4};
    c.scaleShape = {1, 32, 1, 1};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 32, 4, 4}));
}

// y.shape is a plain copy of x.shape: the deq_scale length never affects it,
// whichever axis the 1-D scale would bind to.
TEST_F(AscendDequantInfershapeTest, InfershapeOneDimScaleBindsLastAxis)
{
    InferCase c;
    c.xShape = {2, 32, 4, 4};
    c.scaleShape = {4};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 32, 4, 4}));

    c.scaleShape = {32}; // 非末轴长度：图模式不再拦截，shape 照旧透传
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 32, 4, 4}));
}

// Arbitrary broadcast axes: x={2,3,4} with scale={2,1,4} is numpy-broadcastable.
TEST_F(AscendDequantInfershapeTest, InfershapeAcceptsBroadcastMiddleAxis)
{
    InferCase c;
    c.xShape = {2, 3, 4};
    c.scaleShape = {2, 1, 4};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 3, 4}));
}

// Per-tensor scale (1,) is broadcastable to any x.
TEST_F(AscendDequantInfershapeTest, InfershapeAcceptsPerTensorScale)
{
    InferCase c;
    c.xShape = {2, 3, 4};
    c.scaleShape = {1};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 3, 4}));
}

// No format validation: a registered non-ND format must infer exactly like the
// same shapes declared as ND, because broadcasting is positional.
TEST_F(AscendDequantInfershapeTest, InfershapeAcceptsRegisteredNonNdFormats)
{
    for (const ge::Format format : {ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC}) {
        InferCase c;
        c.xShape = {2, 32, 4, 4};
        c.scaleShape = {4};
        c.xFormat = format;
        c.yFormat = format;
        c.dtypeAttr = 1;
        c.yDtype = ge::DT_FLOAT16;
        std::vector<int64_t> out;
        ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
        EXPECT_EQ(out, (std::vector<int64_t>{2, 32, 4, 4}));
    }
}

TEST_F(AscendDequantInfershapeTest, InfershapeAcceptsNc1hwc0Format)
{
    InferCase c;
    c.xShape = {2, 32, 4, 4};
    c.scaleShape = {4};
    c.xFormat = ge::FORMAT_NC1HWC0;
    c.yFormat = ge::FORMAT_NC1HWC0;
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 32, 4, 4}));

    InferCase nd = c; // identical result when the same shapes are declared as ND
    nd.xFormat = ge::FORMAT_ND;
    nd.yFormat = ge::FORMAT_ND;
    std::vector<int64_t> ndOut;
    ASSERT_EQ(RunInferShape(nd, ndOut), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, ndOut);
}

// The deq_scale format is not validated either.
TEST_F(AscendDequantInfershapeTest, InfershapeAcceptsNonNdScaleFormat)
{
    InferCase c;
    c.xShape = {2, 4, 4, 32};
    c.scaleShape = {32};
    c.scaleFormat = ge::FORMAT_NHWC;
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 4, 4, 32}));

    InferCase nd = c;
    nd.scaleFormat = ge::FORMAT_ND;
    std::vector<int64_t> ndOut;
    ASSERT_EQ(RunInferShape(nd, ndOut), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, ndOut);
}

// dtype=0 (proto default) -> float32 output with the uint64 deq_scale.
TEST_F(AscendDequantInfershapeTest, InfershapeFloat32AttrUint64Scale)
{
    InferCase c;
    c.xShape = {4, 8};
    c.scaleShape = {8};
    c.scaleDtype = ge::DT_UINT64;
    c.dtypeAttr = 0;
    c.yDtype = ge::DT_FLOAT;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{4, 8}));

    ge::DataType dt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(0, dt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dt, ge::DT_FLOAT);
}

// dtype=1 -> float16 output with the uint64 deq_scale.
TEST_F(AscendDequantInfershapeTest, InfershapeFloat16AttrUint64Scale)
{
    InferCase c;
    c.xShape = {4, 8};
    c.scaleShape = {8};
    c.scaleDtype = ge::DT_UINT64;
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{4, 8}));
}

// InferShape does not validate dtypes: a non-uint64 scale still passes the
// shape through, matching the legacy behaviour.
TEST_F(AscendDequantInfershapeTest, InfershapePassThroughIgnoresScaleDtype)
{
    InferCase c;
    c.xShape = {4, 8};
    c.scaleShape = {8};
    c.scaleDtype = ge::DT_FLOAT16;
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{4, 8}));

    InferCase c2 = c;
    c2.dtypeAttr = 0;
    c2.yDtype = ge::DT_FLOAT;
    ASSERT_EQ(RunInferShape(c2, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{4, 8}));
}

TEST_F(AscendDequantInfershapeTest, InferdatatypeDtype0And1)
{
    ge::DataType dt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(0, dt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dt, ge::DT_FLOAT);
    dt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(1, dt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dt, ge::DT_FLOAT16);
}

TEST_F(AscendDequantInfershapeTest, InfershapePassThroughAllowsOutOfRangeDtypeAttr)
{
    InferCase c;
    c.xShape = {2, 3};
    c.scaleShape = {3};
    c.dtypeAttr = 2;
    c.yDtype = ge::DT_FLOAT;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 3}));

    // dtype 推导仍校验属性值域
    ge::DataType dt = ge::DT_UNDEFINED;
    EXPECT_EQ(RunInferDataType(2, dt), ge::GRAPH_FAILED);
}

TEST_F(AscendDequantInfershapeTest, InfershapePassThroughIgnoresDeclaredYDtype)
{
    // dtype=0 asks for float32 while the graph declared a float16 y.
    InferCase c;
    c.xShape = {2, 3};
    c.scaleShape = {3};
    c.dtypeAttr = 0;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 3}));

    // dtype=1 asks for float16 while the graph declared a float32 y.
    InferCase c2;
    c2.xShape = {2, 3};
    c2.scaleShape = {3};
    c2.dtypeAttr = 1;
    c2.yDtype = ge::DT_FLOAT;
    ASSERT_EQ(RunInferShape(c2, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 3}));
}

// Dynamic x shape with a 1-D scale: the length is validated by tiling on the
// real shape, so the broadcast check must accept rank-1 scales even when the
// aligned x dimension is unknown.
TEST_F(AscendDequantInfershapeTest, InfershapeDynamicXWithRank1Scale)
{
    InferCase c;
    c.xShape = {2, -1};
    c.scaleShape = {8};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, -1}));
}

TEST_F(AscendDequantInfershapeTest, InfershapePassThroughSkipsBroadcastCheck)
{
    InferCase c;
    c.xShape = {2, 3};
    c.scaleShape = {4};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 3}));
}

TEST_F(AscendDequantInfershapeTest, InfershapePassThroughIgnoresInputDtype)
{
    InferCase c;
    c.xShape = {2, 3};
    c.scaleShape = {3};
    c.xDtype = ge::DT_FLOAT;
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 3}));
}

TEST_F(AscendDequantInfershapeTest, InfershapePassThroughIgnoresNonUint64ScaleDtype)
{
    InferCase c;
    c.xShape = {2, 3};
    c.scaleShape = {3};
    c.scaleDtype = ge::DT_FLOAT;
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{2, 3}));
}

// An empty tensor (any dim == 0) is legal: tiling short-circuits with
// usedCoreNum = 0 and the kernel returns immediately.
TEST_F(AscendDequantInfershapeTest, InfershapeAcceptsEmptyTensor)
{
    InferCase c;
    c.xShape = {0, 3};
    c.scaleShape = {3};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{0, 3}));
}

TEST_F(AscendDequantInfershapeTest, InfershapePassThroughAllowsRank9)
{
    InferCase c;
    c.xShape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    c.scaleShape = {1};
    c.dtypeAttr = 1;
    c.yDtype = ge::DT_FLOAT16;
    std::vector<int64_t> out;
    ASSERT_EQ(RunInferShape(c, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, (std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1}));
}
