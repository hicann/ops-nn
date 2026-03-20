/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "error_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

using ge::FORMAT_NCHW;
using ge::FORMAT_NHWC;
using ge::FORMAT_ND;

struct AvgPoolV2ProtoTestParam {
    string case_name;

    std::initializer_list<int64_t> xOriginShape;
    std::initializer_list<int64_t> yOriginShape;

    ge::Format xFormat;
    ge::Format yFormat;

    std::vector<int64_t> ksize;
    std::vector<int64_t> strides;
    std::string paddingMode;
    std::vector<int64_t> pads;
    std::string dataFormat;
    bool globalPooling;
    bool ceilMode;
    bool exclusive;

    bool result;
};

// -----------------AvgPoolV2-----------------
class AvgPoolV2RuntimeInferShape : public testing::TestWithParam<AvgPoolV2ProtoTestParam> {
};

TEST_P(AvgPoolV2RuntimeInferShape, general_cases) {
    AvgPoolV2ProtoTestParam param = GetParam();
    std::cout << "run case " << param.case_name << std::endl;
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPoolV2")->infer_shape;

    gert::StorageShape xShape = {param.xOriginShape, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1, 1})
                        .NodeInputTd(0, ge::DT_FLOAT16, param.xFormat, ge::Format::FORMAT_RESERVED)
                        .NodeOutputTd(0, ge::DT_FLOAT16, param.yFormat, ge::Format::FORMAT_RESERVED)
                        .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.ksize)},
                                    {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                                    {"padding_mode", Ops::NN::AnyValue::CreateFrom<std::string>(param.paddingMode)},
                                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                                    {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.dataFormat)},
                                    {"global_pooling", Ops::NN::AnyValue::CreateFrom<bool>(param.globalPooling)},
                                    {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(param.ceilMode)},
                                    {"exclusive", Ops::NN::AnyValue::CreateFrom<bool>(param.exclusive)}})
                        .InputShapes({&xShape})
                        .OutputShapes({&yShape})
                        .Build();

    if (param.result) {
        ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
        gert::Shape *output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Shape2String(*output), Shape2String(CreateShape(std::vector<int64_t>(param.yOriginShape))));
    } else {
        ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    }
}

static AvgPoolV2ProtoTestParam general_cases_params[] = {
  { "AvgPoolV2_basic1",
    {3, 16, 16, 64}, {3, 15, 15, 64}, FORMAT_NHWC, FORMAT_NHWC,
    {1, 2, 2, 1}, {1, 1, 1, 1}, "VALID", {1, 1, 1, 1}, "NHWC", false, false, true, true
  },
  { "AvgPoolV2_basic2",
    {1, 128, 32, 32}, {1, 128, 32, 32}, FORMAT_NCHW, FORMAT_NCHW,
    {1, 1, 2, 2}, {1, 1, 1, 1}, "SAME", {1, 1, 1, 1}, "NCHW", false, false, true, true
  },
  { "AvgPoolV2_basic3",
    {1, 128, 32, 32}, {1, 128, 32, 32}, FORMAT_NCHW, FORMAT_NCHW,
    {1, 1, 2, 2}, {1, 1, 1, 1}, "SAME", {1, 1, 1, 1}, "NCHW", false, true, true, true
  },
  { "AvgPoolV2_basicGlobal",
    {1, 128, 32, 32}, {1, 128, 1, 1}, FORMAT_NCHW, FORMAT_NCHW,
    {1, 1, 32, 32}, {1, 1, 1, 1}, "VALID", {0, 0, 0, 0}, "NCHW", true, true, true, true
  },
  { "AvgPoolV2_invalidKsize1",
    {1, 128, 32, 32}, {1, 128, 32, 32}, FORMAT_NCHW, FORMAT_NCHW,
    {1, 1, 2, -2}, {1, 1, 1, 1}, "SAME", {1, 1, 1, 1}, "NCHW", false, false, true, false
  },
  { "AvgPoolV2_invalidKsize2",
    {1, 128, 32, 32}, {1, 128, 32, 32}, FORMAT_NHWC, FORMAT_NHWC,
    {1, -1, 2, 2}, {1, 1, 1, 1}, "SAME", {1, 1, 1, 1}, "NHWC", false, false, true, false
  },
  { "AvgPoolV2_invalidKsize3",
    {1, 128, 32, 32}, {1, 128, 32, 32}, FORMAT_NHWC, FORMAT_NHWC,
    {1, 2, 2}, {1, 1, 1, 1}, "SAME", {1, 1, 1, 1}, "NHWC", false, false, true, false
  },
  { "AvgPoolV2_invalidStride1",
    {1, 128, 32, 32}, {1, 128, 32, 32}, FORMAT_NHWC, FORMAT_NHWC,
    {1, 1, 2, 2}, {1, -1, -1, 1}, "SAME", {1, 1, 1, 1}, "NHWC", false, false, true, false
  },
  { "AvgPoolV2_invalidStride2",
    {1, 128, 32, 32}, {1, 128, 32, 32}, FORMAT_NHWC, FORMAT_NHWC,
    {1, 1, 2, 2}, {1, 1}, "SAME", {1, 1, 1, 1}, "NHWC", false, false, true, false
  },
  { "AvgPoolV2_invalidOutputFormat",
    {1, 128, 32, 32}, {1, 128, 32, 32}, FORMAT_NCHW, FORMAT_ND,
    {1, 1, 2, 2}, {1, 1, 1, 1}, "SAME", {1, 1, 1, 1}, "NCHW", false, false, true, false
  },
  { "AvgPoolV2_invalidInputDimSize",
    {1, 128, 32}, {1, 128, 32, 32}, FORMAT_NHWC, FORMAT_NHWC,
    {1, 2, 2, 1}, {1, 1, 1, 1}, "SAME", {1, 1, 1, 1}, "NHWC", false, false, true, false
  },
};

INSTANTIATE_TEST_CASE_P(AvgPoolV2, AvgPoolV2RuntimeInferShape, testing::ValuesIn(general_cases_params));