/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

using ge::FORMAT_NDC1HWC0;
using ge::FORMAT_NDHWC;
using ge::FORMAT_NCDHW;
using ge::FORMAT_DHWCN;
using ge::FORMAT_ND;

struct AvgPool3DProtoTestParam {
    string case_name;

    std::initializer_list<int64_t> x_ori_shape;
    std::initializer_list<int64_t> y_ori_shape;

    ge::Format x_ori_format;
    ge::Format y_ori_format;

    std::vector<int64_t> ksize;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    bool ceil_mode;
    std::string data_format;
    std::string padding;

    bool result;
};

class AvgPool3DRuntimeProtoTest : public testing::TestWithParam<AvgPool3DProtoTestParam> {
};

TEST_P(AvgPool3DRuntimeProtoTest, general_cases) {
    AvgPool3DProtoTestParam param = GetParam();
    std::cout << "run case " << param.case_name << std::endl;
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool3D")->infer_shape;

    gert::StorageShape x_shape = {param.x_ori_shape, {}};
    gert::StorageShape y_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, param.x_ori_format, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, param.y_ori_format, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.ksize)},
                                  {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                                  {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                                  {"ceil_mode", ge::AnyValue::CreateFrom<bool>(param.ceil_mode)},
                                  {"count_include_pad", ge::AnyValue::CreateFrom<bool>(true)},
                                  {"divisor_override", ge::AnyValue::CreateFrom<int64_t>(0)},
                                  {"data_format", ge::AnyValue::CreateFrom<std::string>(param.data_format)},
                                  {"padding", ge::AnyValue::CreateFrom<std::string>(param.padding)}})
                      .InputShapes({&x_shape})
                      .OutputShapes({&y_shape})
                      .Build();

    if (param.result) {
        ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
        gert::Shape *output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(gert::Shape(param.y_ori_shape)));
    } else {
        ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    }
}

static AvgPool3DProtoTestParam general_cases_params[] = {
    { "avgpool3d_NDHWC_NDHWC_false_SAME",
      {40, 48, 14, 14, 3}, {40, 48, 7, 5, 3}, FORMAT_NDHWC, FORMAT_NDHWC,
      {3, 5, 3, 3, 3}, {1, 1, 2, 3, 1}, {-1, -1, -1, -1, -1, -1}, false, "NDHWC", "SAME", true
    },

    { "avgpool3d_NCDHW_NCDHW_false_VALID",
      {8, 28, 8, 60, 88}, {8, 28, 4, 30, 44}, FORMAT_NCDHW, FORMAT_NCDHW,
      {32, 28, 1, 2, 2}, {1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, false, "NCDHW", "VALID", true
    },

    { "avgpool3d_ND_NCDHW_false_VALID",
      {8, 28, 8, 60, 88}, {8, 28, 4, 30, 44}, FORMAT_ND, FORMAT_NCDHW,
      {32, 28, 1, 2, 2}, {1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, false, "NCDHW", "VALID", false
    },

    { "avgpool3d_NCDHW_ND_false_VALID",
      {8, 28, 8, 60, 88}, {8, 28, 4, 30, 44}, FORMAT_NCDHW, FORMAT_ND,
      {32, 28, 1, 2, 2}, {1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, false, "NCDHW", "VALID", false
    },

    { "avgpool3d_NCDHW_ND_false_VALID_PadsLen1",
      {8, 28, 8, 60, 88}, {8, 28, 4, 30, 44}, FORMAT_NCDHW, FORMAT_ND,
      {32, 28, 1, 2, 2}, {1, 1, 2, 2, 2}, {0}, false, "NCDHW", "VALID", false
    },

    { "avgpool3d_NCDHW_ND_false_VALID_PadsLen3",
      {8, 28, 8, 60, 88}, {8, 28, 4, 30, 44}, FORMAT_NCDHW, FORMAT_ND,
      {32, 28, 1, 2, 2}, {1, 1, 2, 2, 2}, {0, 0, 0}, false, "NCDHW", "VALID", false
    },

};

INSTANTIATE_TEST_CASE_P(AvgPool3D, AvgPool3DRuntimeProtoTest, testing::ValuesIn(general_cases_params));
