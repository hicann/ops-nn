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
#include "kernel_run_context_facker.h"
#include "log/log.h"

using ge::FORMAT_NCDHW;
using ge::FORMAT_ND;


struct AdaptiveMaxPool3dProtoTestParam {
    string caseName;

    std::initializer_list<int64_t> xOriShape;
    std::initializer_list<int64_t> yOriShape;

    ge::Format xOriFormat;
    ge::Format yOriFormat;

    std::vector<int64_t> outputSize;
    int64_t indicesDtype;
    bool result;
};

class AdaptiveMaxPool3dRuntimeProtoTest : public testing::TestWithParam<AdaptiveMaxPool3dProtoTestParam> {
};

TEST_P(AdaptiveMaxPool3dRuntimeProtoTest, general_cases) {
    AdaptiveMaxPool3dProtoTestParam param = GetParam();
    std::cout << "run case " << param.caseName << std::endl;
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool3d")->infer_shape;


    gert::StorageShape xShape = {param.xOriShape, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};


    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT16, param.xOriFormat, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, param.yOriFormat, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, param.yOriFormat, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.outputSize)},
                                  {"indices_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(param.indicesDtype)}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &indicesShape})
                      .Build();

    if (param.result) {
        ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
        gert::Shape *output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        gert::Shape *indices = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
        ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(gert::Shape(param.yOriShape)));
        ASSERT_EQ(Ops::Base::ToString(*indices), Ops::Base::ToString(gert::Shape(param.yOriShape)));
    } else {
        ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    }
}

static AdaptiveMaxPool3dProtoTestParam general_cases_params[] = {
    { "AdaptiveMaxPool3dInfershapeSize3",
      {3, 2, 14, 14, 3}, {3, 2, 7, 5, 3}, FORMAT_NCDHW, FORMAT_NCDHW,
      {7, 5, 3}, 3, true
    },
    { "AdaptiveMaxPool3dInfershapeSize2",
      {3, 2, 14, 14, 3}, {3, 2, 7, 5, 3}, FORMAT_NCDHW, FORMAT_NCDHW,
      {7, 5}, 3, false
    },
    { "AdaptiveMaxPool3dInfershapeSize1",
      {3, 2, 14, 14, 3}, {3, 2, 3, 3, 3}, FORMAT_NCDHW, FORMAT_NCDHW,
      {3}, 3, true
    },
    { "AdaptiveMaxPool3dInfershapeSize0",
      {3, 2, 14, 14, 3}, {3, 2, 14, 14, 3}, FORMAT_NCDHW, FORMAT_NCDHW,
      {}, 3, true
    },
    { "AdaptiveMaxPool3dInfershapeSizeUnRank",
      {-2}, {-2}, FORMAT_NCDHW, FORMAT_NCDHW,
      {}, 3, true
    },
    { "AdaptiveMaxPool3dInfershapeSizeUnShape",
      {-1}, {-1}, FORMAT_NCDHW, FORMAT_NCDHW,
      {}, 3, true
    },
};

INSTANTIATE_TEST_CASE_P(AdaptiveMaxPool3d, AdaptiveMaxPool3dRuntimeProtoTest, testing::ValuesIn(general_cases_params));