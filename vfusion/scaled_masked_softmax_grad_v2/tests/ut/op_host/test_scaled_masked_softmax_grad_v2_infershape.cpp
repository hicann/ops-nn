/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scaled_masked_softmax_grad_v2_infersahpe.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

class ScaledMaskedSoftmaxGradV2Proto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScaledMaskedSoftmaxGradV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ScaledMaskedSoftmaxGradV2 Proto Test TearDown" << std::endl;
    }
};

TEST_F(ScaledMaskedSoftmaxGradV2Proto, scaled_masked_softmax_grad_v2_infershape_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScaledMaskedSoftmaxGradV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape yGradShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    gert::StorageShape yShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    gert::StorageShape maskShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    gert::StorageShape xGradShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&yGradShape, &yShape, &maskShape})
                      .OutputShapes({&xGradShape})
                      .NodeAttrs(
                          {{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 32, 32, 128]");
}

TEST_F(ScaledMaskedSoftmaxGradV2Proto, scaled_masked_softmax_grad_v2_inferdtype_test)
{
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScaledMaskedSoftmaxGradV2")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType inputRef = ge::DT_FLOAT16;
    ge::DataType outputRef = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputRef, &inputRef})
                              .OutputDataTypes({&outputRef})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), outputRef);
}