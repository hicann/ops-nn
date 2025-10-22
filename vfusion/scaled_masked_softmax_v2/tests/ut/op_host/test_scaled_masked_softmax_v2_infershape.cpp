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
 * \file test_scaled_masked_softmax_v2_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

class ScaledMaskedSoftmaxV2Proto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScaledMaskedSoftmaxV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ScaledMaskedSoftmaxV2 Proto Test TearDown" << std::endl;
    }
};

TEST_F(ScaledMaskedSoftmaxV2Proto, inferShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScaledMaskedSoftmaxV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    gert::StorageShape maskShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .NodeAttrs(
                          {{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 32, 32, 128]");
}

TEST_F(ScaledMaskedSoftmaxV2Proto, inferDataType)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScaledMaskedSoftmaxV2")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);
    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType maskDtype = ge::DT_BOOL;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputDataTypes({&xDtype, &maskDtype})
                      .OutputDataTypes({&yDtype})
                      .NodeAttrs(
                          {{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferDataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}
