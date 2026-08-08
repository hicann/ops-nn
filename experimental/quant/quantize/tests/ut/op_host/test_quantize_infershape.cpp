/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_quantize_infershape.cpp
 * \brief Quantize infershape UT (output shape == x shape; output dtype from `dtype` attr).
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class QuantizeInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QuantizeInfershapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "QuantizeInfershapeTest TearDown" << std::endl; }
};

TEST_F(QuantizeInfershapeTest, quantize_infershape_per_channel_4d)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("Quantize");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{3, 4, 5, 6}, {3, 4, 5, 6}};
    gert::StorageShape scalesShape = {{4}, {4}};
    gert::StorageShape zeroPointsShape = {{4}, {4}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &scalesShape, &zeroPointsShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(outShape, nullptr);
    ASSERT_EQ(outShape->GetDimNum(), 4U);
    EXPECT_EQ(outShape->GetDim(0), 3);
    EXPECT_EQ(outShape->GetDim(1), 4);
    EXPECT_EQ(outShape->GetDim(2), 5);
    EXPECT_EQ(outShape->GetDim(3), 6);
}

TEST_F(QuantizeInfershapeTest, quantize_infershape_per_tensor_2d_no_zp)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("Quantize");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{8, 16}, {8, 16}};
    gert::StorageShape scalesShape = {{1}, {1}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &scalesShape, nullptr})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(outShape, nullptr);
    ASSERT_EQ(outShape->GetDimNum(), 2U);
    EXPECT_EQ(outShape->GetDim(0), 8);
    EXPECT_EQ(outShape->GetDim(1), 16);
}
