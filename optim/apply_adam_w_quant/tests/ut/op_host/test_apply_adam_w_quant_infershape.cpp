/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"

class ApplyAdamWQuantTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyAdamWQuantTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyAdamWQuantTest TearDown" << std::endl;
    }
};

TEST_F(ApplyAdamWQuantTest, apply_adam_w_quant_infershape_verify_test_01)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWQuant"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWQuant")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape gradShape = {{96, 256}, {96, 256}};
    gert::StorageShape mShape = {{96, 256}, {96, 256}};
    gert::StorageShape vShape = {{96, 256}, {96, 256}};
    gert::StorageShape qmapMShape = {{256}, {256}};
    gert::StorageShape qmapVShape = {{256}, {256}};
    gert::StorageShape absmaxMShape = {{96}, {96}};
    gert::StorageShape absmaxVShape = {{96}, {96}};
    gert::StorageShape stepShape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(9, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&varShape, &gradShape, &mShape, &vShape, &qmapMShape, &qmapVShape, &absmaxMShape,
                           &absmaxMShape, &stepShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &absmaxMShape, &absmaxMShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}