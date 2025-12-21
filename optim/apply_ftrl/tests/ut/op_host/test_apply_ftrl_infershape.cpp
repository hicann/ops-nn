/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/apply_ftrl_proto.h"

class ApplyFtrl : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyFtrl SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyFtrl TearDown" << std::endl;
    }
};

TEST_F(ApplyFtrl, ApplyFtrl_infershape_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape accumShape = {{96, 256}, {96, 256}};
    gert::StorageShape linearShape = {{96, 256}, {96, 256}};
    gert::StorageShape gradShape = {{96, 256}, {96, 256}};
    gert::StorageShape lrShape = {{1}, {1}};
    gert::StorageShape l1Shape = {{1}, {1}};
    gert::StorageShape l2Shape = {{1}, {1}};
    gert::StorageShape lrPowerShape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(8, 3)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes({&varShape, &accumShape, &linearShape, &gradShape,
                        &lrShape, &l1Shape, &l2Shape, &lrPowerShape})
        .OutputShapes({&varShape, &accumShape, &linearShape})
        .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}