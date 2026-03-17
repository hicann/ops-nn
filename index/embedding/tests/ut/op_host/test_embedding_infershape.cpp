/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_embedding_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "error_util.h"
#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_graph/embedding_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"

class EmbeddingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "EmbeddingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "EmbeddingTest TearDown" << std::endl;
  }
};

TEST_F(EmbeddingTest, embedding_infer_shape_runtime_test_0) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Embedding")->infer_shape;

    gert::StorageShape xShape = {{8, 10}, {8, 10}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{9}, {9}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .InputShapes({&xShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[9, 10]");
}