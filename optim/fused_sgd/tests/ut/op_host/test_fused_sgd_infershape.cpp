/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <array>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/storage_format.h"
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"

#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/kernel_context.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"

using namespace ge;
using namespace ut_util;
using namespace std;

class FusedSgdInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedSgdInferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "FusedSgdInferShape TearDown" << std::endl; }
};

namespace {
std::vector<int64_t> ToVectorForFused(const gert::Shape& shape)
{
    size_t shape_size = shape.GetDimNum();
    std::vector<int64_t> shape_vec(shape_size, 0);

    for (size_t i = 0; i < shape_size; i++) {
        shape_vec[i] = shape.GetDim(i);
    }
    return shape_vec;
}
} // namespace

TEST_F(FusedSgdInferShape, test_fused_sgd_infershape_same_shape)
{
    gert::StorageShape paramsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape gradsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape momentumShape = {{3, 4, 5}, {3, 6}};
    std::string opType("FusedSgd");
    auto infershape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType)->infer_shape;

    gert::StorageShape paramsRefShape;
    gert::StorageShape gradsRefShape;
    gert::StorageShape momentumRefShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .OutputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infershape_func(context), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedOutputShape1 = {3, 4, 5};
    for (int i = 0; i < 3; i++) {
        auto tmpOutShape1 = context->GetOutputShape(i);
        EXPECT_EQ(ToVectorForFused(*tmpOutShape1), expectedOutputShape1);
    }
}