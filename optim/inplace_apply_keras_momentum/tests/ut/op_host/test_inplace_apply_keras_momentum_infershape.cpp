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
 * \file test_inplace_apply_keras_momentum_infershape.cpp
 * \brief InplaceApplyKerasMomentum InferShape host UT.
 *
 * The test calls the real registered InferShape callback through
 * InferShapeContextFaker; it does not copy the implementation into the test.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "util/shape_util.h"

using namespace ge;
using namespace std;

namespace {

constexpr const char* kOpType = "InplaceApplyKerasMomentum";

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableStorageShape().AppendDim(dim);
        shape.MutableOriginShape().AppendDim(dim);
    }
    return shape;
}

class InplaceApplyKerasMomentumInferShapeTest : public testing::Test {};

void RunInferShapeCase(const std::vector<int64_t>& varDims, const std::vector<int64_t>& accumDims, ge::DataType dtype,
                       const std::vector<int64_t>& expectedDims)
{
    auto var = MakeShape(varDims);
    auto accum = MakeShape(accumDims);
    auto lr = MakeShape({});
    auto grad = MakeShape(varDims);
    auto momentum = MakeShape({});
    auto varOut = MakeShape({});
    auto accumOut = MakeShape({});

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&var, &accum, &lr, &grad, &momentum})
                      .OutputShapes({&varOut, &accumOut})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    auto expected = MakeShape(expectedDims);
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), Ops::Base::ToString(expected.GetStorageShape()));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(1)), Ops::Base::ToString(expected.GetStorageShape()));
}

} // namespace

TEST_F(InplaceApplyKerasMomentumInferShapeTest, infer_shape_1d) { RunInferShapeCase({64}, {64}, ge::DT_FLOAT, {64}); }

TEST_F(InplaceApplyKerasMomentumInferShapeTest, infer_shape_4d)
{
    RunInferShapeCase({1000, 10, 10, 10}, {1000, 10, 10, 10}, ge::DT_FLOAT, {1000, 10, 10, 10});
}

TEST_F(InplaceApplyKerasMomentumInferShapeTest, infer_shape_fp16)
{
    RunInferShapeCase({8, 16}, {8, 16}, ge::DT_FLOAT16, {8, 16});
}

TEST_F(InplaceApplyKerasMomentumInferShapeTest, infer_shape_bf16)
{
    RunInferShapeCase({8, 16}, {8, 16}, ge::DT_BF16, {8, 16});
}

TEST_F(InplaceApplyKerasMomentumInferShapeTest, infer_shape_rank_zero) { RunInferShapeCase({}, {}, ge::DT_FLOAT, {}); }

TEST_F(InplaceApplyKerasMomentumInferShapeTest, infer_shape_empty_tensor)
{
    RunInferShapeCase({0, 1}, {0, 1}, ge::DT_FLOAT, {0, 1});
}

TEST_F(InplaceApplyKerasMomentumInferShapeTest, infer_shape_dynamic_dimension)
{
    RunInferShapeCase({-1, 4}, {-1, 4}, ge::DT_FLOAT, {-1, 4});
}

TEST_F(InplaceApplyKerasMomentumInferShapeTest, infer_shape_dynamic_rank)
{
    RunInferShapeCase({-2}, {-2}, ge::DT_FLOAT, {-2});
}

TEST_F(InplaceApplyKerasMomentumInferShapeTest, infer_shape_rank_eight)
{
    RunInferShapeCase({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}, ge::DT_FLOAT, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST_F(InplaceApplyKerasMomentumInferShapeTest, both_outputs_follow_var_shape)
{
    // accum uses a different shape on purpose: both outputs must follow input var.
    RunInferShapeCase({2, 3}, {2, 4}, ge::DT_FLOAT, {2, 3});
}
