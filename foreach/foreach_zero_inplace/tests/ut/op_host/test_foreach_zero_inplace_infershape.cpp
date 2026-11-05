/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/kernel_context.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/foreach_zero_inplace_proto.h"

class ForeachZeroInplaceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachZeroInplace SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ForeachZeroInplace TearDown" << std::endl; }
};

TEST_F(ForeachZeroInplaceTest, infer_callbacks_registered)
{
    const auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachZeroInplace");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_shape, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}

TEST_F(ForeachZeroInplaceTest, infer_shape_mirrors_dynamic_inputs)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachZeroInplace")->infer_shape;
    ASSERT_NE(inferShape, nullptr);

    gert::StorageShape xShape0 = {{1}, {}};
    gert::StorageShape xShape1 = {{2, 3}, {}};
    gert::StorageShape xShape2 = {{}, {}};
    gert::StorageShape yShape0 = {{}, {}};
    gert::StorageShape yShape1 = {{}, {}};
    gert::StorageShape yShape2 = {{}, {}};
    std::vector<void*> inputShapes = {&xShape0, &xShape1, &xShape2};
    std::vector<void*> outputShapes = {&yShape0, &yShape1, &yShape2};

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({3}, {3})
                      .InputShapes(inputShapes)
                      .OutputShapes(outputShapes)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShape(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[1]");
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(1)), "[2, 3]");
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(2)), "[]");
}

TEST_F(ForeachZeroInplaceTest, infer_shape_rejects_rank_above_limit)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachZeroInplace")->infer_shape;
    ASSERT_NE(inferShape, nullptr);

    gert::StorageShape xShape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}};
    gert::StorageShape yShape = {{}, {}};
    std::vector<void*> inputShapes = {&xShape};
    std::vector<void*> outputShapes = {&yShape};
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1}, {1})
                      .InputShapes(inputShapes)
                      .OutputShapes(outputShapes)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShape(context), ge::GRAPH_FAILED);
}

TEST_F(ForeachZeroInplaceTest, infer_shape_rejects_output_count_mismatch)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachZeroInplace")->infer_shape;
    ASSERT_NE(inferShape, nullptr);

    gert::StorageShape xShape0 = {{1}, {}};
    gert::StorageShape xShape1 = {{2}, {}};
    gert::StorageShape yShape = {{}, {}};
    std::vector<void*> inputShapes = {&xShape0, &xShape1};
    std::vector<void*> outputShapes = {&yShape};
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({2}, {1})
                      .InputShapes(inputShapes)
                      .OutputShapes(outputShapes)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShape(context), ge::GRAPH_FAILED);
}

TEST_F(ForeachZeroInplaceTest, infer_dtype_mirrors_dynamic_inputs)
{
    auto inferDtype = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachZeroInplace")->infer_datatype;
    ASSERT_NE(inferDtype, nullptr);

    ge::DataType xDtype0 = ge::DT_FLOAT16;
    ge::DataType xDtype1 = ge::DT_FLOAT16;
    ge::DataType xDtype2 = ge::DT_FLOAT16;
    std::vector<void*> inputDtypes = {&xDtype0, &xDtype1, &xDtype2};
    std::vector<void*> outputDtypes(3);
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInstanceNum({3}, {3})
                      .InputDataTypes(inputDtypes)
                      .OutputDataTypes(outputDtypes)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT16);
}

TEST_F(ForeachZeroInplaceTest, infer_dtype_rejects_mixed_inputs)
{
    auto inferDtype = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachZeroInplace")->infer_datatype;
    ASSERT_NE(inferDtype, nullptr);

    ge::DataType xDtype0 = ge::DT_FLOAT16;
    ge::DataType xDtype1 = ge::DT_FLOAT;
    std::vector<void*> inputDtypes = {&xDtype0, &xDtype1};
    std::vector<void*> outputDtypes(2);
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInstanceNum({2}, {2})
                      .InputDataTypes(inputDtypes)
                      .OutputDataTypes(outputDtypes)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDtype(context), ge::GRAPH_FAILED);
}

TEST_F(ForeachZeroInplaceTest, infer_dtype_rejects_unsupported_inputs)
{
    auto inferDtype = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachZeroInplace")->infer_datatype;
    ASSERT_NE(inferDtype, nullptr);

    for (auto dtype : {ge::DT_INT8, ge::DT_DOUBLE}) {
        std::vector<void*> inputDtypes = {&dtype};
        std::vector<void*> outputDtypes(1);
        auto holder = gert::InferDataTypeContextFaker()
                          .IrInstanceNum({1}, {1})
                          .InputDataTypes(inputDtypes)
                          .OutputDataTypes(outputDtypes)
                          .Build();

        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(inferDtype(context), ge::GRAPH_FAILED);
    }
}
