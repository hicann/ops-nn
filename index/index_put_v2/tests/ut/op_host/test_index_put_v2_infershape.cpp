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
 * \file test_index_put_v2_infershape.cpp
 * \brief
 */

#include "ut_op_util.h"
#include "infershape_test_util.h"
#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/index_put_v2_proto.h"
#include "ut_op_common.h"

class IndexPutV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IndexPutV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IndexPutV2 TearDown" << std::endl;
    }
};

TEST_F(IndexPutV2, IndexPutV2_infershape_case_1)
{
    ge::op::IndexPutV2 op;
    op.UpdateInputDesc("x", create_desc({1000, 1000}, ge::DT_INT64));
    op.UpdateInputDesc("value", create_desc({1, 1}, ge::DT_INT64));
    op.UpdateInputDesc("indexed_sizes", create_desc({1, 1}, ge::DT_INT64));
    op.UpdateInputDesc("indexed_strides", create_desc({1, 1}, ge::DT_INT64));
    op.UpdateInputDesc("indices", create_desc({1, 1}, ge::DT_INT64));
    op.SetAttr("accumulate", false);
    Runtime2TestParam param{{"accumulate"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {1000, 1000};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(IndexPutV2, IndexPutV2_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("IndexPutV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexPutV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_INT64;
        ge::DataType input_vaule_ref = ge::DT_INT64;
        ge::DataType input_indexed_sizes_ref = ge::DT_INT64;
        ge::DataType input_indexed_strides_ref = ge::DT_INT64;
        ge::DataType input_indices_ref = ge::DT_INT64;
        ge::DataType output_y_ref = ge::DT_INT64;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(5)
                                  .NodeIoNum(5, 1)
                                  .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"accumulate", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                                  .InputDataTypes({&input_x_ref, &input_vaule_ref, &input_indexed_sizes_ref, &input_indexed_strides_ref, &input_indices_ref})
                                  .OutputDataTypes({&output_y_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_vaule_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_indexed_sizes_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_indexed_strides_ref);
        EXPECT_EQ(context->GetInputDataType(4), input_indices_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
    }
}