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
 * \file test_grouped_dynamic_mx_quant_infershape.cpp
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
#include "../../../op_graph/grouped_dynamic_mx_quant_proto.h"
#include "ut_op_common.h"

class GroupedDynamicMxQuant : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedDynamicMxQuant SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedDynamicMxQuant TearDown" << std::endl;
    }
};

TEST_F(GroupedDynamicMxQuant, GroupedDynamicMxQuant_infershape_case_1)
{
    constexpr int32_t BLOCK_SIZE = 32;
    ge::op::GroupedDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({32, 128}, ge::DT_BF16));
    op.UpdateInputDesc("group_index", create_desc({1}, ge::DT_INT32));
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", (int64_t)ge::DT_FLOAT8_E4M3FN);
    op.SetAttr("blocksize", BLOCK_SIZE);
    Runtime2TestParam param{{"round_mode", "dst_type", "blocksize"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {32, 128};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(GroupedDynamicMxQuant, GroupedDynamicMxQuant_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicMxQuant")->infer_datatype;

    constexpr int32_t BLOCK_SIZE = 32;
    if (data_type_func != nullptr) {
        ge::DataType inDtype = ge::DT_BF16;
        ge::DataType in2Dtype = ge::DT_INT32;
        ge::DataType outDtype = ge::DT_FLOAT8_E4M3FN;
        ge::DataType out2Dtype = ge::DT_FLOAT8_E8M0;
        int64_t blockSize = 32;
        auto context_holder = gert::InferDataTypeContextFaker()
                    .IrInputNum(2)
                    .NodeIoNum(2, 2)
                    .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, in2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, out2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom((int64_t)outDtype)},
                                {"blocksize", Ops::NN::AnyValue::CreateFrom(blockSize)}})
                    .InputDataTypes({&inDtype, &in2Dtype})
                    .OutputDataTypes({&outDtype, &out2Dtype})
                    .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), inDtype);
        EXPECT_EQ(context->GetInputDataType(1), in2Dtype);
        EXPECT_EQ(context->GetOutputDataType(0), outDtype);
        EXPECT_EQ(context->GetOutputDataType(1), out2Dtype);
    }
}