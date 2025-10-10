/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "../../../norm/group_norm_swish_grad/tests/ut/op_host/nn_norm.h"

class GroupNormSwishProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupNormSwish Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupNormSwish Proto Test TearDown" << std::endl;
    }
};

TEST_F(GroupNormSwishProto, group_norm_swish_infershape_test){
  ge::op::GroupNormSwish op;
  op.UpdateInputDesc("x", create_desc({1, 1152, 64, 64}, ge::DT_FLOAT16));
  op.UpdateInputDesc("gamma", create_desc({1152}, ge::DT_FLOAT16));
  op.UpdateInputDesc("beta", create_desc({1152}, ge::DT_FLOAT16));
  op.SetAttr("num_groups", 32);
  Runtime2TestParam param{{"num_groups"}};

  auto ret = InferShapeTest(op, param);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  std::vector<int64_t> expected_output_shape = {1, 1152, 64, 64};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GroupNormSwishProto, group_norm_swish_inferdtype_test) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormSwish"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormSwish")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_FLOAT16;
    ge::DataType output_ref = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 3)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&input_ref, &input_ref, &input_ref})
        .OutputDataTypes({&output_ref})
        .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), output_ref);
  }
}
