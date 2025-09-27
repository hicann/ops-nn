/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dequant_swiglu_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry_base.h"
#include "platform/platform_infos_def.h"
#include "../../../op_graph/dequant_swiglu_quant_proto.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/op_desc_utils.h"
#include "tiling/platform/platform_ascendc.h"
class DequantSwigluQuantTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DequantSwigluQuantTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DequantSwigluQuantTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(DequantSwigluQuantTest, DequantSwigluQuant_infershape_diff_test_legal_input) {
  ge::op::DequantSwigluQuant op;
  op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
  op.SetAttr("activate_left", false);
  op.SetAttr("quant_mode", "static");
  Runtime2TestParam param{{"activate_left", "quant_mode"}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc(0);
  std::vector<int64_t> expected_output_shape = {4, 2};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DequantSwigluQuantTest, DequantSwigluQuant_infer_dtype_test) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DequantSwigluQuant"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DequantSwigluQuant")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_FLOAT16;
    ge::DataType output_ref0 = ge::DT_INT8;
    ge::DataType output_ref = ge::DT_FLOAT;
    auto context_holder =
        gert::InferDataTypeContextFaker()
            .IrInputNum(6)
            .NodeIoNum(6, 2)
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&input_ref, &output_ref, &output_ref, &output_ref, &output_ref, &output_ref})
            .OutputDataTypes({&output_ref0, &output_ref})
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
  }
}