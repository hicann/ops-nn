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
#include "op_proto_test_util.h"
#include "nn_training_ops.h"
#include "common/utils/ut_op_common.h"
#include "array_ops.h"
#include "util/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

class appadamd : public testing::Test {
  protected:
  static void SetUpTestCase() {
      std::cout << "appadamd SetUp" << std::endl;
  }

  static void TearDownTestCase() {
      std::cout << "appadamd TearDown" << std::endl;
  }
};

static ge::Operator BuildGraph4InferAxisType(const std::initializer_list<int64_t>& dims, const ge::Format& format,
                                             const ge::DataType& dataType) {
  auto apply_op = op::ApplyAdamD("ApplyAdamD");
  TENSOR_INPUT_WITH_SHAPE(apply_op, var, dims, dataType, format, {});
  TENSOR_INPUT_WITH_SHAPE(apply_op, m, dims, dataType, format, {});
  TENSOR_INPUT_WITH_SHAPE(apply_op, v, dims, dataType, format, {});
  TENSOR_INPUT_WITH_SHAPE(apply_op, beta1_power, {1}, dataType, format, {});
  TENSOR_INPUT_WITH_SHAPE(apply_op, beta2_power, {1}, dataType, format, {});
  TENSOR_INPUT_WITH_SHAPE(apply_op, lr, {1}, dataType, format, {});
  TENSOR_INPUT_WITH_SHAPE(apply_op, beta1, {1}, dataType, format, {});
  TENSOR_INPUT_WITH_SHAPE(apply_op, beta2, {1}, dataType, format, {});
  TENSOR_INPUT_WITH_SHAPE(apply_op, epsilon, {1}, dataType, format, {});
  TENSOR_INPUT_WITH_SHAPE(apply_op, grad, dims, dataType, format, {});
  apply_op.InferShapeAndType();
  return apply_op;
}

TEST_F(appadamd, infer_axis_type_with_input_dim_num_great_than_1) {
  auto op1 = BuildGraph4InferAxisType({-1, -1}, ge::FORMAT_ND, ge::DT_FLOAT16);

  ge::InferAxisTypeInfoFunc infer_func = GetInferAxisTypeFunc("ApplyAdamD");
  EXPECT_NE(infer_func, nullptr);

  std::vector<ge::AxisTypeInfo> infos;
  const uint32_t infer_axis_ret = static_cast<uint32_t>(infer_func(op1, infos));
  EXPECT_EQ(infer_axis_ret, ge::GRAPH_SUCCESS);

  std::vector<ge::AxisTypeInfo> expect_infos = {
    ge::AxisTypeInfoBuilder()
      .AxisType(ge::AxisType::ELEMENTWISE)
      .AddInputCutInfo({0, {0}})
      .AddInputCutInfo({1, {0}})
      .AddInputCutInfo({2, {0}})
      .AddInputCutInfo({9, {0}})
      .AddOutputCutInfo({0, {0}})
      .AddOutputCutInfo({1, {0}})
      .AddOutputCutInfo({2, {0}})
      .Build(),
    ge::AxisTypeInfoBuilder()
      .AxisType(ge::AxisType::ELEMENTWISE)
      .AddInputCutInfo({0, {1}})
      .AddInputCutInfo({1, {1}})
      .AddInputCutInfo({2, {1}})
      .AddInputCutInfo({9, {1}})
      .AddOutputCutInfo({0, {1}})
      .AddOutputCutInfo({1, {1}})
      .AddOutputCutInfo({2, {1}})
      .Build(),
  };

  EXPECT_STREQ(AxisTypeInfoToString(infos).c_str(),
               AxisTypeInfoToString(expect_infos).c_str());
}