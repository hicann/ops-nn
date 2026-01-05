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
#include <vector>
#include "array_ops.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class SoftmaxV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SoftmaxV2Test Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SoftmaxV2Test Proto Test TearDown" << std::endl;
  }
};


TEST_F(SoftmaxV2Test, softmax_v2_test_1) {
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxV2")->infer_shape;

    gert::StorageShape xShape = {{9, 35}, {9, 35}};
    gert::StorageShape yShape = {{}, {}};
    std::vector<int64_t> axes = {1};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axes", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(axes)}, {"half_to_float", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {9, 35};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}