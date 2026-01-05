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

class ApplyAdagradDTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAdagradD SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAdagradD TearDown" << std::endl;
  }
};

TEST_F(ApplyAdagradDTest, scewlg_infer_shape_fp16) {
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdagradD")->infer_shape;
    gert::StorageShape Shape1 = {{64}, {-1}};
    gert::StorageShape Shape2 = {{1}, {-1}};


    gert::StorageShape varShape = {{}, {}};
    gert::StorageShape accumShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4, 1})
                      .InputShapes({&Shape1, &Shape1, &Shape2, &Shape1})
                      .OutputShapes({&varShape,&accumShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"update_slots", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {64};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}