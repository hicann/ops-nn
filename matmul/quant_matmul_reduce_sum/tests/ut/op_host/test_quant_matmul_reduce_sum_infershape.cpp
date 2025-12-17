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
 * \file test_quant_matmul_reduce_sum_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "test_cube_util.h"
#include "runtime/infer_shape_range_context.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

namespace ge {

class QuantMatmulReduceSumInferShapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantMatmulReduceSumInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantMatmulReduceSumInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(QuantMatmulReduceSumInferShapeTest, basic) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulReduceSum")->infer_shape;

  gert::StorageShape x1Shape = {{8, 2048, 1024}, {8, 2048, 1024}};
  gert::StorageShape x2Shape = {{8, 1024, 7168}, {8, 1024/32, 7168/16, 16, 32}};
  gert::StorageShape dimsShape = {{1}, {1}};
  gert::StorageShape biasShape = {{}, {}};
  gert::StorageShape x1ScaleShape = {{8, 2048}, {8, 2048}};
  gert::StorageShape x2ScaleShape = {{7168}, {7168}};
  gert::StorageShape outputShape = {{}, {}};
  int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
  gert::Shape expect_output_shape = {2048, 7168};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(6, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1, 1})
                    .InputShapes({&x1Shape, &x2Shape, &dimsShape, &biasShape, &x1ScaleShape, &x2ScaleShape})
                    .OutputShapes({&outputShape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}
}
