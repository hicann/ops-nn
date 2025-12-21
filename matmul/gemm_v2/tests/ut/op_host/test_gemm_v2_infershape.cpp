/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "runtime/infer_shape_range_context.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

namespace ge {

class GemmV2 : public testing::Test {
    protected:
        static void SetUpTestCase() {
            std::cout << "GemmV2 Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase() {
        std::cout << "GemmV2 Proto Test TearDown" << std::endl;
        }
};

TEST_F(GemmV2, GemmV2_SetOutPutDataType_1) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV2")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType a = ge::DT_FLOAT16;
    ge::DataType b = ge::DT_FLOAT16;
    ge::DataType alpha = ge::DT_FLOAT16;
    ge::DataType beta = ge::DT_FLOAT16;
    ge::DataType c = ge::DT_FLOAT;
    ge::DataType out_datatype = ge::DT_FLOAT;
    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto context_holder = gert::InferDataTypeContextFaker()
      .IrInputNum(5)
      .NodeIoNum(5, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
      .InputDataTypes({&a, &b, &alpha, &beta, &c})
      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)}})
      .OutputDataTypes({&out_datatype})
      .Build();
  auto context = context_holder.GetContext<gert::InferDataTypeContext>();
  EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->GetOutputDataType(0), out_datatype);
}

TEST_F(GemmV2, GemmV2_SetOutPutDataType_2) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV2")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType a = ge::DT_BF16;
    ge::DataType b = ge::DT_BF16;
    ge::DataType alpha = ge::DT_BF16;
    ge::DataType beta = ge::DT_BF16;
    ge::DataType c = ge::DT_FLOAT;
    ge::DataType out_datatype = ge::DT_FLOAT;
    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto context_holder = gert::InferDataTypeContextFaker()
      .IrInputNum(5)
      .NodeIoNum(5, 1)
      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .InputDataTypes({&a, &b, &alpha, &beta, &c})
      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)}})
      .OutputDataTypes({&out_datatype})
      .Build();
  auto context = context_holder.GetContext<gert::InferDataTypeContext>();
  EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->GetOutputDataType(0), out_datatype);
}

TEST_F(GemmV2, GemmV2_SetOutPutShape_1) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV2")->infer_shape;

  gert::StorageShape a_shape = {{32, 64}, {32, 64}};
  gert::StorageShape b_shape = {{64, 128}, {64, 128}};
  gert::StorageShape alpha = {{1},{1}};
  gert::StorageShape beta = {{1},{1}};
  gert::StorageShape c = {{32, 128},{32, 128}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&a_shape, &b_shape, &alpha, &beta, &c})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  EXPECT_EQ(Ops::Base::ToString(*output), "[32, 128]");
}

TEST_F(GemmV2, GemmV2_SetOutPutShape_2) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV2")->infer_shape;

  gert::StorageShape a_shape = {{256, 1024}, {256, 1024}};
  gert::StorageShape b_shape = {{64, 1024}, {64, 1024}};
  gert::StorageShape alpha = {{1},{1}};
  gert::StorageShape beta = {{1},{1}};
  gert::StorageShape c = {{256, 64},{256, 64}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&a_shape, &b_shape, &alpha, &beta, &c})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  EXPECT_EQ(Ops::Base::ToString(*output), "[256, 64]");
}

TEST_F(GemmV2, GemmV2_SetOutPutShape_3) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmV2")->infer_shape;

  gert::StorageShape a_shape = {{1024, 512}, {1024, 512}};
  gert::StorageShape b_shape = {{64, 1024}, {64, 1024}};
  gert::StorageShape alpha = {{1},{1}};
  gert::StorageShape beta = {{1},{1}};
  gert::StorageShape c = {{512, 64},{512, 64}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&a_shape, &b_shape, &alpha, &beta, &c})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  EXPECT_EQ(Ops::Base::ToString(*output), "[512, 64]");
}
}