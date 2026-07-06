/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"

class LayerNormQuant : public testing::Test {
 protected:
  static void SetUpTestCase() {
    OP_LOGI("LayerNormQuant", "Proto Test SetUp");
  }

  static void TearDownTestCase() {
    OP_LOGI("LayerNormQuant", "Proto Test TearDown");
  }
};

TEST_F(LayerNormQuant, LayerNormQuant_infershape) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant"), nullptr);
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant")->infer_shape;
  ASSERT_NE(inferShapeFunc, nullptr);

  gert::StorageShape xShape = {{32, 32, 512}, {32, 32, 512}};
  gert::StorageShape gammaShape = {{1, 512}, {1, 512}};
  gert::StorageShape betaShape = {{1, 512}, {1, 512}};
  gert::StorageShape scaleShape = {{1}, {1}};
  gert::StorageShape offsetShape = {{1}, {1}};
  gert::StorageShape resultShape;

  auto holder = gert::InferShapeContextFaker()
      .NodeIoNum(5, 1)
      .IrInstanceNum({1, 1, 1, 1, 1})
      .InputShapes({&xShape, &gammaShape, &betaShape, &scaleShape, &offsetShape})
      .OutputShapes({&resultShape})
      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
      .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output0), "[32, 32, 512]");
}

TEST_F(LayerNormQuant, LayerNormQuant_infershape_failed_in_shape) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant"), nullptr);
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant")->infer_shape;
  ASSERT_NE(inferShapeFunc, nullptr);

  gert::StorageShape xShape = {{32, 32, 512}, {32, 32, 512}};
  gert::StorageShape gammaShape = {{1, 512}, {1, 512}};
  gert::StorageShape betaShape = {{1, 512, 2}, {1, 512, 2}};
  gert::StorageShape scaleShape = {{1}, {1}};
  gert::StorageShape offsetShape = {{1}, {1}};
  gert::StorageShape resultShape;

  auto holder = gert::InferShapeContextFaker()
      .NodeIoNum(5, 1)
      .IrInstanceNum({1, 1, 1, 1, 1})
      .InputShapes({&xShape, &gammaShape, &betaShape, &scaleShape, &offsetShape})
      .OutputShapes({&resultShape})
      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
      .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(LayerNormQuant, LayerNormQuant_infershape_failed_in_dim) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant"), nullptr);
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant")->infer_shape;
  ASSERT_NE(inferShapeFunc, nullptr);

  gert::StorageShape xShape = {{32, 32, 512}, {32, 32, 512}};
  gert::StorageShape gammaShape = {{1, 512}, {1, 512}};
  gert::StorageShape betaShape = {{10, 12}, {10, 12}};
  gert::StorageShape scaleShape = {{1}, {1}};
  gert::StorageShape offsetShape = {{1}, {1}};
  gert::StorageShape resultShape;

  auto holder = gert::InferShapeContextFaker()
      .NodeIoNum(5, 1)
      .IrInstanceNum({1, 1, 1, 1, 1})
      .InputShapes({&xShape, &gammaShape, &betaShape, &scaleShape, &offsetShape})
      .OutputShapes({&resultShape})
      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
      .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(LayerNormQuant, LayerNormQuant_infershape_failed_in_scaleSize) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant"), nullptr);
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant")->infer_shape;
  ASSERT_NE(inferShapeFunc, nullptr);

  gert::StorageShape xShape = {{32, 32, 512}, {32, 32, 512}};
  gert::StorageShape gammaShape = {{1, 512}, {1, 512}};
  gert::StorageShape betaShape = {{1, 512}, {1, 512}};
  gert::StorageShape scaleShape = {{2}, {2}};
  gert::StorageShape offsetShape = {{1}, {1}};
  gert::StorageShape resultShape;

  auto holder = gert::InferShapeContextFaker()
      .NodeIoNum(5, 1)
      .IrInstanceNum({1, 1, 1, 1, 1})
      .InputShapes({&xShape, &gammaShape, &betaShape, &scaleShape, &offsetShape})
      .OutputShapes({&resultShape})
      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
      .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(LayerNormQuant, LayerNormQuant_infershape_failed_in_offsetSize) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant"), nullptr);
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant")->infer_shape;
  ASSERT_NE(inferShapeFunc, nullptr);

  gert::StorageShape xShape = {{32, 32, 512}, {32, 32, 512}};
  gert::StorageShape gammaShape = {{1, 512}, {1, 512}};
  gert::StorageShape betaShape = {{1, 512}, {1, 512}};
  gert::StorageShape scaleShape = {{1}, {1}};
  gert::StorageShape offsetShape = {{3}, {3}};
  gert::StorageShape resultShape;

  auto holder = gert::InferShapeContextFaker()
      .NodeIoNum(5, 1)
      .IrInstanceNum({1, 1, 1, 1, 1})
      .InputShapes({&xShape, &gammaShape, &betaShape, &scaleShape, &offsetShape})
      .OutputShapes({&resultShape})
      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
      .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(LayerNormQuant, LayerNormQuant_inferdtype) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormQuant")->infer_datatype;
  ASSERT_NE(data_type_func, nullptr);
  ge::DataType dbf16 = ge::DT_BF16;
  ge::DataType dint8 = ge::DT_INT8;

  auto context_holder = gert::InferDataTypeContextFaker()
      .IrInputNum(5)
      .NodeIoNum(5, 1)
      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(4, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
      .InputDataTypes({&dbf16, &dbf16, &dbf16, &dbf16, &dint8})
      .OutputDataTypes({&dint8})
      .Build();
  auto context = context_holder.GetContext<gert::InferDataTypeContext>();
  EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), dint8);
}
