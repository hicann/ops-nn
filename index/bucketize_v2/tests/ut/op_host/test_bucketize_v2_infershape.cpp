/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gelu_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/bucketize_v2_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class BucketizeV2ProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gelu Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gelu Proto Test TearDown" << std::endl;
  }
};

TEST_F(BucketizeV2ProtoTest, bucketize_v2_infershape_test0) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BucketizeV2")->infer_shape;

  gert::Shape input_shape_0 = {4, 3, 4};
  gert::Shape input_shape_1 = {2};
  gert::Shape output_shape_0 = {};
  ge::DataType output_dtype_ref = ge::DT_FLOAT;
  bool out_int32 = true;
  bool right = false;
  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&input_shape_0, &input_shape_1})
                    .OutputShapes({&output_shape_0})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, output_dtype_ref, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"out_int32", Ops::NN::AnyValue::CreateFrom<bool>(out_int32)},
                      {"right", Ops::NN::AnyValue::CreateFrom<bool>(right)}
                      })
                    .Build();
  
  auto context = holder.GetContext<gert::InferShapeContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

  auto output_shape = context->GetOutputShape(0);
  ASSERT_NE(output_shape, nullptr);
  EXPECT_EQ(*output_shape, input_shape_0);
}

TEST_F(BucketizeV2ProtoTest, bucketize_v2_inferdtype_test0) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BucketizeV2")->infer_datatype;


  ge::DataType output_dtype_ref = ge::DT_FLOAT;
  ge::DataType expect_out_dtype = ge::DT_INT32;
  ge::DataType inputDtype  = ge::DT_FLOAT;
  bool out_int32 = true;
  bool right = false;
  auto holder = gert::InferDataTypeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputDataTypes({&inputDtype, &inputDtype})
                    .OutputDataTypes({&output_dtype_ref})
                    .NodeAttrs({
                      {"out_int32", Ops::NN::AnyValue::CreateFrom<bool>(out_int32)},
                      {"right", Ops::NN::AnyValue::CreateFrom<bool>(right)}
                      })
                    .Build();
  
  auto context = holder.GetContext<gert::InferDataTypeContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);

  auto output_dtype = context->GetOutputDataType(0);

  EXPECT_EQ(output_dtype, expect_out_dtype);
}

TEST_F(BucketizeV2ProtoTest, bucketize_v2_inferdtype_test1) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BucketizeV2")->infer_datatype;


  ge::DataType output_dtype_ref = ge::DT_FLOAT;
  ge::DataType expect_out_dtype = ge::DT_INT64;
  ge::DataType inputDtype  = ge::DT_FLOAT;
  bool out_int32 = false;
  bool right = false;
  auto holder = gert::InferDataTypeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputDataTypes({&inputDtype, &inputDtype})
                    .OutputDataTypes({&output_dtype_ref})
                    .NodeAttrs({
                      {"out_int32", Ops::NN::AnyValue::CreateFrom<bool>(out_int32)},
                      {"right", Ops::NN::AnyValue::CreateFrom<bool>(right)}
                      })
                    .Build();
  
  auto context = holder.GetContext<gert::InferDataTypeContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);

  auto output_dtype = context->GetOutputDataType(0);

  EXPECT_EQ(output_dtype, expect_out_dtype);
}
