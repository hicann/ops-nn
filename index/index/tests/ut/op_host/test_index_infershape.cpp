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
 * \file test_index_infershape.cpp
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
#include "../../../op_graph/index_proto.h"

class IndexTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Index SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Index TearDown" << std::endl;
    }
};

TEST_F(IndexTest, index_infershape_success_01)
{
    gert::StorageShape x_shape = {{1000}, {1000}};
    gert::StorageShape indexed_sizes_shape = {{1}, {1}};
    gert::StorageShape indexed_strides_shape = {{1}, {1}};
    gert::StorageShape indices_shape = {{1}, {1}};
    gert::Shape y_shape = {{1}};
    gert::Shape expected_output_shape = {1};

    ge::DataType dtype = ge::DT_FLOAT;

    vector<int64_t> indexed_sizes_data{1};
    size_t total_size = 0;
    auto tensor_holder =
        gert::Tensor::CreateFollowing(indexed_sizes_shape.GetStorageShape().GetDimNum(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(indexed_sizes_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(indexed_sizes_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(
        tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), indexed_sizes_data.data(),
        indexed_sizes_data.size() * sizeof(int64_t));

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Index")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x_shape, tensor, &indexed_strides_shape, &indices_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expected_output_shape));
}

TEST_F(IndexTest, index_infershape_success_02)
{
    gert::StorageShape x_shape = {{24572, 7500}, {24572, 7500}};
    gert::StorageShape indexed_sizes_shape = {{2}, {2}};
    gert::StorageShape indexed_strides_shape = {{1}, {1}};
    gert::StorageShape indices_shape_01 = {{24572}, {24572}};
    gert::StorageShape indices_shape_02 = {{24572}, {24572}};
    gert::Shape y_shape = {{1}};
    gert::Shape expected_output_shape = {24572, 7500};

    ge::DataType dtype = ge::DT_FLOAT;

    vector<int64_t> indexed_sizes_data{1, 1};
    size_t total_size = 0;
    auto tensor_holder =
        gert::Tensor::CreateFollowing(indexed_sizes_shape.GetStorageShape().GetDimNum(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(indexed_sizes_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(indexed_sizes_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(
        tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), indexed_sizes_data.data(),
        indexed_sizes_data.size() * sizeof(int64_t));

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Index")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 2})
                      .InputShapes({&x_shape, tensor, &indexed_strides_shape, &indices_shape_01, &indices_shape_02})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expected_output_shape));
}

TEST_F(IndexTest, index_infershape_success_03)
{
    gert::StorageShape x_shape = {{26, 51866}, {26, 51866}};
    gert::StorageShape indexed_sizes_shape = {{2}, {2}};
    gert::StorageShape indexed_strides_shape = {{1}, {1}};
    gert::StorageShape indices_shape_01 = {{466794}, {466794}};
    gert::StorageShape indices_shape_02 = {{466794}, {466794}};
    gert::Shape y_shape = {{1}};
    gert::Shape expected_output_shape = {26, 51866};

    ge::DataType dtype = ge::DT_FLOAT;

    vector<int64_t> indexed_sizes_data{1, 1};
    size_t total_size = 0;
    auto tensor_holder =
        gert::Tensor::CreateFollowing(indexed_sizes_shape.GetStorageShape().GetDimNum(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(indexed_sizes_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(indexed_sizes_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(
        tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), indexed_sizes_data.data(),
        indexed_sizes_data.size() * sizeof(int64_t));

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Index")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 2})
                      .InputShapes({&x_shape, tensor, &indexed_strides_shape, &indices_shape_01, &indices_shape_02})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expected_output_shape));
}


TEST_F(IndexTest, index_inferdtype_success_01)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Index")->infer_datatype;

    ge::DataType x_dtype = ge::DT_FLOAT16;
    ge::DataType y_dtype = ge::DT_FLOAT16;
    ge::DataType argmax_dtype = ge::DT_INT32;
    ge::DataType expect_output_dtype = ge::DT_FLOAT16;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({
                          1,1,1,1
                      })
                      .InputDataTypes({&x_dtype, &x_dtype, &x_dtype, &x_dtype})
                      .OutputDataTypes({&y_dtype})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), expect_output_dtype);
}