/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_max_pool_v2_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "../../../op_graph/max_pool_v2_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class MaxPoolV2 : public testing::Test
{
protected:
    static void SetUpTestCase() {
        std::cout << "MaxPoolV2 SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MaxPoolV2 TearDown" << std::endl;
    }
};

TEST_F(MaxPoolV2, MaxPoolV2_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape kSizeShape = {{4}, {4}};
    gert::StorageShape stridesShape = {{4}, {4}};
    gert::StorageShape yShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};

    vector<int32_t> constInputData{1, 1, 1, 1};
    size_t totalSize = 0;
    auto tensorHolder =
        gert::Tensor::CreateFollowing(kSizeShape.GetStorageShape().GetDim(0), ge::DT_INT32, totalSize);
    auto ksizeTensor = reinterpret_cast<gert::Tensor*>(tensorHolder.get());
    ksizeTensor->MutableStorageShape().AppendDim(kSizeShape.GetStorageShape().GetDim(0));
    ksizeTensor->MutableOriginShape().AppendDim(kSizeShape.GetOriginShape().GetDim(0));
    ksizeTensor->SetOriginFormat(ge::FORMAT_ND);
    ksizeTensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(
        ksizeTensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), constInputData.data(),
        constInputData.size() * sizeof(int32_t));
    
    auto tensorHolder1 =
        gert::Tensor::CreateFollowing(stridesShape.GetStorageShape().GetDim(0), ge::DT_INT32, totalSize);
    auto stridesTensor = reinterpret_cast<gert::Tensor*>(tensorHolder1.get());
    stridesTensor->MutableStorageShape().AppendDim(stridesShape.GetStorageShape().GetDim(0));
    stridesTensor->MutableOriginShape().AppendDim(stridesShape.GetOriginShape().GetDim(0));
    stridesTensor->SetOriginFormat(ge::FORMAT_ND);
    stridesTensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(
        stridesTensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), constInputData.data(),
        constInputData.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")}
                      })
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolV2, MaxPoolV2_infershape_test2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV2")->infer_shape;

    gert::StorageShape xShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape kSizeShape = {{4}, {4}};
    gert::StorageShape stridesShape = {{4}, {4}};
    gert::StorageShape yShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};

    vector<int32_t> constInputData{1, 1, 1, 1};
    size_t totalSize = 0;
    auto tensorHolder =
        gert::Tensor::CreateFollowing(kSizeShape.GetStorageShape().GetDim(0), ge::DT_INT32, totalSize);
    auto ksizeTensor = reinterpret_cast<gert::Tensor*>(tensorHolder.get());
    ksizeTensor->MutableStorageShape().AppendDim(kSizeShape.GetStorageShape().GetDim(0));
    ksizeTensor->MutableOriginShape().AppendDim(kSizeShape.GetOriginShape().GetDim(0));
    ksizeTensor->SetOriginFormat(ge::FORMAT_ND);
    ksizeTensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(
        ksizeTensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), constInputData.data(),
        constInputData.size() * sizeof(int32_t));
    
    auto tensorHolder1 =
        gert::Tensor::CreateFollowing(stridesShape.GetStorageShape().GetDim(0), ge::DT_INT32, totalSize);
    auto stridesTensor = reinterpret_cast<gert::Tensor*>(tensorHolder1.get());
    stridesTensor->MutableStorageShape().AppendDim(stridesShape.GetStorageShape().GetDim(0));
    stridesTensor->MutableOriginShape().AppendDim(stridesShape.GetOriginShape().GetDim(0));
    stridesTensor->SetOriginFormat(ge::FORMAT_ND);
    stridesTensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(
        stridesTensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), constInputData.data(),
        constInputData.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")}
                      })
                      .InputShapes({&xShape, ksizeTensor, stridesTensor})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}