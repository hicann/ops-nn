/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "../../../op_graph/unsorted_segment_sum_proto.h"

namespace {
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 1;
constexpr size_t kOutputIndex = 0;

class UnsortedSegmentSumInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UnsortedSegmentSumInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UnsortedSegmentSumInfershapeTest TearDown" << std::endl; }
};

// 与tests/ut/common/infershape_case_executor.cpp的Tensor构造方式一致：
// 常量输入(constValue非空)的TensorData携带host侧常量值地址，供GetConstInt读取；
// 非常量输入地址传空
std::shared_ptr<gert::Tensor> CreateTensor(const std::initializer_list<int64_t>& dims, ge::DataType dtype,
                                           void* constValue = nullptr)
{
    return std::make_shared<gert::Tensor>(gert::StorageShape(dims, dims),
                                          gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                          gert::TensorPlacement::kOnHost, dtype, constValue);
}
} // namespace

TEST_F(UnsortedSegmentSumInfershapeTest, unsorted_segment_sum_infershape_int32_num_segments_1d_segment_ids)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    int32_t numSegments = 5;
    std::vector<std::shared_ptr<gert::Tensor>> inputTensors = {
        CreateTensor({8, 16}, ge::DT_FLOAT),
        CreateTensor({8}, ge::DT_INT32),
        CreateTensor({1}, ge::DT_INT32, &numSegments),
    };
    constexpr int64_t expectedRank = 2;
    constexpr int64_t expectedDim0 = 5;
    constexpr int64_t expectedDim1 = 16;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .NodeOutputTd(kOutputIndex, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({inputTensors[0].get(), inputTensors[1].get(), inputTensors[2].get()})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), expectedRank);
    EXPECT_EQ(outputShape->GetDim(0), expectedDim0);
    EXPECT_EQ(outputShape->GetDim(1), expectedDim1);
}

TEST_F(UnsortedSegmentSumInfershapeTest, unsorted_segment_sum_infershape_int64_num_segments_2d_segment_ids)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    int64_t numSegments = 6;
    std::vector<std::shared_ptr<gert::Tensor>> inputTensors = {
        CreateTensor({2, 4, 16, 32}, ge::DT_FLOAT16),
        CreateTensor({2, 4}, ge::DT_INT64),
        CreateTensor({1}, ge::DT_INT64, &numSegments),
    };
    constexpr int64_t expectedRank = 3;
    constexpr int64_t expectedDim0 = 6;
    constexpr int64_t expectedDim1 = 16;
    constexpr int64_t expectedDim2 = 32;
    constexpr size_t expectedDim2Index = 2;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .NodeOutputTd(kOutputIndex, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({inputTensors[0].get(), inputTensors[1].get(), inputTensors[2].get()})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), expectedRank);
    EXPECT_EQ(outputShape->GetDim(0), expectedDim0);
    EXPECT_EQ(outputShape->GetDim(1), expectedDim1);
    EXPECT_EQ(outputShape->GetDim(expectedDim2Index), expectedDim2);
}

TEST_F(UnsortedSegmentSumInfershapeTest, unsorted_segment_sum_infershape_non_const_num_segments)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    std::vector<std::shared_ptr<gert::Tensor>> inputTensors = {
        CreateTensor({8, 16}, ge::DT_DOUBLE),
        CreateTensor({8}, ge::DT_INT32),
        CreateTensor({1}, ge::DT_INT32),
    };
    constexpr int64_t expectedRank = 2;
    constexpr int64_t expectedDim0 = -1;
    constexpr int64_t expectedDim1 = 16;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .NodeOutputTd(kOutputIndex, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({inputTensors[0].get(), inputTensors[1].get(), inputTensors[2].get()})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), expectedRank);
    EXPECT_EQ(outputShape->GetDim(0), expectedDim0);
    EXPECT_EQ(outputShape->GetDim(1), expectedDim1);
}

TEST_F(UnsortedSegmentSumInfershapeTest, unsorted_segment_sum_infershape_x_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    int32_t numSegments = 5;
    std::vector<std::shared_ptr<gert::Tensor>> inputTensors = {
        CreateTensor({-2}, ge::DT_FLOAT),
        CreateTensor({8}, ge::DT_INT32),
        CreateTensor({1}, ge::DT_INT32, &numSegments),
    };
    constexpr int64_t expectedDim0 = -2;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .NodeOutputTd(kOutputIndex, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({inputTensors[0].get(), inputTensors[1].get(), inputTensors[2].get()})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), 1);
    EXPECT_EQ(outputShape->GetDim(0), expectedDim0);
}

TEST_F(UnsortedSegmentSumInfershapeTest, unsorted_segment_sum_infershape_num_segments_not_scalar)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    std::vector<std::shared_ptr<gert::Tensor>> inputTensors = {
        CreateTensor({8, 16}, ge::DT_FLOAT),
        CreateTensor({8}, ge::DT_INT32),
        CreateTensor({2}, ge::DT_INT32),
    };

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .NodeOutputTd(kOutputIndex, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({inputTensors[0].get(), inputTensors[1].get(), inputTensors[2].get()})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(UnsortedSegmentSumInfershapeTest, unsorted_segment_sum_infershape_segment_ids_rank_greater_than_x)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    int32_t numSegments = 5;
    std::vector<std::shared_ptr<gert::Tensor>> inputTensors = {
        CreateTensor({8}, ge::DT_FLOAT),
        CreateTensor({8, 2}, ge::DT_INT32),
        CreateTensor({1}, ge::DT_INT32, &numSegments),
    };

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .NodeOutputTd(kOutputIndex, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({inputTensors[0].get(), inputTensors[1].get(), inputTensors[2].get()})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
