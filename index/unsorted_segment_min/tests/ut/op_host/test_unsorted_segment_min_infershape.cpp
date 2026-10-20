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
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "exe_graph/runtime/expand_dims_type.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"

using namespace ge;

class UnsortedSegmentMinInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UnsortedSegmentMinInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UnsortedSegmentMinInfershape TearDown" << std::endl; }
};

// 常规场景：x为3维，segment_ids为2维，num_segments为int32常量，推导输出为{num_segments, x.shape[2:]}
TEST_F(UnsortedSegmentMinInfershape, unsorted_segment_min_infershape_test_1)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentMin");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {2, 3, 4};
    gert::StorageShape xStorageShape = {{2, 3, 4}, {2, 3, 4}};
    gert::Shape segmentIdsShape = {2, 3};
    gert::StorageShape segmentIdsStorageShape = {{2, 3}, {2, 3}};
    gert::Shape numSegmentsShape = {1};
    gert::StorageShape numSegmentsStorageShape = {{1}, {1}};
    gert::Shape outputShape = {};
    int32_t numSegmentsData = 5;

    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor segmentIdsTensor(segmentIdsStorageShape,
                                  gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                  gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor numSegmentsTensor(numSegmentsStorageShape,
                                   gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                   gert::TensorPlacement::kOnHost, ge::DT_INT32, &numSegmentsData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &segmentIdsShape, &numSegmentsShape})
                      .OutputShapes({&outputShape})
                      .InputTensors({&xTensor, &segmentIdsTensor, &numSegmentsTensor})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {5, 4};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

// x为1维时输出为1维：y = {num_segments}
TEST_F(UnsortedSegmentMinInfershape, unsorted_segment_min_infershape_test_2)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentMin");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8};
    gert::StorageShape xStorageShape = {{8}, {8}};
    gert::Shape segmentIdsShape = {8};
    gert::StorageShape segmentIdsStorageShape = {{8}, {8}};
    gert::Shape numSegmentsShape = {1};
    gert::StorageShape numSegmentsStorageShape = {{1}, {1}};
    gert::Shape outputShape = {};
    int64_t numSegmentsData = 4;

    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr);
    gert::Tensor segmentIdsTensor(segmentIdsStorageShape,
                                  gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                  gert::TensorPlacement::kOnHost, ge::DT_INT64, nullptr);
    gert::Tensor numSegmentsTensor(numSegmentsStorageShape,
                                   gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                   gert::TensorPlacement::kOnHost, ge::DT_INT64, &numSegmentsData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &segmentIdsShape, &numSegmentsShape})
                      .OutputShapes({&outputShape})
                      .InputTensors({&xTensor, &segmentIdsTensor, &numSegmentsTensor})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {4};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

// num_segments为int64常量的常规场景
TEST_F(UnsortedSegmentMinInfershape, unsorted_segment_min_infershape_test_3)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentMin");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {4, 8};
    gert::StorageShape xStorageShape = {{4, 8}, {4, 8}};
    gert::Shape segmentIdsShape = {4};
    gert::StorageShape segmentIdsStorageShape = {{4}, {4}};
    gert::Shape numSegmentsShape = {1};
    gert::StorageShape numSegmentsStorageShape = {{1}, {1}};
    gert::Shape outputShape = {};
    int64_t numSegmentsData = 3;

    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor segmentIdsTensor(segmentIdsStorageShape,
                                  gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                  gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor numSegmentsTensor(numSegmentsStorageShape,
                                   gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                   gert::TensorPlacement::kOnHost, ge::DT_INT64, &numSegmentsData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &segmentIdsShape, &numSegmentsShape})
                      .OutputShapes({&outputShape})
                      .InputTensors({&xTensor, &segmentIdsTensor, &numSegmentsTensor})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {3, 8};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

// num_segments非常量（取不到常量值）时，输出第0维推导为-1
TEST_F(UnsortedSegmentMinInfershape, unsorted_segment_min_infershape_test_4)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentMin");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {6, 4};
    gert::StorageShape xStorageShape = {{6, 4}, {6, 4}};
    gert::Shape segmentIdsShape = {6};
    gert::StorageShape segmentIdsStorageShape = {{6}, {6}};
    gert::Shape numSegmentsShape = {1};
    gert::StorageShape numSegmentsStorageShape = {{1}, {1}};
    gert::Shape outputShape = {};

    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor segmentIdsTensor(segmentIdsStorageShape,
                                  gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                  gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor numSegmentsTensor(numSegmentsStorageShape,
                                   gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                   gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &segmentIdsShape, &numSegmentsShape})
                      .OutputShapes({&outputShape})
                      .InputTensors({&xTensor, &segmentIdsTensor, &numSegmentsTensor})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {-1, 4};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

// x为未知rank（-2）时，输出直接置为x的shape
TEST_F(UnsortedSegmentMinInfershape, unsorted_segment_min_infershape_test_5)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentMin");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-2};
    gert::StorageShape xStorageShape = {{-2}, {-2}};
    gert::Shape segmentIdsShape = {6};
    gert::StorageShape segmentIdsStorageShape = {{6}, {6}};
    gert::Shape numSegmentsShape = {1};
    gert::StorageShape numSegmentsStorageShape = {{1}, {1}};
    gert::Shape outputShape = {};
    int32_t numSegmentsData = 4;

    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor segmentIdsTensor(segmentIdsStorageShape,
                                  gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                  gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor numSegmentsTensor(numSegmentsStorageShape,
                                   gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                   gert::TensorPlacement::kOnHost, ge::DT_INT32, &numSegmentsData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &segmentIdsShape, &numSegmentsShape})
                      .OutputShapes({&outputShape})
                      .InputTensors({&xTensor, &segmentIdsTensor, &numSegmentsTensor})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

// segment_ids为未知rank（-2）时，输出直接置为segment_ids的shape
TEST_F(UnsortedSegmentMinInfershape, unsorted_segment_min_infershape_test_6)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentMin");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {6, 4};
    gert::StorageShape xStorageShape = {{6, 4}, {6, 4}};
    gert::Shape segmentIdsShape = {-2};
    gert::StorageShape segmentIdsStorageShape = {{-2}, {-2}};
    gert::Shape numSegmentsShape = {1};
    gert::StorageShape numSegmentsStorageShape = {{1}, {1}};
    gert::Shape outputShape = {};
    int32_t numSegmentsData = 4;

    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor segmentIdsTensor(segmentIdsStorageShape,
                                  gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                  gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor numSegmentsTensor(numSegmentsStorageShape,
                                   gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                   gert::TensorPlacement::kOnHost, ge::DT_INT32, &numSegmentsData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &segmentIdsShape, &numSegmentsShape})
                      .OutputShapes({&outputShape})
                      .InputTensors({&xTensor, &segmentIdsTensor, &numSegmentsTensor})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

// num_segments的shape size大于1时，推导失败
TEST_F(UnsortedSegmentMinInfershape, unsorted_segment_min_infershape_test_7)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("UnsortedSegmentMin");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {6, 4};
    gert::StorageShape xStorageShape = {{6, 4}, {6, 4}};
    gert::Shape segmentIdsShape = {6};
    gert::StorageShape segmentIdsStorageShape = {{6}, {6}};
    gert::Shape numSegmentsShape = {2};
    gert::StorageShape numSegmentsStorageShape = {{2}, {2}};
    gert::Shape outputShape = {};
    int32_t numSegmentsData[2] = {4, 4};

    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor segmentIdsTensor(segmentIdsStorageShape,
                                  gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                  gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor numSegmentsTensor(numSegmentsStorageShape,
                                   gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                   gert::TensorPlacement::kOnHost, ge::DT_INT32, numSegmentsData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &segmentIdsShape, &numSegmentsShape})
                      .OutputShapes({&outputShape})
                      .InputTensors({&xTensor, &segmentIdsTensor, &numSegmentsTensor})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
