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
#include <securec.h>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "../../../op_graph/experimental_index_proto.h"

namespace {
void ExpectShapeEq(const gert::Shape& actual, const std::vector<int64_t>& expected)
{
    ASSERT_EQ(actual.GetDimNum(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual.GetDim(i), expected[i]);
    }
}

std::unique_ptr<uint8_t[]> MakeConstTensor(const gert::StorageShape& shape, const std::vector<int64_t>& data)
{
    std::unique_ptr<uint8_t[]> tensorHolder(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * data.size()]);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensorHolder.get());
    gert::Tensor tensorValue(shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, ge::DT_INT64, nullptr);
    std::memcpy(tensor, &tensorValue, sizeof(gert::Tensor));
    auto tensorData = reinterpret_cast<int64_t*>(tensor + 1);
    for (size_t i = 0; i < data.size(); ++i) {
        tensorData[i] = data[i];
    }
    tensor->SetData(gert::TensorData(tensorData, nullptr, sizeof(int64_t) * data.size(), gert::kFollowing));
    return tensorHolder;
}
}  // namespace

class TestIndexInfershape : public testing::Test {
};

TEST_F(TestIndexInfershape, index_aicore_infershape_continuous_indices)
{
    gert::StorageShape xShape = {{8, 9, 4}, {8, 9, 4}};
    gert::StorageShape indexedSizesShape = {{3}, {3}};
    gert::StorageShape indexedStridesShape = {{3}, {3}};
    gert::StorageShape index0Shape = {{2, 1}, {2, 1}};
    gert::StorageShape index1Shape = {{1, 3}, {1, 3}};
    gert::Shape yShape = {};
    std::vector<int64_t> expectedShape = {2, 3, 4};
    auto indexedSizes = MakeConstTensor(indexedSizesShape, {1, 1, 0});
    auto indexedSizesTensor = reinterpret_cast<gert::Tensor*>(indexedSizes.get());

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexAiCore")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 2}, {1})
                      .InputShapes({&xShape, indexedSizesTensor, &indexedStridesShape, &index0Shape, &index1Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ExpectShapeEq(*output, expectedShape);
}

TEST_F(TestIndexInfershape, index_aicore_infershape_non_continuous_indices)
{
    gert::StorageShape xShape = {{5, 6, 7}, {5, 6, 7}};
    gert::StorageShape indexedSizesShape = {{3}, {3}};
    gert::StorageShape indexedStridesShape = {{3}, {3}};
    gert::StorageShape index0Shape = {{4}, {4}};
    gert::StorageShape index1Shape = {{4}, {4}};
    gert::Shape yShape = {};
    std::vector<int64_t> expectedShape = {4, 6};
    auto indexedSizes = MakeConstTensor(indexedSizesShape, {1, 0, 1});
    auto indexedSizesTensor = reinterpret_cast<gert::Tensor*>(indexedSizes.get());

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexAiCore")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 2}, {1})
                      .InputShapes({&xShape, indexedSizesTensor, &indexedStridesShape, &index0Shape, &index1Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ExpectShapeEq(*output, expectedShape);
}

TEST_F(TestIndexInfershape, index_aicore_infershape_invalid_indexed_sizes_value)
{
    gert::StorageShape xShape = {{8, 9}, {8, 9}};
    gert::StorageShape indexedSizesShape = {{2}, {2}};
    gert::StorageShape indexedStridesShape = {{2}, {2}};
    gert::StorageShape indexShape = {{2}, {2}};
    gert::Shape yShape = {};
    auto indexedSizes = MakeConstTensor(indexedSizesShape, {2, 0});
    auto indexedSizesTensor = reinterpret_cast<gert::Tensor*>(indexedSizes.get());

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexAiCore")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1}, {1})
                      .InputShapes({&xShape, indexedSizesTensor, &indexedStridesShape, &indexShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TestIndexInfershape, index_aicore_inferdtype_follows_x)
{
    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType sizesDtype = ge::DT_INT64;
    ge::DataType yDtype = ge::DT_FLOAT;

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexAiCore")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1}, {1})
                      .InputDataTypes({&xDtype, &sizesDtype, &sizesDtype, &sizesDtype})
                      .OutputDataTypes({&yDtype})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}
