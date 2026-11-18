/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "infer_datatype_context_faker.h"
#include "infer_shape_context_faker.h"
#include "op_impl_registry.h"

namespace {
constexpr size_t INPUT_COUNT = 2;
constexpr size_t OUTPUT_COUNT = 1;
constexpr size_t BASE_ADDR_INPUT = 0;
constexpr size_t INDEX_INPUT = 1;
constexpr size_t ADDRS_TABLE_OUTPUT = 0;
constexpr int64_t ORI_SHAPE_ROW = 16;
constexpr int64_t ORI_SHAPE_COL = 16;
constexpr int64_t OUTPUT_COL_NUM = 4;
constexpr const char* OP_TYPE = "IndexToAddr";
using Attrs = std::vector<std::pair<std::string, Ops::NN::AnyValue>>;

Attrs MakeAttrs(const std::vector<int64_t>& blockSize)
{
    return {{"ori_shape", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({ORI_SHAPE_ROW, ORI_SHAPE_COL})},
            {"block_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(blockSize)}};
}

gert::Shape MakeInputShape() { return gert::Shape({2}); }

class IndexToAddrInferTest : public testing::Test {
protected:
    void SetUp() override
    {
        impl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
        ASSERT_NE(impl, nullptr);
        ASSERT_NE(impl->infer_shape, nullptr);
        ASSERT_NE(impl->infer_datatype, nullptr);
    }
    const gert::OpImplKernelRegistry::OpImplFunctions* impl = nullptr;
};

TEST_F(IndexToAddrInferTest, InferShapeSucceedsWithValidBlockSize)
{
    for (const auto& blockSize : std::vector<std::vector<int64_t>>{{4, 4}, {8, 2}, {16, 4}}) {
        gert::Shape baseAddrShape = MakeInputShape();
        gert::Shape indexShape = MakeInputShape();
        gert::Shape outShape;
        auto holder = gert::InferShapeContextFaker()
                          .SetOpType(OP_TYPE)
                          .NodeIoNum(INPUT_COUNT, OUTPUT_COUNT)
                          .NodeInputTd(BASE_ADDR_INPUT, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(INDEX_INPUT, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .InputShapes({&baseAddrShape, &indexShape})
                          .OutputShapes({&outShape})
                          .NodeAttrs(MakeAttrs(blockSize))
                          .Build();
        auto* context = holder.GetContext<gert::InferShapeContext>();
        ASSERT_EQ(impl->infer_shape(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(*context->GetOutputShape(ADDRS_TABLE_OUTPUT), gert::Shape({blockSize[0], OUTPUT_COL_NUM}));
    }
}

TEST_F(IndexToAddrInferTest, InferShapeRejectsInvalidBlockSizeListSize)
{
    for (const auto& blockSize : std::vector<std::vector<int64_t>>{{}, {4}, {4, 4, 4}}) {
        gert::Shape baseAddrShape = MakeInputShape();
        gert::Shape indexShape = MakeInputShape();
        gert::Shape outShape;
        auto holder = gert::InferShapeContextFaker()
                          .SetOpType(OP_TYPE)
                          .NodeIoNum(INPUT_COUNT, OUTPUT_COUNT)
                          .NodeInputTd(BASE_ADDR_INPUT, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(INDEX_INPUT, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                          .InputShapes({&baseAddrShape, &indexShape})
                          .OutputShapes({&outShape})
                          .NodeAttrs(MakeAttrs(blockSize))
                          .Build();
        auto* context = holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
        EXPECT_EQ(*context->GetOutputShape(ADDRS_TABLE_OUTPUT), gert::Shape());
    }
}

TEST_F(IndexToAddrInferTest, InferDataTypeCopiesInputDtype)
{
    for (auto dtype : {ge::DT_INT64, ge::DT_UINT64}) {
        auto holder = gert::InferDataTypeContextFaker()
                          .SetOpType(OP_TYPE)
                          .NodeIoNum(INPUT_COUNT, OUTPUT_COUNT)
                          .NodeInputTd(BASE_ADDR_INPUT, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(INDEX_INPUT, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(ADDRS_TABLE_OUTPUT, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs(MakeAttrs({4, 4}))
                          .Build();
        auto* context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(ADDRS_TABLE_OUTPUT), dtype);
    }
}
} // namespace
