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
 * \file test_embedding_hash_table_evict_infershape.cpp
 * \brief embedding_hash_table_evict infer test
 */

#include <iostream>
#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "../../../op_graph/embedding_hash_table_evict_proto.h"

namespace {
class EmbeddingHashTableEvictInferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EmbeddingHashTableEvict Infer Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "EmbeddingHashTableEvict Infer Test TearDown" << std::endl; }
};

TEST_F(EmbeddingHashTableEvictInferTest, infer_dtype_without_optional_sampled_values)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableEvict"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableEvict")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType tableHandleDtype = ge::DT_INT64;
    ge::DataType keysDtype = ge::DT_INT64;
    ge::DataType outputDtype = ge::DT_INT64;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 1)
                             .IrInstanceNum({1, 1, 0}, {1})
                             .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&tableHandleDtype, &keysDtype})
                             .OutputDataTypes({&outputDtype})
                             .Build();

    EXPECT_EQ(dataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingHashTableEvictInferTest, infer_dtype_with_optional_sampled_values)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableEvict"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableEvict")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType tableHandleDtype = ge::DT_INT64;
    ge::DataType keysDtype = ge::DT_INT64;
    ge::DataType sampledValuesDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_INT64;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 1)
                             .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&tableHandleDtype, &keysDtype, &sampledValuesDtype})
                             .OutputDataTypes({&outputDtype})
                             .Build();

    EXPECT_EQ(dataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingHashTableEvictInferTest, infer_dtype_fail_when_keys_not_int64)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableEvict"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableEvict")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType tableHandleDtype = ge::DT_INT64;
    ge::DataType keysDtype = ge::DT_INT32;
    ge::DataType outputDtype = ge::DT_INT64;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 1)
                             .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&tableHandleDtype, &keysDtype})
                             .OutputDataTypes({&outputDtype})
                             .Build();

    EXPECT_EQ(dataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}
} // namespace
