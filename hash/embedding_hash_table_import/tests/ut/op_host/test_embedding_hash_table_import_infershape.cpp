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
 * \file test_embedding_hash_table_import_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/embedding_hash_table_import_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

namespace {
// ----------------EmbeddingHashTableImport-------------------
class EmbeddingHashTableImportProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EmbeddingHashTableImport Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "EmbeddingHashTableImport Proto Test TearDown" << std::endl; }
};

//   pass cases
TEST_F(EmbeddingHashTableImportProtoTest, embedding_hash_table_import_infer_shape_test1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableImport")->infer_shape;

    gert::StorageShape table_handles_shape = {{1},
                                              {
                                                  1,
                                              }};
    gert::StorageShape embedding_dims_shape = {{
                                                   1,
                                               },
                                               {
                                                   1,
                                               }};
    gert::StorageShape bucket_sizes_shape = {{
                                                 1,
                                             },
                                             {
                                                 1,
                                             }};
    gert::StorageShape keys_shape = {{
                                         2,
                                     },
                                     {
                                         2,
                                     }};
    gert::StorageShape counters_shape = {{
                                             2,
                                         },
                                         {
                                             2,
                                         }};
    gert::StorageShape filter_flags_shape = {{
                                                 2,
                                             },
                                             {
                                                 2,
                                             }};
    gert::StorageShape values_shape = {{
                                           10,
                                       },
                                       {
                                           10,
                                       }};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&table_handles_shape, &embedding_dims_shape, &bucket_sizes_shape, &keys_shape,
                                    &counters_shape, &filter_flags_shape, &values_shape})
                      .OutputShapes({&table_handles_shape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingHashTableImportProtoTest, embedding_hash_table_import_infer_dtype_test1)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableImport");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1}, {1})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(dataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingHashTableImportProtoTest, embedding_hash_table_import_infer_dtype_counters_int64)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableImport");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1}, {1})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(dataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}

// listShapes 按 keys、counters、filter_flags、values 顺序平铺展开
static gert::KernelRunContextHolder BuildImportInferShapeHolder(const std::vector<uint32_t>& irInstanceNum,
                                                                gert::Shape tableParamsShape,
                                                                const std::vector<gert::Shape>& listShapes)
{
    std::vector<gert::Shape> inputShapes = {tableParamsShape, tableParamsShape, tableParamsShape};
    inputShapes.insert(inputShapes.end(), listShapes.begin(), listShapes.end());

    std::vector<ge::DataType> dtypes = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
    for (uint32_t k = 0; k < irInstanceNum[3]; k++) {
        dtypes.push_back(ge::DT_INT64);
    }
    for (uint32_t k = 0; k < irInstanceNum[4]; k++) {
        dtypes.push_back(ge::DT_UINT64);
    }
    for (uint32_t k = 0; k < irInstanceNum[5]; k++) {
        dtypes.push_back(ge::DT_UINT8);
    }
    for (uint32_t k = 0; k < irInstanceNum[6]; k++) {
        dtypes.push_back(ge::DT_FLOAT);
    }

    gert::InferShapeContextFaker faker;
    faker.IrInstanceNum(irInstanceNum, {1});
    std::vector<gert::Shape*> shapePtrs;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        shapePtrs.push_back(&inputShapes[i]);
        faker.NodeInputTd(static_cast<int32_t>(i), dtypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    faker.InputShapes(shapePtrs);
    return faker.Build();
}

TEST_F(EmbeddingHashTableImportProtoTest, embedding_hash_table_import_infer_shape_counters_shorter_than_keys)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableImport")->infer_shape;

    auto holder = BuildImportInferShapeHolder({1, 1, 1, 1, 1, 1, 1}, {1}, {{4}, {2}, {4}, {8}});

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(EmbeddingHashTableImportProtoTest, embedding_hash_table_import_infer_shape_filter_flags_shorter_than_keys)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableImport")->infer_shape;

    auto holder = BuildImportInferShapeHolder({1, 1, 1, 1, 1, 1, 1}, {1}, {{4}, {4}, {2}, {8}});

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(EmbeddingHashTableImportProtoTest, embedding_hash_table_import_infer_shape_list_num_not_equal_table_num)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableImport")->infer_shape;

    auto holder = BuildImportInferShapeHolder({1, 1, 1, 1, 1, 1, 1}, {2}, {{2}, {2}, {2}, {4}});

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(EmbeddingHashTableImportProtoTest, embedding_hash_table_import_infer_shape_list_instance_num_mismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableImport")->infer_shape;

    auto holder = BuildImportInferShapeHolder({1, 1, 1, 2, 1, 1, 1}, {1}, {{2}, {2}, {2}, {2}, {4}});

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(EmbeddingHashTableImportProtoTest, embedding_hash_table_import_infer_shape_counters_longer_than_keys)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableImport")->infer_shape;

    auto holder = BuildImportInferShapeHolder({1, 1, 1, 1, 1, 1, 1}, {1}, {{3}, {5}, {3}, {6}});

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

} // namespace
