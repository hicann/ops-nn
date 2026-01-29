/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file test_embedding_hash_table_lookup_or_insert_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/embedding_hash_table_lookup_or_insert_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

namespace {
// ----------------EmbeddingHashTableLookupOrInsert-------------------
class EmbeddingHashTableLookupOrInsertProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EmbeddingHashTableLookupOrInsert Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EmbeddingHashTableLookupOrInsert Proto Test TearDown" << std::endl;
    }
};

//   pass cases
TEST_F(EmbeddingHashTableLookupOrInsertProtoTest, embedding_hash_table_lookup_or_insert_infer_shape_test1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableLookupOrInsert")->infer_shape;

    gert::StorageShape hash_shape = {{5,}, {5,}};
    gert::StorageShape keys_shape = {{1,}, {1,}};
    gert::StorageShape values_shape = {{1, 2}, {1, 2}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&hash_shape, &keys_shape})
                      .OutputShapes({&values_shape})
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                        {"bucket_size", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
                        {"embedding_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                        {"filter_mode", Ops::NN::AnyValue::CreateFrom<std::string>("no_filter")},
                        {"filter_freq", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                        {"default_key_or_value", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                        {"default_key", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                        {"default_value", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                        {"filter_key_flag", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                        {"filter_key", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                      })
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

} // namespace