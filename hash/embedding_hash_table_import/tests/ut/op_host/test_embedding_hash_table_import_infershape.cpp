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
    static void SetUpTestCase()
    {
        std::cout << "EmbeddingHashTableImport Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EmbeddingHashTableImport Proto Test TearDown" << std::endl;
    }
};

//   pass cases
TEST_F(EmbeddingHashTableImportProtoTest, embedding_hash_table_import_infer_shape_test1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_95";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableImport")->infer_shape;

    gert::StorageShape table_handles_shape = {{1}, {1,}};
    gert::StorageShape embedding_dims_shape = {{1,}, {1,}};
    gert::StorageShape bucket_sizes_shape = {{1,}, {1,}};
    gert::StorageShape keys_shape = {{2,}, {2,}};
    gert::StorageShape counters_shape = {{2,}, {2,}};
    gert::StorageShape filter_flags_shape = {{2,}, {2,}};
    gert::StorageShape values_shape = {{10,}, {10,}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&table_handles_shape, &embedding_dims_shape, &bucket_sizes_shape,
                                  &keys_shape, &counters_shape, &filter_flags_shape, &values_shape})
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

} // namespace