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
 * \file test_embedding_hash_table_applyadamw_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/embedding_hash_table_apply_adam_w_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

namespace {
// ----------------EmbeddingHashTableApplyAdamW-------------------
class EmbeddingHashTableApplyAdamWProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EmbeddingHashTableApplyAdamW Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EmbeddingHashTableApplyAdamW Proto Test TearDown" << std::endl;
    }
};

//   pass cases
TEST_F(EmbeddingHashTableApplyAdamWProtoTest, embedding_hash_table_apply_adam_w_infer_shape_test1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910_95";
    optiCompilationInfo.soc_version = "Ascend910_95";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_95"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingHashTableApplyAdamW")->infer_shape;

    gert::StorageShape table_handle_shape = {{1}, {1}};
    gert::StorageShape keys_shape = {{2}, {2}};
    gert::StorageShape m_shape = {{2, 32}, {2, 32}};
    gert::StorageShape v_shape = {{2, 32}, {2, 32}};
    gert::StorageShape beta1_power_shape = {{1}, {1}};
    gert::StorageShape beta2_power_shape = {{1}, {1}};
    gert::StorageShape lr_shape = {{1}, {1}};
    gert::StorageShape weight_decay_shape = {{1}, {1}};
    gert::StorageShape beta1_shape = {{1}, {1}};
    gert::StorageShape beta2_shape = {{1}, {1}};
    gert::StorageShape epsilon_shape = {{1}, {1}};
    gert::StorageShape grad_shape = {{2, 32}, {2, 32}};
    gert::StorageShape max_grad_norm_shape = {{2, 32}, {2, 32}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&table_handle_shape, &keys_shape, &m_shape, &v_shape, &beta1_power_shape, &beta2_power_shape,
                           &lr_shape, &weight_decay_shape, &beta1_shape, &beta2_shape, &epsilon_shape, &grad_shape,
                           &max_grad_norm_shape})
                      .OutputShapes({&m_shape, &v_shape, &beta1_power_shape, &beta2_power_shape, &max_grad_norm_shape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"embedding_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                           {"bucket_size", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                           {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

} // namespace